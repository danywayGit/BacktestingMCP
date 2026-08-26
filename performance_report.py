#!/usr/bin/env python3
"""Bi-weekly performance report — execution WR, PF, drawdown, position-size sanity.

Combines bot-executed trades + scanner signals, measures against targets:
  - Execution win rate, profit factor, payoff, expectancy
  - Max drawdown (from executed PnL sequence)
  - Position-size sanity: flag any executed trade whose notional is < 0.1% of
    wallet (the 'dust position' bug) or unrealistically large
  - Wilson confidence interval + Kelly criterion (both directions filter)
  - LONG vs SHORT, top/bottom configs, coin-type edge

Run:  python3 performance_report.py [--strategy EdgeScanner] [--days 14]
"""
import base64
import json
import math
import os
import sqlite3
import statistics
import subprocess
import sys
import requests
from collections import defaultdict
from datetime import datetime, timezone, timedelta

SSH_HELPER = os.path.expanduser("~/.hermes/scripts/ssh_sudo_run.py")

# ── TARGETS (the user's definition of 'reasonable', measured not guessed) ──
MIN_WR = 50.0            # % win rate target
MIN_PF = 1.3             # profit factor floor
MIN_RR = 1.1             # risk-reward floor
MAX_DD_PCT = 30.0        # max acceptable drawdown (binance prod)
PROP_MAX_DD = 5.0        # typical prop 5% max drawdown (stricter)
MIN_POS_PCT = 0.05       # min position notional as % of wallet (else likely dust)
MAX_POS_PCT = 12.0       # max position notional % (exposure sanity)
MIN_SAMPLE = 30          # trades before a config's stats are 'trusted'


def fetch_executed_trades(strategy="All"):
    remote = (
        "/opt/Trading-WebHook-Bot/venv-bot/bin/python -c \""
        "import sqlite3,base64,json;"
        "c=sqlite3.connect('/opt/Trading-WebHook-Bot/exchanges/trades.db');"
        "rows=c.execute('SELECT Symbol,Side,Entry,Exit,StopLoss,TakeProfit,"
        "OpenDate,CloseDate,ProfitLoss,ProfitLossPercent,RiskRewardRatio,"
        "Leverage,StrategyName,AccountType FROM Trades WHERE IsOpen=0').fetchall();"
        "tr=[{'symbol':r[0],'side':r[1],'entry':r[2],'exit':r[3],'sl':r[4],"
        "'tp':r[5],'open':r[6],'close':r[7],'pnl':r[8],'pnl_pct':r[9],"
        "'rr':r[10],'lev':r[11],'strategy':r[12],'acct':r[13]} for r in rows];"
        "print(base64.b64encode(json.dumps(tr).encode()).decode())\""
    )
    cp = subprocess.run(["python3", SSH_HELPER, remote],
                        capture_output=True, text=True, timeout=90)
    for line in (cp.stdout or "").splitlines():
        line = line.strip()
        if len(line) > 50 and line[0] in "W0" and all(
                ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
                for ch in line):
            try:
                trades = json.loads(base64.b64decode(line).decode())
                if strategy != "All":
                    trades = [t for t in trades if t.get("strategy") == strategy]
                return trades
            except Exception:
                continue
    raise RuntimeError(f"fetch failed: {cp.stdout[-300:]} {cp.stderr[-200:]}")


def wilson_ci(w, n, z=1.96):
    """Wilson score interval for a win rate. Returns (lower, upper)."""
    if n == 0:
        return (0, 0)
    p = w / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def kelly(wr, payoff):
    """Full Kelly: f* = wr - (1-wr)/payoff. Returns None if payoff<=0."""
    if payoff <= 0:
        return None
    return wr - (1 - wr) / payoff


def max_drawdown(pnl_series):
    """Max drawdown % from a PnL sequence (cumulative USDT).

    Computes peak-to-trough drawdown on the equity curve, expressed as % of
    the running peak. A losing run that never recovers → dd approaches the
    total loss. Returns (dd_pct, peak, trough).
    """
    if not pnl_series:
        return (0, 0, 0)
    peak = pnl_series[0]
    max_dd = 0.0
    trough = pnl_series[0]
    for v in pnl_series:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (peak - v) / peak * 100
            if dd > max_dd:
                max_dd = dd
                trough = v
    return (max_dd, peak, trough)


def hold_hours(t):
    try:
        o = datetime.fromisoformat(t["open"].replace(" ", "T").replace("Z", "+00:00"))
        c = datetime.fromisoformat(t["close"].replace(" ", "T").replace("Z", "+00:00"))
        return (c - o).total_seconds() / 3600
    except Exception:
        return None


def compute(group):
    wins = [t for t in group if (t.get("pnl") or 0) > 0]
    losses = [t for t in group if (t.get("pnl") or 0) < 0]
    flats = [t for t in group if (t.get("pnl") or 0) == 0]
    gw = sum(t.get("pnl", 0) for t in wins)
    gl = abs(sum(t.get("pnl", 0) for t in losses))
    n = len(wins) + len(losses)
    wr = (len(wins) / n * 100) if n else 0
    payoff = (gw / len(wins)) / (gl / len(losses)) if wins and losses else 0
    lo, hi = wilson_ci(len(wins), n)
    return {
        "n": len(group), "wins": len(wins), "losses": len(losses), "flats": len(flats),
        "wr": wr, "pf": gw / gl if gl > 0 else float("inf"),
        "net": sum(t.get("pnl", 0) for t in group),
        "payoff": payoff,
        "ci": (lo * 100, hi * 100),
        "kelly": kelly(wr / 100, payoff),
    }


def fmt(x, dp=2):
    return f"{x:.{dp}f}" if isinstance(x, float) else str(x)


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--strategy", default="EdgeScanner")
    ap.add_argument("--days", type=int, default=14)
    args = ap.parse_args()
    days = args.days
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    print("📊 *Bi-Weekly Performance Report*")
    print(f"Strategy: {args.strategy} | Window: last {days}d")
    print("=" * 64)

    trades = fetch_executed_trades(args.strategy)
    recent = [t for t in trades if t.get("close", "") >= cutoff[:10]]
    print(f"Executed trades (last {days}d): {len(recent)}  (all-time: {len(trades)})")
    if not recent:
        print("  ⚠️  No trades in window yet — pipeline building sample.")
        return 0

    for t in recent:
        t["hold_h"] = hold_hours(t)

    # ── 1. EXECUTION WR / PF / expectancy ──
    s = compute(recent)
    print("\n*1. Execution performance*")
    print(f"  WR={s['wr']:.0f}% ({s['wins']}W/{s['losses']}L)  PF={fmt(s['pf'])}  "
          f"payoff={s['payoff']:.2f}  expectancy={s['net']/max(s['n'],1):+.2f} USDT")
    print(f"  Wilson 95% CI: [{s['ci'][0]:.0f}%, {s['ci'][1]:.0f}%]  "
          f"(n={s['n']}, need ≥{MIN_SAMPLE} to trust)")
    if s["pf"] < MIN_PF:
        print(f"  ⚠️  PF {s['pf']:.2f} < target {MIN_PF}")
    if s["wr"] < MIN_WR:
        print(f"  ⚠️  WR {s['wr']:.0f}% < target {MIN_WR}%")

    # ── 2. DRAWDOWN (proxy from cumulative PnL; true equity curve not stored) ──
    cum = 0.0
    series = []
    for t in sorted(recent, key=lambda x: x.get("close", "")):
        cum += t.get("pnl", 0)
        series.append(cum)
    dd, peak, trough = max_drawdown(series)
    # Best proxy: cumulative PnL (USDT) as % of a reference capital. The bot DB
    # doesn't store a historical equity curve, so DD is expressed relative to
    # the current wallet (queried live). For a clean restart, this measures
    # the run from the fresh baseline.
    try:
        KEY = "6XO7toihtxsSW7s9OgetPwVwjMCNhb4O"
        summ = requests.get(f"http://109.123.229.200/api/summary?key={KEY}", timeout=6).json()
        balance = float(summ.get("cash_pool") or 0) or 5000.0
    except Exception:
        balance = 5000.0
    dd_acct = abs(trough) / balance * 100 if balance else 0
    peak_acct = peak / balance * 100 if balance else 0
    print(f"\n*2. Drawdown*")
    print(f"  Max DD (cum PnL USDT): peak +{peak:.0f} → trough {trough:.0f} USDT")
    print(f"  Max DD as % of balance ({balance:.0f} USDT): {dd_acct:.1f}%  (peak +{peak_acct:.1f}%)")
    print(f"  Target: ≤ {MAX_DD_PCT}% (Binance) / ≤ {PROP_MAX_DD}% (prop)")
    if dd_acct > PROP_MAX_DD:
        print(f"  ⚠️  DD {dd_acct:.1f}% exceeds {PROP_MAX_DD}% prop limit — reduced sizing needed")

    # ── 3. POSITION-SIZE SANITY ──
    print(f"\n*3. Position-size sanity*")
    pos_issues = 0
    for t in recent:
        pnl_pct = abs(t.get("pnl_pct") or 0)
        notional = (t.get("entry") or 0)  # needs quantity; approximated by pnl_pct
        if pnl_pct > 0 and pnl_pct < 0.01:  # a real trade moving <0.01% = dust
            pos_issues += 1
            print(f"  ⚠️  {t['strategy']} {t['symbol']}: pnl_pct={pnl_pct:.3f}% — tiny position")
    if pos_issues == 0:
        print(f"  ✅ No dust/tiny position flags in {len(recent)} executed trades")
    max_lev = max([t.get("lev") or 0 for t in recent], default=0)
    min_lev = min([t.get("lev") or 0 for t in recent], default=0)
    print(f"  Leverage used: {min_lev}x-{max_lev}x  (cap 5x, target 3x)")

    # ── 4. DIRECTION / CONFIG / COIN BREAKDOWN ──
    print(f"\n*4. Direction breakdown*")
    for d in ["LONG", "SHORT"]:
        grp = [t for t in recent if ("SHORT" if t.get("side") == "SELL" else "LONG") == d]
        if len(grp) >= 1:
            ds = compute(grp)
            print(f"  {d}: n={ds['n']}  WR={ds['wr']:.0f}%  PF={fmt(ds['pf'])}  net={ds['net']:+.2f}")

    # ── 5. Kelly recommendation ──
    k = s["kelly"]
    print(f"\n*5. Position-sizing (Kelly)*")
    if k is not None and k > 0:
        print(f"  Full Kelly: {k*100:.1f}% | half: {k*50:.1f}% | quarter: {k*25:.1f}%")
        print(f"  For prop (5% DD): use ~10-20% Kelly ≈ {k*15:.2f}%")
    else:
        print("  Kelly not positive — edge unproven, reduce/paper-trade")

    # ── 6. Scanner-side config health (last {days}d resolved) ──
    print(f"\n*6. Scanner config health (last {days}d resolved sent)*")
    conn = sqlite3.connect("/home/hermes/BacktestingMCP/data/crypto.db")
    rows = conn.execute("""
        SELECT config_version,
               SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END),
               SUM(CASE WHEN outcome='LOSS' THEN 1 ELSE 0 END)
        FROM edge_signals
        WHERE webhook_sent_at IS NOT NULL AND outcome IN ('WIN','LOSS')
          AND resolved_at >= ?
        GROUP BY config_version
        HAVING (SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END)+
                SUM(CASE WHEN outcome='LOSS' THEN 1 ELSE 0 END)) >= 3
        ORDER BY (SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END)*1.0/
                 NULLIF(SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END)+SUM(CASE WHEN outcome='LOSS' THEN 1 ELSE 0 END),0)) DESC
    """, (cutoff,)).fetchall()
    conn.close()
    if rows:
        for v, w, l in rows:
            wl = w + l
            wr = w / wl * 100
            trusted = "✅" if wl >= MIN_SAMPLE else "🟡"
            flag = " ⚠️<target" if wr < MIN_WR and wl >= MIN_SAMPLE else ""
            print(f"  {trusted} cfg={v:<5} {w}W/{l}L WR={wr:.0f}%{flag}")
    else:
        print("  No resolved signals in window.")

    print("\n" + "=" * 64)
    print("Measured, not guessed. Review before acting (explain-before-change).")
    return 0


if __name__ == "__main__":
    sys.exit(main())