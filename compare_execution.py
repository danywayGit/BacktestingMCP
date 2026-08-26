#!/usr/bin/env python3
"""Compare & optimize: backtested (scanner) vs executed (bot) strategies.

Fetches executed trades from the bot server (SSH, ssh_sudo_run.py) and
resolved sent signals from the local scanner DB, then computes per-strategy /
direction / coin-type stats for BOTH sides and reports deviations:

  - Win rate, profit factor, avg win/loss %, expectancy
  - LONG vs SHORT breakdown (is a strategy better one direction?)
  - Coin-type subset analysis (LAYER1/DEFI/AI/MEME...)
  - Hold-time analysis: backtest time-to-resolve vs actual execution hold
    (wins avg 4h in backtest but open 9h live = exit too late, etc.)
  - Realized R:R vs planned R:R, entry slippage

Output: a structured deviation report with actionable recommendations.

Run:  python3 compare_execution.py [--min-trades 5]
"""
import argparse
import base64
import json
import os
import sqlite3
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone

DB_PATH = "data/crypto.db"
SSH_HELPER = os.path.expanduser("~/.hermes/scripts/ssh_sudo_run.py")
BOT_DB = "/opt/Trading-WebHook-Bot/exchanges/trades.db"

# ── Data fetch ──────────────────────────────────────────────────────────

def fetch_executed_trades() -> list:
    """Fetch closed EdgeScanner trades from the bot server via SSH (base64)."""
    remote = (
        "cd /opt/Trading-WebHook-Bot && "
        f"{BOT_DB} >/dev/null 2>&1; "  # no-op (keeps quoting simple)
        f"/opt/Trading-WebHook-Bot/venv-bot/bin/python -c \""
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
                return json.loads(base64.b64decode(line).decode())
            except Exception:
                continue
    raise RuntimeError(f"Could not fetch executed trades: {cp.stdout[-300:]} {cp.stderr[-200:]}")


def fetch_scanner_signals() -> list:
    """Load resolved sent signals from the local scanner DB."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT symbol, direction, config_version, composite_score,
               entry_price, stop_price, target_price, created_at, resolved_at,
               outcome, forward_return_pct, time_to_resolve_hours, coin_type,
               webhook_sent_at
        FROM edge_signals
        WHERE webhook_sent_at IS NOT NULL AND outcome IN ('WIN','LOSS')
    """).fetchall()
    conn.close()
    sigs = []
    for r in rows:
        sym, direction, cfg, score, entry, stop, tgt, created, resolved, \
            outcome, ret, ttr, ctype, sent = r
        sigs.append({
            "symbol": sym.replace("USDT", "").upper(),
            "direction": direction,
            "config": cfg,
            "score": score,
            "entry": entry, "stop": stop, "target": tgt,
            "created": created, "resolved": resolved,
            "outcome": outcome, "ret_pct": ret,
            "time_to_resolve_h": ttr,
            "coin_type": (ctype or "OTHER").upper(),
            "sent_at": sent,
        })
    return sigs


# ── Stats ───────────────────────────────────────────────────────────────

def stats(trades):
    """Compute summary stats for a list of {pnl, hold_h, ...} dicts."""
    if not trades:
        return {"n": 0}
    wins = [t for t in trades if (t.get("pnl") or 0) > 0]
    losses = [t for t in trades if (t.get("pnl") or 0) < 0]
    flats = [t for t in trades if (t.get("pnl") or 0) == 0]
    gw = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses))
    n = len(wins) + len(losses)
    holds = [t.get("hold_h", 0) for t in trades if t.get("hold_h")]
    win_holds = [t["hold_h"] for t in wins if t.get("hold_h")]
    loss_holds = [t["hold_h"] for t in losses if t.get("hold_h")]
    return {
        "n": len(trades),
        "wins": len(wins), "losses": len(losses), "flats": len(flats),
        "wr": len(wins) / n * 100 if n else 0,
        "pf": gw / gl if gl > 0 else float("inf"),
        "net": sum(t.get("pnl", 0) for t in trades),
        "avg_win": gw / len(wins) if wins else 0,
        "avg_loss": -gl / len(losses) if losses else 0,
        "payoff": (gw / len(wins)) / (gl / len(losses)) if wins and losses else 0,
        "expectancy": (gw - gl) / n if n else 0,
        "avg_hold_h": statistics.mean(holds) if holds else 0,
        "avg_win_hold_h": statistics.mean(win_holds) if win_holds else 0,
        "avg_loss_hold_h": statistics.mean(loss_holds) if loss_holds else 0,
    }


def hold_hours(t):
    """Compute hold hours from open/close ISO strings."""
    try:
        o = datetime.fromisoformat(t["open"].replace(" ", "T").replace("Z", "+00:00"))
        c = datetime.fromisoformat(t["close"].replace(" ", "T").replace("Z", "+00:00"))
        return round((c - o).total_seconds() / 3600, 2)
    except Exception:
        return None


# ── Report ──────────────────────────────────────────────────────────────

def fmt_num(x, dp=2):
    if x == float("inf"):
        return "inf"
    if isinstance(x, float):
        return f"{x:.{dp}f}"
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-trades", type=int, default=5,
                    help="Minimum trades for a group to be reported")
    ap.add_argument("--strategy", default="EdgeScanner",
                    help="Strategy to compare (EdgeScanner, ManualTrading, DCA_Spot, or ALL)")
    args = ap.parse_args()
    STRAT = args.strategy

    print("=" * 72)
    print(f"COMPARE & OPTIMIZE — backtested (scanner) vs executed (bot)")
    print(f"Strategy: {STRAT}")
    print("=" * 72)

    # Fetch data
    print("\n[1/4] Fetching data...")
    exec_trades = fetch_executed_trades()
    if STRAT != "ALL":
        exec_trades = [t for t in exec_trades if t["strategy"] == STRAT]
    print(f"  Executed ({STRAT}): {len(exec_trades)} closed trades")
    for t in exec_trades:
        t["hold_h"] = hold_hours(t)
        t["symbol"] = t["symbol"].replace("USDT", "").upper()
        t["direction"] = "LONG" if t["side"] == "BUY" else "SHORT"

    scanner = fetch_scanner_signals()
    print(f"  Backtest (scanner): {len(scanner)} resolved sent signals")
    # NOTE: scanner signals aren't 1:1 strategy-tagged (they carry config_version
    # like 1.4, 22.0 which all feed EdgeScanner). We use ALL resolved sent
    # signals as the backtest baseline for EdgeScanner; for other strategies the
    # executed side is compared against the scanner baseline with that caveat.

    # [2] Overall + direction comparison
    print("\n[2/4] OVERALL & DIRECTION COMPARISON")
    for label, group in [
        ("ALL", exec_trades), ("LONG", [t for t in exec_trades if t["direction"] == "LONG"]),
        ("SHORT", [t for t in exec_trades if t["direction"] == "SHORT"]),
    ]:
        s = stats(group)
        if s["n"] < args.min_trades:
            continue
        print(f"\n  ── Executed {label} ({s['n']} trades) ──")
        print(f"     WR={s['wr']:.0f}%  PF={fmt_num(s['pf'])}  Net={s['net']:+.2f} USDT")
        print(f"     avgWin={s['avg_win']:+.2f}  avgLoss={s['avg_loss']:.2f}  payoff={s['payoff']:.2f}")
        print(f"     hold: all={s['avg_hold_h']:.1f}h  win={s['avg_win_hold_h']:.1f}h  loss={s['avg_loss_hold_h']:.1f}h")

    # Scanner-side direction comparison (for reference)
    for label, group in [
        ("LONG", [t for t in scanner if t["direction"] == "LONG"]),
        ("SHORT", [t for t in scanner if t["direction"] == "SHORT"]),
    ]:
        s = stats([{"pnl": t["ret_pct"], "hold_h": t["time_to_resolve_h"]} for t in group])
        if s["n"] < args.min_trades:
            continue
        print(f"  ── Backtest {label} ({s['n']} sig) ──")
        print(f"     WR={s['wr']:.0f}%  PF={fmt_num(s['pf'])}  avgRet={s['net']/s['n']:+.2f}%/sig")
        print(f"     timeToResolve: win={s['avg_win_hold_h']:.1f}h  loss={s['avg_loss_hold_h']:.1f}h")

    # [3] Coin-type subsets (executed + backtest)
    print("\n[3/4] COIN-TYPE SUBSET ANALYSIS")
    ctypes = defaultdict(list)
    for t in exec_trades:
        # Map executed symbols to coin_type via scanner signals
        pass  # done below after building map
    scanner_by_sym = {}
    for t in scanner:
        scanner_by_sym.setdefault(t["symbol"], []).append(t)

    def coin_type_of(sym):
        sigs = scanner_by_sym.get(sym, [])
        if sigs:
            return sigs[0]["coin_type"]
        return "UNKNOWN"

    for t in exec_trades:
        t["coin_type"] = coin_type_of(t["symbol"])

    print("  ── Executed by coin type (>= %d trades) ──" % args.min_trades)
    ct_groups = defaultdict(list)
    for t in exec_trades:
        ct_groups[t["coin_type"]].append(t)
    for ct, grp in sorted(ct_groups.items(), key=lambda kv: -len(kv[1])):
        s = stats(grp)
        if s["n"] < args.min_trades:
            continue
        print(f"  {ct:<8} n={s['n']:<3} WR={s['wr']:.0f}%  PF={fmt_num(s['pf'])}  "
              f"Net={s['net']:+.2f}  winHold={s['avg_win_hold_h']:.1f}h  lossHold={s['avg_loss_hold_h']:.1f}h")

    print("\n  ── Backtest by coin type (>= %d sig) ──" % args.min_trades)
    bt_groups = defaultdict(list)
    for t in scanner:
        bt_groups[t["coin_type"]].append(t)
    for ct, grp in sorted(bt_groups.items(), key=lambda kv: -len(kv[1])):
        s = stats([{"pnl": t["ret_pct"], "hold_h": t["time_to_resolve_h"]} for t in grp])
        if s["n"] < args.min_trades:
            continue
        print(f"  {ct:<8} n={s['n']:<3} WR={s['wr']:.0f}%  PF={fmt_num(s['pf'])}  "
              f"avgRet={s['net']/s['n']:+.2f}%  winTTR={s['avg_win_hold_h']:.1f}h  lossTTR={s['avg_loss_hold_h']:.1f}h")

    # [4] Deviations + recommendations
    print("\n[4/4] DEVIATIONS & RECOMMENDATIONS")
    bt_all = stats([{"pnl": t["ret_pct"], "hold_h": t["time_to_resolve_h"]} for t in scanner])
    ex_all = stats(exec_trades)
    if bt_all["n"] >= args.min_trades and ex_all["n"] >= args.min_trades:
        print(f"  • WR deviation: backtest {bt_all['wr']:.0f}% vs executed {ex_all['wr']:.0f}% "
              f"({ex_all['wr']-bt_all['wr']:+.0f} pts)")
        if ex_all["wr"] < bt_all["wr"] - 10:
            print("    → Execution underperforms backtest WR — check slippage/entry timing")
        if bt_all["avg_win_hold_h"] and ex_all["avg_win_hold_h"]:
            diff = ex_all["avg_win_hold_h"] - bt_all["avg_win_hold_h"]
            print(f"  • WIN hold deviation: backtest resolves wins in {bt_all['avg_win_hold_h']:.1f}h "
                  f"but executed holds {ex_all['avg_win_hold_h']:.1f}h ({diff:+.1f}h)")
            if diff > 2:
                print("    → Trades held TOO LONG on winners — consider earlier TP/time-stop")
            if diff < -2:
                print("    → Trades closed EARLIER than backtest — could be cutting winners short")
        if bt_all["avg_loss_hold_h"] and ex_all["avg_loss_hold_h"]:
            diff = ex_all["avg_loss_hold_h"] - bt_all["avg_loss_hold_h"]
            print(f"  • LOSS hold deviation: backtest resolves losses in {bt_all['avg_loss_hold_h']:.1f}h "
                  f"but executed holds {ex_all['avg_loss_hold_h']:.1f}h ({diff:+.1f}h)")
            if diff > 2:
                print("    → Losers held TOO LONG — tighten SL or add time-stop")

    # Direction preference
    bt_long = stats([{"pnl": t["ret_pct"]} for t in scanner if t["direction"] == "LONG"])
    bt_short = stats([{"pnl": t["ret_pct"]} for t in scanner if t["direction"] == "SHORT"])
    if bt_long["n"] >= args.min_trades and bt_short["n"] >= args.min_trades:
        print(f"  • Direction edge (backtest): LONG WR={bt_long['wr']:.0f}% PF={fmt_num(bt_long['pf'])} "
              f"| SHORT WR={bt_short['wr']:.0f}% PF={fmt_num(bt_short['pf'])}")
        if bt_long["pf"] > bt_short["pf"] * 1.3:
            print("    → LONG significantly stronger — consider LONG-only filter")
        elif bt_short["pf"] > bt_long["pf"] * 1.3:
            print("    → SHORT significantly stronger — consider SHORT-only filter")

    # Coin-type winners/losers
    print("  • Coin-type edge (backtest, PF by type):")
    for ct, grp in sorted(bt_groups.items(), key=lambda kv: -stats([{"pnl": t["ret_pct"]} for t in kv[1]])["pf"]):
        s = stats([{"pnl": t["ret_pct"]} for t in grp])
        if s["n"] < args.min_trades:
            continue
        print(f"    {ct:<8} PF={fmt_num(s['pf'])}  n={s['n']}")

    print("\n" + "=" * 72)
    print("Done. Adjustments should be reviewed before applying (explain-before-change).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
