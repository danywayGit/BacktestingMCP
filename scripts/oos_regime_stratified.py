#!/usr/bin/env python3
"""oos_regime_stratified.py — FINAL gate: regime-stratified OOS confirmation.

The small method-stable candidate set is 6.0, 12.0, 1.0, 6.2, 16.0. Each config
passed under its OWN signals (oos_signal_sweep.py) but we must also prove the
edge survives REGIMES — i.e. it stays positive in BOTH an up-trending window AND
a down-trending window, because that is the "less drawdown" property. A config
that wins only in bull months will still blow up a prop account in a drawdown.

Method:
  1. For each candidate, take its resolved signals (its own entry, direction,
     forward-return outcome).
  2. Label each symbol-window's regime by the 50-EMA-slope of the entering
     symbol's 1h market at signal time: REGIME=UP (EMA rising), DOWN (falling).
     (Regime label is exogenous — from price, not from the config.)
  3. Report per config: WR / median return / Wilson lower bound in UP-regime
     signals vs DOWN-regime signals, plus the fraction of symbols positive.
  4. PASS_FINAL = it wins (WR>=52% OR WilsonLo>0) in BOTH regimes AND in >=60%
     of its symbols — i.e. not a single-regime or single-symbol result.

Output: results/_oos_regime_stratified.json. Compute-only.
"""
import json, os, sqlite3, math
from collections import defaultdict

DB = "/home/hermes/BacktestingMCP/data/crypto.db"
OUT = "/home/hermes/BacktestingMCP/results/_oos_regime_stratified.json"
EMA_N = 50
CANDIDATES = ["6.0", "12.0", "1.0", "6.2", "16.0"]  # method-stable set from oos_signal_sweep
BREAKEVEN = 0.52
MIN_REGIME_TRADES = 25
MIN_SYMBOLS = 4


def wilson_lower(w, n, z=1.96):
    if n == 0:
        return 0.0
    p = w / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n))) / den
    return max(0.0, center - half)


def ema(arr):
    a = [float(x) for x in arr]
    if not a:
        return []
    alpha = 2.0 / (EMA_N + 1)
    out = [a[0]]
    for x in a[1:]:
        out.append(alpha * x + (1 - alpha) * out[-1])
    return out


def regime_at(conn, symbol, ts):
    """Return 1 if the symbol's 1h market was UP at ts (50-EMA rising), -1 DOWN, 0 unknown."""
    ohlc = symbol if symbol.endswith("USDT") else symbol + "USDT"
    rows = conn.execute(
        "SELECT timestamp, close FROM market_data WHERE symbol=? AND timeframe='1h' "
        "AND timestamp<=? ORDER BY timestamp DESC LIMIT 200", (ohlc, ts)).fetchall()
    if len(rows) < EMA_N + 1:
        return 0
    rows.reverse()
    closes = [r[1] for r in rows]
    e = ema(closes)
    # rising if last EMA >= mean of prior 4
    if len(e) < 5:
        return 0
    return 1.0 if e[-1] >= (sum(e[-5:-1]) / 4.0) else -1.0


def _ts(s):
    import time, datetime
    s = s.strip().replace('T', ' ').split('+')[0].split('Z')[0]
    return int(time.mktime(datetime.datetime.strptime(s[:19], '%Y-%m-%d %H:%M:%S').timetuple()))


def stats(items):
    if not items:
        return None
    n = len(items)
    w = sum(1 for x in items if x["fr"] > 0)
    frs = sorted(x["fr"] for x in items)
    n_sym = len({x["sym"] for x in items})
    return {"n": n, "wr": w / n, "wilson_l": wilson_lower(w, n),
            "med_r": round(frs[n // 2], 3), "total_r": round(sum(x["fr"] for x in items), 2),
            "n_sym": n_sym}


def main():
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        """SELECT config_version, COALESCE(symbol,pair) sym, entry_time, direction,
                  composite_score, forward_return_pct, outcome
           FROM edge_signals WHERE status='RESOLVED'
             AND outcome IN ('WIN','LOSS') AND forward_return_pct IS NOT NULL
             AND entry_time IS NOT NULL AND direction IS NOT NULL""").fetchall()
    conn.close()
    by_cfg = defaultdict(list)
    for cv, sym, et, d, sc, fr, oc in rows:
        try:
            fr = float(fr)
        except (TypeError, ValueError):
            continue
        by_cfg[str(cv)].append({"sym": str(sym), "time": str(et),
                                "dir": (d or '').upper(), "fr": fr})

    results = {}
    conn = sqlite3.connect(DB)
    for cv in CANDIDATES:
        recs = by_cfg.get(cv, [])
        recs.sort(key=lambda x: _ts(x["time"]))
        # label regime per signal (exogenous from market)
        up, down = [], []
        for r in recs:
            rg = regime_at(conn, r["sym"], _ts(r["time"]))
            r["regime"] = rg
            if rg > 0:
                up.append(r)
            elif rg < 0:
                down.append(r)
        up_s = stats(up); down_s = stats(down)
        # total across both regimes
        all_s = stats(up + down)
        # fraction of symbols with net positive forward return
        sym_net = defaultdict(float); sym_n = defaultdict(int)
        for r in (up + down):
            sym_net[r["sym"]] += r["fr"]; sym_n[r["sym"]] += 1
        sym_pos = sum(1 for s, v in sym_net.items() if v > 0)
        n_sym = len(sym_net)

        pass_final = False
        reason = []
        if up_s and down_s:
            up_ok = up_s["wr"] >= BREAKEVEN or up_s["wilson_l"] > 0
            down_ok = down_s["wr"] >= BREAKEVEN or down_s["wilson_l"] > 0
            enough = up_s["n"] >= MIN_REGIME_TRADES and down_s["n"] >= MIN_REGIME_TRADES
            sym_ok = (sym_pos / n_sym) >= 0.60 if n_sym else False
            pass_final = up_ok and down_ok and enough and sym_ok
            if not up_ok:
                reason.append(f"UP wr={up_s['wr']:.0%}")
            if not down_ok:
                reason.append(f"DOWN wr={down_s['wr']:.0%}")
            if not enough:
                reason.append(f"regime n<{MIN_REGIME_TRADES}")
            if not sym_ok:
                reason.append(f"syms+ve {sym_pos}/{n_sym}")
        else:
            reason.append("missing regime (no market data)")

        results[cv] = {
            "status": "PASS_FINAL" if pass_final else "REJECT",
            "reasons": reason,
            "up": up_s, "down": down_s, "overall": all_s,
            "symbols_positive": sym_pos, "n_symbols": n_sym,
        }
    conn.close()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)

    print(f"wrote {OUT}\n")
    print("=== FINAL REGIME-STRATIFIED GATE (must win in BOTH up & down windows) ===")
    for cv in CANDIDATES:
        r = results[cv]
        up = r["up"]; dn = r["down"]
        uph = f"n{up['n']} WR{up['wr']:.0%} wil{up['wilson_l']:.0%} med{up['med_r']:+.2f}" if up else "n/a"
        dnh = f"n{dn['n']} WR{dn['wr']:.0%} wil{dn['wilson_l']:.0%} med{dn['med_r']:+.2f}" if dn else "n/a"
        print(f"\n{cv}: {r['status']}  syms+ve {r['symbols_positive']}/{r['n_symbols']}")
        print(f"   UP-regime   : {uph}")
        print(f"   DOWN-regime : {dnh}")
        if r["reasons"]:
            print(f"   -> reject if: {', '.join(r['reasons'])}")


if __name__ == "__main__":
    main()