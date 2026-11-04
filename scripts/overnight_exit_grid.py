#!/usr/bin/env python3
"""overnight_exit_grid.py — broad cross-symbol exit-geometry optimization.

Sweeps the exit-geometry grid (atr_stop_mult x rr_ratio x max_hold) across EVERY
liquid 1h symbol in the DB, using the CPU fallback (byte-identical math to the
4090's gpu_server, see exit_geometry_sweep.py). Then aggregates across symbols
to find the geometry that is positive on the MOST symbols (robust, not just the
single best on one chart).

Output: results/_overnight_exit_grid.json
  {"per_symbol": {SYM: [combo...]}, "aggregate": [robust combos ranked]}
  Per-symbol results sorted by EV; aggregate keeps combos with N_trades>=10 on
  the symbol, records median EV across symbols and the count of symbols where
  the combo is positive.

This is compute-only: reads OHLC, writes a results JSON. No trading/system
changes, so it is safe to run unattended overnight.
"""
import json, os, sys
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.dirname(__file__))
try:
    from exit_geometry_sweep import sweep_single_close
except Exception:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from scripts.exit_geometry_sweep import sweep_single_close

import sqlite3

DB = "/home/hermes/BacktestingMCP/data/crypto.db"
OUT = "/home/hermes/BacktestingMCP/results/_overnight_exit_grid.json"

# The exit grid (matching what we run on the GPU)
STOPS = [2.0, 2.5, 3.0, 3.5, 4.0]
RRS = [1.0, 1.1, 1.2, 1.5, 2.0]
HOLDS = [0, 12, 24, 48]
MIN_TRADES_PER_COMBO = 10  # a combo needs >= this many trades to count as "tested"


def symbol_floor(sym):
    """Only OHLC-suffixed, non-USD samplers, with >=1000 1h bars."""
    if not sym.endswith("USDT"):
        return None
    if sym.endswith("USDTUSDT"):  # duplicated sampler (BTCUSDTUSDT) skip
        return None
    return sym


def main():
    conn = sqlite3.connect(DB)
    syms = conn.execute(
        """SELECT symbol, COUNT(*) FROM market_data WHERE timeframe='1h'
           GROUP BY symbol HAVING COUNT(*) >= 1000"""
    ).fetchall()
    conn.close()
    pool = [symbol_floor(s) for s, _ in syms if symbol_floor(s)]
    pool.sort(key=lambda s: dict(syms)[s], reverse=True)
    # cap pool so a single overnight run stays bounded (~120 symbols)
    pool = pool[:120]
    print(f"sweeping {len(pool)} symbols, grid {len(STOPS)}x{len(RRS)}x{len(HOLDS)} "
          f"({len(STOPS)*len(RRS)*len(HOLDS)} combos) = {len(pool)*len(STOPS)*len(RRS)*len(HOLDS)} sims")

    # re-open cleanly
    conn = sqlite3.connect(DB)
    per_symbol = {}
    agg = defaultdict(list)
    for sym in pool:
        rows = conn.execute(
            "SELECT close FROM market_data WHERE symbol=? AND timeframe='1h' ORDER BY timestamp",
            (sym,)).fetchall()
        close = [float(r[0]) for r in rows]
        if len(close) < 1000:
            continue
        # use LAST 1000 bars for a consistent regime-length window
        close = close[-1000:]
        res = sweep_single_close(close, STOPS, RRS, HOLDS)
        res["results"].sort(key=lambda x: -x["ev_r"])
        per_symbol[sym] = res["results"]
        for r in res["results"]:
            if r["trades"] >= MIN_TRADES_PER_COMBO:
                k = (r["atr_stop_mult"], r["rr_ratio"], r["max_hold_hours"])
                agg[k].append(r["ev_r"])
    conn.close()

    # aggregate ranking: combos that are positive on the most symbols, then by median EV
    aggregate = []
    for (s, r, h), evs in agg.items():
        n_pos = sum(1 for e in evs if e > 0)
        aggregate.append({
            "atr_stop_mult": s, "rr_ratio": r, "max_hold_hours": h,
            "n_symbols": len(evs), "n_pos_symbols": n_pos,
            "pos_frac": round(n_pos / len(evs), 3),
            "median_ev": round(float(np.median(evs)), 4),
            "mean_ev": round(float(np.mean(evs)), 4),
        })
    aggregate.sort(key=lambda x: (-x["pos_frac"], -x["median_ev"]))

    data = {"per_symbol": per_symbol, "aggregate": aggregate,
            "n_symbols": len(pool), "n_combos": len(aggregate)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nwrote {OUT}  ({len(pool)} symbols, {len(aggregate)} robust combos)")
    print("\nTOP ROBUST EXIT GEOMETRIES (positive on most symbols):")
    print(f"{'stop':>5} {'rr':>4} {'hold':>5} {'nSym':>5} {'posFrac':>7} {'medEV':>7} {'meanEV':>7}")
    for c in aggregate[:20]:
        print(f"{c['atr_stop_mult']:>5.1f} {c['rr_ratio']:>4.1f} {c['max_hold_hours']:>5} "
              f"{c['n_symbols']:>5} {c['pos_frac']:>7.1%} {c['median_ev']:>+7.3f} {c['mean_ev']:>+7.3f}")


if __name__ == "__main__":
    main()