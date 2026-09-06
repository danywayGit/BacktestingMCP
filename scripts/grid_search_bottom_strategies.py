#!/usr/bin/env python3
"""Grid-search bottom-market strategies on local 1h OHLC data.

Tests parameter variations across strategies that work in the engine AND are
candidate fits for a bottoming/recovering market:
  - unusual_volume_breakout, resistance_breakout, new_local_high_breakout (momentum/breakout)
  - bollinger_bands, macd (mean-reversion / pullback)
  - swing5_keltner_breakout, vp1_volume_profile_breakout, swing2_bb_squeeze

Runs on FULL period + 10 bottom/near-bottom sub-periods across 4 majors.
"""
import sys, os, time, itertools, csv, json
sys.path.insert(0, '/home/hermes/BacktestingMCP')
os.chdir('/home/hermes/BacktestingMCP')
from datetime import datetime, timezone, timedelta

from src.core.backtesting_engine import engine
from config.settings import TimeFrame
from src.strategies.templates import get_strategy_class

OUT = "research/bottom_strategy_grid_results.csv"

STRATEGIES = {
    "unusual_volume_breakout": {
        "volume_lookback": [10, 20, 30],
        "volume_multiplier": [1.5, 2.0, 3.0],
        "breakout_lookback": [10, 20, 40],
    },
    "resistance_breakout": {
        "resistance_lookback": [30, 50, 80],
        "level_tolerance": [0.01, 0.02],
        "min_touches": [2, 3],
    },
    "new_local_high_breakout": {
        "local_high_lookback": [20, 30, 50],
        "min_relative_volume": [1.0, 1.5, 2.0],
    },
    "bollinger_bands": {
        "bb_period": [10, 20, 30],
        "bb_std": [1.5, 2.0, 2.5],
    },
    "macd": {
        "macd_fast": [8, 12, 16],
        "macd_slow": [21, 26, 34],
        "macd_signal": [7, 9, 12],
    },
    "swing5_keltner_breakout": {
        "kc_length": [15, 20, 30],
        "kc_mult": [1.5, 2.0, 2.5],
    },
    "vp1_volume_profile_breakout": {
        "profile_lookback": [100, 200, 300],
    },
    "swing2_bb_squeeze": {
        "bb_length": [15, 20, 30],
        "bb_mult": [1.5, 2.0, 2.5],
    },
}

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
FULL = (datetime(2020,1,1,tzinfo=timezone.utc), datetime(2026,8,15,tzinfo=timezone.utc))
BOTTOM_DATES = [
    datetime(2020,4,1,tzinfo=timezone.utc), datetime(2021,1,22,tzinfo=timezone.utc),
    datetime(2021,6,22,tzinfo=timezone.utc), datetime(2021,9,21,tzinfo=timezone.utc),
    datetime(2022,1,24,tzinfo=timezone.utc), datetime(2022,6,18,tzinfo=timezone.utc),
    datetime(2022,11,21,tzinfo=timezone.utc), datetime(2024,8,5,tzinfo=timezone.utc),
    datetime(2025,4,7,tzinfo=timezone.utc), datetime(2026,2,5,tzinfo=timezone.utc),
]
def bottom_windows(b):
    return (b - timedelta(days=21), b + timedelta(days=45))

def run_one(cls, symbol, start, end, params):
    base = {"return_pct": float('nan'), "sharpe": float('nan'), "trades": 0,
            "winrate": float('nan'), "pf": float('nan'), "maxdd": float('nan'),
            "buy_hold": 0, "error": ""}
    try:
        res = engine.run_backtest(cls, symbol, TimeFrame.H1, start, end, cash=100000, parameters=params)
        s = res.stats
        base.update({"return_pct": s.get("total_return_pct", float('nan')), "sharpe": s.get("sharpe_ratio", float('nan')),
                "trades": s.get("num_trades", 0), "winrate": s.get("win_rate_pct", float('nan')),
                "pf": s.get("profit_factor", float('nan')), "maxdd": s.get("max_drawdown_pct", float('nan')),
                "buy_hold": s.get("buy_hold_return_pct", 0)})
    except Exception as e:
        base["error"] = str(e)[:80]
    return base

def iter_grid(grid):
    for combo in itertools.product(*grid.values()):
        yield dict(zip(grid.keys(), combo))

def main(mode):
    total_combos = sum(len(list(iter_grid(g))) for g in STRATEGIES.values())
    rows = []
    done = 0
    periods = [("FULL", FULL)] if mode in ("both","full") else []
    if mode in ("both","bottom"):
        periods += [(b.strftime("%Y-%m-%d"), bottom_windows(b)) for b in BOTTOM_DATES]
    t0 = time.time()
    for sname, grid in STRATEGIES.items():
        cls = get_strategy_class(sname)
        combos = list(iter_grid(grid))
        for params in combos:
            for pname,(start,end) in periods:
                for sym in SYMBOLS:
                    r = run_one(cls, sym, start, end, params)
                    rows.append({"strategy":sname, **params, "period":pname, "symbol":sym, **r})
            done += 1
            if done % 10 == 0:
                el = time.time()-t0
                print(f"  {done}/{total_combos} elapsed {el:.0f}s", flush=True)
                _write(rows)
    _write(rows)
    print(f"DONE {len(rows)} rows -> {OUT}", flush=True)

def _write(rows):
    if not rows: return
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with open(OUT,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction="ignore"); w.writeheader(); w.writerows(rows)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "both")