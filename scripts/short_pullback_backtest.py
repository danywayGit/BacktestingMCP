#!/usr/bin/env python3
"""SHORT-only backtest over the recent surge+pullback (Aug-Sep 2026).

Map the top-5 long-adaptable edge configs to engine strategies and run each
SHORT-only across the window covering the +24% rally and its pullback.
Tests both the full window and the peak/pullback sub-window.
"""
import sys, os, datetime, itertools
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.backtesting_engine import engine
from config.settings import TimeFrame
from src.core.backtesting_engine import Direction
from src.strategies.templates import get_strategy_class

ROWS = []
def run_one(name, symbol, start, end, direction, params, tag, tf=TimeFrame.H1):
    cls = get_strategy_class(name)
    row = dict(strategy=name, symbol=symbol, tag=tag, params=str(params), error="")
    try:
        p = dict(params or {})
        p["direction"] = direction
        res = engine.run_backtest(cls, symbol, tf, start, end, cash=100000, parameters=p)
        s = res.stats
        row.update(n=len(res.trades), wr=s.get("win_rate_pct"), pf=s.get("profit_factor"),
                   ret=s.get("total_return_pct"), ret_abs=s.get("final_equity",0)-100000,
                   avg=s.get("avg_trade_pct"), exp=s.get("expectancy_pct"),
                   dd=s.get("max_drawdown_pct"), sharpe=s.get("sharpe_ratio"),
                   sqn=s.get("sqn"))
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {str(e)[:80]}"
    ROWS.append(row)

# Recent move: BTC +24% Jul17->Aug27 peak 80.2k, pullback to ~78k by Aug31.
# Full window spans rally + pullback; peak window isolates topping/decline.
WINDOWS = {
    "full_Jul18_Aug31": (datetime.datetime(2026,7,18), datetime.datetime(2026,8,31,hour=23)),
    "pullback_Aug22_31": (datetime.datetime(2026,8,22), datetime.datetime(2026,8,31,hour=23)),
}

# Edge-config -> engine-strategy mappings (SHORT-adaptable)
CFG = {
    "18.0 MeanReversion": ["bollinger_bands", "rsi_mean_reversion"],
    "9.0 VolImbalance":   ["unusual_volume_breakout"],
    "2.1 MTF alignment":  ["swing6_mtf_ema_stack"],
    "17.0 LiqProximity":  ["support_resistance"],
    "4.0 TR/ATR breakout":["resistance_breakout"],
    "2.2 SoftMT alignment":["moving_average_crossover"],
}
DIR = Direction.SHORT
for symbol in ["BTCUSDT", "ETHUSDT"]:
    for wname,(ws,we) in WINDOWS.items():
        for cfgname, stratlist in CFG.items():
            for sname in stratlist:
                run_one(sname, symbol, ws, we, DIR, {}, f"{cfgname}|{wname}")

# Print table
print(f"{'strategy':28} {'sym':7} {'n':>3} {'WR%':>5} {'PF':>6} {'RET%':>7} {'$PnL':>8} {'avg%':>6} {'DD%':>6}")
ROWS.sort(key=lambda r:-abs(r.get("ret") or 0))
for r in sorted(ROWS, key=lambda r:-(r.get("n") or 0)):
    if r.get("error"):
        print(f"{r['strategy']:28} {r['symbol']:7}   ERR {r['error']}")
        continue
    print(f"{r['strategy']:28} {r['symbol']:7} {r.get('n',0):>3} {r.get('wr',0):>5.1f} {r.get('pf',0):>6.2f} {r.get('ret',0):>7.2f} {r.get('ret_abs',0):>8.0f} {r.get('avg',0):>6.2f} {r.get('dd',0):>6.1f}")