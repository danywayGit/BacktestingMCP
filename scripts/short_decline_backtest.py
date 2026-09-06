#!/usr/bin/env python3
"""SHORT-only backtest isolated to the TRUE decline window (peak => Aug31)."""
import sys, os, datetime
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.core.backtesting_engine import engine, Direction
from config.settings import TimeFrame
from src.strategies.templates import get_strategy_class

# Genuine pullback: BTC topped 80.2k Aug27, drifted to ~78k Aug31.
W = (datetime.datetime(2026,8,26), datetime.datetime(2026,8,31,hour=23))

CFG = {
    "unusual_volume_breakout": {"sl_mode":"embedded", "rr_ratio":1.5, "direction":Direction.SHORT},
    "resistance_breakout":     {"sl_mode":"embedded", "rr_ratio":1.5, "direction":Direction.SHORT},
    "bollinger_bands":         {"sl_mode":"embedded", "rr_ratio":1.5, "direction":Direction.SHORT},
    "rsi_mean_reversion":      {"sl_mode":"embedded", "rr_ratio":1.5, "direction":Direction.SHORT},
    "swing6_mtf_ema_stack":    {"sl_mode":"embedded", "rr_ratio":1.5, "direction":Direction.SHORT},
    "moving_average_crossover":{"sl_mode":"embedded", "rr_ratio":1.5, "direction":Direction.SHORT},
    "support_resistance":      {"sl_mode":"embedded", "rr_ratio":1.5, "direction":Direction.SHORT},
    "macd":                    {"sl_mode":"embedded", "rr_ratio":1.5, "direction":Direction.SHORT},
}
for symbol in ["BTCUSDT","ETHUSDT"]:
    for name, p in CFG.items():
        cls=get_strategy_class(name)
        try:
            res=engine.run_backtest(cls, symbol, TimeFrame.H1, W[0], W[1], cash=100000, parameters=p)
            s=res.stats
            print(f"{name:26} {symbol:7} n={len(res.trades):>2} WR={s.get('win_rate_pct',0):>5.1f} PF={s.get('profit_factor',0):>5.2f} RET={s.get('total_return_pct',0):>6.2f} PnL={s.get('final_equity',0)-100000:>7.0f} DD={s.get('max_drawdown_pct',0):>5.1f}")
        except Exception as e:
            print(f"{name:26} {symbol:7} ERR {str(e)[:70]}")