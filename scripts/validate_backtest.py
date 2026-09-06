#!/usr/bin/env python3
"""Validate backtest engine works on local 1h data before grid search."""
import sys, os
sys.path.insert(0, '/home/hermes/BacktestingMCP')
os.chdir('/home/hermes/BacktestingMCP')
from datetime import datetime, timezone

from src.core.backtesting_engine import engine
from config.settings import TimeFrame
from src.strategies.templates import get_strategy_class

btc = get_strategy_class('bollinger_bands')
start = datetime(2020, 1, 1, tzinfo=timezone.utc)
end = datetime(2026, 8, 1, tzinfo=timezone.utc)

print("running single backtest BollingerBands BTC 1h 2020-2026...")
try:
    res = engine.run_backtest(
        btc, "BTCUSDT", TimeFrame.H1, start, end, cash=100000,
        parameters={}  # defaults
    )
    print("strategy:", res.strategy_name, "| stats keys:", len(res.stats))
    for k in ['Return [%]','Sharpe Ratio','# Trades','Win Rate [%]','Max. Drawdown [%]','Profit Factor','Expectancy [%]']:
        if k in res.stats:
            print(f"  {k}: {res.stats[k]}")
except Exception as e:
    import traceback; traceback.print_exc()