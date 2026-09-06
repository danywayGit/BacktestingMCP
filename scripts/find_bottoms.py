#!/usr/bin/env python3
"""Find BTC historical bottoms from local 1h data to define regime-matched periods."""
import sys, os
sys.path.insert(0, '/home/hermes/BacktestingMCP')
os.chdir('/home/hermes/BacktestingMCP')
import sqlite3
from datetime import datetime, timezone
import pandas as pd

conn = sqlite3.connect('data/crypto.db')
df = pd.read_sql_query(
    "SELECT timestamp, close FROM market_data WHERE symbol='BTCUSDT' AND timeframe='1h' ORDER BY timestamp",
    conn)
conn.close()
df['dt'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
df = df.set_index('dt')['close']
# resample daily to find lows
daily = df.resample('1D').agg(['min','max','last'])
daily.columns = ['low','high','close']

print("=== Major BTC drawdown episodes (price falls >20% then finds a low) ===")
# Find bottoms via rolling: a day that is the min of following 90 days AND down >25% from 90d prior high
import numpy as np
low = daily['low'].values
high = daily['high'].values
idx = daily.index
N = len(low)
bottoms = []
for i in range(90, N-90):
    # is this a 90-day forward low?
    if low[i] <= np.min(low[i+1:i+90]):
        past_high = np.max(high[i-90:i])
        if (past_high - low[i])/past_high > 0.22:  # >=22% drawdown
            bottoms.append((idx[i], low[i], past_high, (past_high-low[i])/past_high*100))

# dedupe (keep first of cluster within 60 days)
deduped = []
for b in bottoms:
    if not deduped or (b[0]-deduped[-1][0]).days > 60:
        deduped.append(b)

print(f"\nFound {len(deduped)} major bottom episodes:")
for dt_, lo, hi, dd in deduped:
    print(f"  Bottom {dt_.strftime('%Y-%m-%d')}  low=${lo:,.0f}  (-{dd:.0f}% from ${hi:,.0f})")