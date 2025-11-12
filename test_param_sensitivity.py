"""
Test to compare Numba vs original strategy to find bugs
"""
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import pandas as pd
import sqlite3

# Load original strategy
import importlib.util
spec = importlib.util.spec_from_file_location(
    "fractional_dca_comparison", 
    "scripts/fractional_dca_comparison.py"
)
frac_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frac_module)

FractionalSignalDCAStrategy = frac_module.FractionalSignalDCAStrategy

# Load data
conn = sqlite3.connect('data/crypto.db')
start_ts = int(pd.Timestamp('2023-01-01').timestamp())
end_ts = int(pd.Timestamp('2024-12-31').timestamp())

query = """
    SELECT timestamp, open, high, low, close, volume
    FROM market_data
    WHERE symbol = 'BTCUSDT'
    AND timeframe = '1h'
    AND timestamp >= ?
    AND timestamp <= ?
    ORDER BY timestamp
"""
df = pd.read_sql_query(query, conn, params=(start_ts, end_ts))
conn.close()

df['Timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
df = df[['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]
df.set_index('Timestamp', inplace=True)

print("=" * 70)
print("Comparing Default vs Optimized Parameters")
print("=" * 70)
print()

# Test 1: Default parameters
print("Test 1: Default Parameters")
print("-" * 70)
strategy1 = FractionalSignalDCAStrategy(
    symbol='BTCUSDT',
    initial_cash=5000.0,
    monthly_contribution=1000.0,
    commission=0.001
)
result1 = strategy1.run_backtest(df)
print(f"Return: {result1['total_return']:.2f}%")
print(f"Trades: {result1['total_trades']} (Buy: {result1['buy_trades']}, Sell: {result1['sell_trades']})")
print(f"Final Equity: ${result1['final_equity']:,.2f}")
print()

# Test 2: Optimized parameters from Numba
print("Test 2: Optimized Parameters (min_score=4, aggressive thresholds)")
print("-" * 70)
class OptimizedSignalDCA(FractionalSignalDCAStrategy):
    def __init__(self, symbol, initial_cash, monthly_contribution, commission=0.001):
        super().__init__(symbol, initial_cash, monthly_contribution, commission)
        self.rsi_period = 14
        self.ema_period = 200
        self.min_signal_score = 4
        self.strong_signal_threshold = 7
        self.extreme_signal_score = 10
        self.rsi_oversold_extreme = 30.0
        self.rsi_oversold_moderate = 40.0
        self.ema_distance_extreme = -20.0
        self.ema_distance_strong = -14.0
        self.ema_distance_moderate = -8.0
        self.strong_signal_multiplier = 1.2
        self.extreme_signal_multiplier = 1.4

strategy2 = OptimizedSignalDCA(
    symbol='BTCUSDT',
    initial_cash=5000.0,
    monthly_contribution=1000.0,
    commission=0.001
)
result2 = strategy2.run_backtest(df)
print(f"Return: {result2['total_return']:.2f}%")
print(f"Trades: {result2['total_trades']} (Buy: {result2['buy_trades']}, Sell: {result2['sell_trades']})")
print(f"Final Equity: ${result2['final_equity']:,.2f}")
print()

# Test 3: Very aggressive (should produce most trades)
print("Test 3: Very Aggressive (min_score=4, all thresholds lowered)")
print("-" * 70)
class VeryAggressiveDCA(FractionalSignalDCAStrategy):
    def __init__(self, symbol, initial_cash, monthly_contribution, commission=0.001):
        super().__init__(symbol, initial_cash, monthly_contribution, commission)
        self.rsi_period = 21  # Longer period = smoother
        self.ema_period = 150  # Shorter EMA = more responsive
        self.min_signal_score = 4  # Very low threshold
        self.strong_signal_threshold = 7
        self.extreme_signal_score = 9
        self.rsi_oversold_extreme = 30.0
        self.rsi_oversold_moderate = 40.0
        self.ema_distance_extreme = -20.0
        self.ema_distance_strong = -14.0
        self.ema_distance_moderate = -8.0
        self.strong_signal_multiplier = 1.3
        self.extreme_signal_multiplier = 1.5

strategy3 = VeryAggressiveDCA(
    symbol='BTCUSDT',
    initial_cash=5000.0,
    monthly_contribution=1000.0,
    commission=0.001
)
result3 = strategy3.run_backtest(df)
print(f"Return: {result3['total_return']:.2f}%")
print(f"Trades: {result3['total_trades']} (Buy: {result3['buy_trades']}, Sell: {result3['sell_trades']})")
print(f"Final Equity: ${result3['final_equity']:,.2f}")
print()

# Test 4: Very conservative (should produce fewest trades)
print("Test 4: Very Conservative (min_score=6, all thresholds raised)")
print("-" * 70)
class VeryConservativeDCA(FractionalSignalDCAStrategy):
    def __init__(self, symbol, initial_cash, monthly_contribution, commission=0.001):
        super().__init__(symbol, initial_cash, monthly_contribution, commission)
        self.rsi_period = 14
        self.ema_period = 200
        self.min_signal_score = 6  # High threshold
        self.strong_signal_threshold = 8
        self.extreme_signal_score = 10
        self.rsi_oversold_extreme = 35.0  # Higher = harder to trigger
        self.rsi_oversold_moderate = 45.0
        self.ema_distance_extreme = -18.0  # Less negative = harder to trigger
        self.ema_distance_strong = -12.0
        self.ema_distance_moderate = -8.0
        self.strong_signal_multiplier = 1.2
        self.extreme_signal_multiplier = 1.4

strategy4 = VeryConservativeDCA(
    symbol='BTCUSDT',
    initial_cash=5000.0,
    monthly_contribution=1000.0,
    commission=0.001
)
result4 = strategy4.run_backtest(df)
print(f"Return: {result4['total_return']:.2f}%")
print(f"Trades: {result4['total_trades']} (Buy: {result4['buy_trades']}, Sell: {result4['sell_trades']})")
print(f"Final Equity: ${result4['final_equity']:,.2f}")
print()

print("=" * 70)
print("Analysis")
print("=" * 70)
print(f"Default:        {result1['total_return']:7.2f}% | {result1['total_trades']:2d} trades")
print(f"Optimized:      {result2['total_return']:7.2f}% | {result2['total_trades']:2d} trades")
print(f"Very Aggressive:{result3['total_return']:7.2f}% | {result3['total_trades']:2d} trades")
print(f"Very Conservative:{result4['total_return']:7.2f}% | {result4['total_trades']:2d} trades")
print()

if result1['total_trades'] == result2['total_trades'] == result3['total_trades'] == result4['total_trades']:
    print("⚠️  WARNING: All configurations produce same number of trades!")
    print("This suggests:")
    print("1. The parameter changes are not affecting buy decisions")
    print("2. There might be bugs in the refactored code")
    print("3. The buy opportunities are so limited that all params trigger same trades")
else:
    print("✓ Parameters affect trade frequency as expected")
    print(f"Range: {min(result1['total_trades'], result2['total_trades'], result3['total_trades'], result4['total_trades'])} - {max(result1['total_trades'], result2['total_trades'], result3['total_trades'], result4['total_trades'])} trades")
