"""Quick test to understand Signal DCA behavior"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

import sqlite3
import pandas as pd
import importlib.util

# Load fractional module
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

print(f"Data loaded: {len(df)} candles")
print(f"Date range: {df.index[0]} to {df.index[-1]}")
print(f"Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
print()

# Run strategy
strategy = FractionalSignalDCAStrategy(
    symbol='BTCUSDT',
    initial_cash=5000.0,
    monthly_contribution=1000.0,
    commission=0.001
)

result = strategy.run_backtest(df)

print("=" * 70)
print("BACKTEST RESULTS")
print("=" * 70)
print(f"Initial Capital: ${result['initial_cash']:,.2f}")
print(f"Final Cash: ${result['final_cash']:,.2f}")
print(f"Final Position Value: ${result['final_position_value']:,.2f}")
print(f"Final Equity: ${result['final_equity']:,.2f}")
print(f"Total Return: {result['total_return']:.2f}%")
print()
print(f"Coins Held: {result['coins_held']:.6f}")
print(f"Final BTC Price: ${df.iloc[-1]['Close']:,.2f}")
print()
print(f"Total Trades: {result['total_trades']}")
print(f"  Buy Trades: {result['buy_trades']}")
print(f"  Sell Trades: {result['sell_trades']}")
print()

if result['buy_trades'] > 0:
    print("BUY TRADES:")
    print("-" * 70)
    for trade in result['trades']:
        if trade.type == 'BUY':
            print(f"  {trade.date.strftime('%Y-%m-%d')}: Bought {trade.coins:.6f} BTC @ ${trade.price:,.2f} (${trade.amount_usd:,.2f})")
    print()

if result['sell_trades'] > 0:
    print("SELL TRADES:")
    print("-" * 70)
    for trade in result['trades']:
        if trade.type == 'SELL':
            print(f"  {trade.date.strftime('%Y-%m-%d')}: Sold {trade.coins:.6f} BTC @ ${trade.price:,.2f} (${trade.amount_usd:,.2f})")
else:
    print("NO SELL TRADES - Strategy is in accumulation mode")
    print()
    print("Sell Condition: Price > 15% above 50 EMA")
    print(f"Current Price: ${df.iloc[-1]['Close']:,.2f}")
    
    # Calculate 50 EMA at end
    close_prices = df['Close'].tail(51).values
    ema_50 = close_prices[0]
    alpha = 2 / 51
    for price in close_prices[1:]:
        ema_50 = alpha * price + (1 - alpha) * ema_50
    
    sell_threshold = ema_50 * 1.15
    print(f"50 EMA: ${ema_50:,.2f}")
    print(f"Sell Threshold (50 EMA * 1.15): ${sell_threshold:,.2f}")
    print(f"Distance to Sell: {((sell_threshold - df.iloc[-1]['Close']) / df.iloc[-1]['Close'] * 100):.2f}%")
