"""
Simple Working Backtest Example
===============================

This script demonstrates how to run a backtest with the existing data.
Run this from the project root directory.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Simple Moving Average Crossover Strategy
class SimpleMAStrategy(Strategy):
    # Strategy Parameters
    fast_ma = 10  # Fast moving average period
    slow_ma = 20  # Slow moving average period
    
    def init(self):
        # Calculate moving averages
        self.ma_fast = self.I(lambda x: pd.Series(x).rolling(self.fast_ma).mean(), self.data.Close)
        self.ma_slow = self.I(lambda x: pd.Series(x).rolling(self.slow_ma).mean(), self.data.Close)
    
    def next(self):
        # Buy when fast MA crosses above slow MA
        if crossover(self.ma_fast, self.ma_slow):
            # Calculate how many shares we can buy with available cash
            available_cash = self.equity * 0.95  # Use 95% of equity
            shares_to_buy = int(available_cash / self.data.Close[-1])
            if shares_to_buy > 0:
                self.buy(size=shares_to_buy)
        
        # Sell when fast MA crosses below slow MA
        elif crossover(self.ma_slow, self.ma_fast):
            if self.position.size > 0:
                self.sell(size=self.position.size)


def load_sample_data():
    """Load sample BTCUSDT data if available, or create sample data."""
    try:
        # Try to load existing data
        data_file = "data/raw/BTCUSDT_1h.csv"
        if os.path.exists(data_file):
            print(f"📊 Loading existing data from {data_file}")
            df = pd.read_csv(data_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            
            # Ensure we have the required columns with correct case
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            print(f"✅ Loaded {len(df)} rows of data")
            return df
            
    except Exception as e:
        print(f"⚠️  Could not load existing data: {e}")
    
    # Create sample data if no real data available
    print("📈 Creating sample BTC data for demonstration...")
    
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='h')  # Updated to use 'h'
    np.random.seed(42)  # For reproducible results
    
    # Simulate BTC price movement starting at $30
    initial_price = 30  # Simulate fractional BTC or a smaller crypto
    returns = np.random.normal(0.0001, 0.01, len(dates))  # Smaller volatility
    prices = [initial_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 1))  # Minimum price floor
    
    # Create OHLC data with some realistic spreads
    df = pd.DataFrame(index=dates)
    df['Close'] = prices
    df['Open'] = df['Close'].shift(1).fillna(df['Close'].iloc[0])
    df['High'] = df[['Open', 'Close']].max(axis=1) * np.random.uniform(1.000, 1.005, len(df))
    df['Low'] = df[['Open', 'Close']].min(axis=1) * np.random.uniform(0.995, 1.000, len(df))
    df['Volume'] = np.random.uniform(100, 1000, len(df))
    
    print(f"✅ Created {len(df)} rows of sample data")
    return df


def run_backtest():
    """Run the backtest and display results."""
    print("🚀 Starting Simple Moving Average Crossover Backtest")
    print("=" * 50)
    
    # Load data
    data = load_sample_data()
    
    if data is None or len(data) < 50:
        print("❌ Not enough data to run backtest")
        return
    
    # Show data summary
    print(f"\n📊 Data Summary:")
    print(f"   • Period: {data.index[0]} to {data.index[-1]}")
    print(f"   • Total candles: {len(data)}")
    print(f"   • Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
    
    # Initialize backtest
    bt = Backtest(data, SimpleMAStrategy, cash=10000, commission=0.002)
    
    print(f"\n⚙️  Strategy Configuration:")
    print(f"   • Fast MA: {SimpleMAStrategy.fast_ma} periods")
    print(f"   • Slow MA: {SimpleMAStrategy.slow_ma} periods")
    print(f"   • Initial cash: $10,000")
    print(f"   • Commission: 0.2%")
    
    # Run backtest
    print(f"\n🔄 Running backtest...")
    try:
        result = bt.run()
        
        print(f"\n✅ Backtest completed successfully!")
        print(f"=" * 50)
        print(f"📈 Results:")
        print(f"   • Initial cash: ${result['_equity_curve']['Equity'].iloc[0]:,.2f}")
        print(f"   • Final equity: ${result['_equity_curve']['Equity'].iloc[-1]:,.2f}")
        print(f"   • Total return: {((result['_equity_curve']['Equity'].iloc[-1] / result['_equity_curve']['Equity'].iloc[0]) - 1) * 100:.2f}%")
        print(f"   • Number of trades: {len(result['_trades'])}")
        
        if len(result['_trades']) > 0:
            winning_trades = result['_trades'][result['_trades']['PnL'] > 0]
            print(f"   • Winning trades: {len(winning_trades)}/{len(result['_trades'])} ({len(winning_trades)/len(result['_trades'])*100:.1f}%)")
            print(f"   • Average trade PnL: ${result['_trades']['PnL'].mean():.2f}")
            print(f"   • Max drawdown: ${result['_trades']['PnL'].cumsum().min():.2f}")
        
        # Show trade details
        if len(result['_trades']) > 0:
            print(f"\n📋 Recent trades:")
            trades_to_show = result['_trades'].tail(5)
            for idx, trade in trades_to_show.iterrows():
                entry_time = trade['EntryTime'].strftime('%Y-%m-%d %H:%M') if hasattr(trade['EntryTime'], 'strftime') else str(trade['EntryTime'])
                exit_time = trade['ExitTime'].strftime('%Y-%m-%d %H:%M') if hasattr(trade['ExitTime'], 'strftime') else str(trade['ExitTime'])
                pnl_symbol = "📈" if trade['PnL'] > 0 else "📉"
                print(f"   {pnl_symbol} {entry_time} → {exit_time}: ${trade['PnL']:.2f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        return None


if __name__ == "__main__":
    print("🎯 Simple Crypto Backtesting Demo")
    print("=" * 50)
    
    # Run the backtest
    result = run_backtest()
    
    if result is not None:
        print(f"\n🎉 Demo completed successfully!")
        print(f"📖 This demonstrates how to:")
        print(f"   • Load historical data (real or simulated)")
        print(f"   • Define a simple strategy (MA crossover)")
        print(f"   • Run a backtest with risk management")
        print(f"   • Analyze the results")
        print(f"\n💡 Next steps:")
        print(f"   • Modify the strategy parameters")
        print(f"   • Try different strategies from src/strategies/templates.py")
        print(f"   • Use real data from exchanges via the CLI")
    else:
        print(f"\n❌ Demo failed - check the error messages above")
