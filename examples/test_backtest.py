#!/usr/bin/env python3
"""
Simple backtest example using basic moving averages without technical analysis libraries.
"""

import sys
import os
from datetime import datetime, timezone

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_simple_ma_strategy():
    """Create a simple moving average strategy using basic pandas operations."""
    
    from core.backtesting_engine import BaseStrategy
    from backtesting.lib import crossover
    
    class SimpleMACrossover(BaseStrategy):
        """Simple MA Crossover using pandas rolling mean (no TA-lib dependencies)."""
        
        fast_period = 10
        slow_period = 30
        
        def init(self):
            """Initialize using simple pandas rolling mean."""
            # Use basic pandas operations instead of TA library
            self.fast_ma = self.I(lambda x: x.rolling(self.fast_period).mean(), self.data.Close)
            self.slow_ma = self.I(lambda x: x.rolling(self.slow_period).mean(), self.data.Close)
        
        def next(self):
            """Simple crossover strategy."""
            if len(self.data) < max(self.fast_period, self.slow_period):
                return
                
            # Skip if not enough data or indicators are NaN
            if pd.isna(self.fast_ma[-1]) or pd.isna(self.slow_ma[-1]):
                return
            
            # Entry conditions
            if not self.position:
                # Long entry: fast MA crosses above slow MA
                if crossover(self.fast_ma, self.slow_ma):
                    self.buy(size=0.1)  # Buy 0.1 of available cash
                
    return SimpleMACrossover


def run_simple_backtest():
    """Run a simple backtest with the custom strategy."""
    try:
        from core.backtesting_engine import engine
        from config.settings import TimeFrame
        import pandas as pd
        
        print("🧪 Testing Simple Moving Average Crossover...")
        
        # Create the simple strategy
        SimpleStrategy = create_simple_ma_strategy()
        
        result = engine.run_backtest(
            strategy_class=SimpleStrategy,
            symbol="BTCUSDT",
            timeframe=TimeFrame.H1,
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 2, 1, tzinfo=timezone.utc),
            parameters={
                'fast_period': 10,
                'slow_period': 30
            },
            cash=100000  # $100K starting capital
        )
        
        print(f"✅ Backtest completed successfully!")
        print(f"📊 Results:")
        print(f"   Strategy: {result.strategy_name}")
        print(f"   Symbol: {result.symbol}")
        print(f"   Period: {result.start_date} to {result.end_date}")
        print(f"   Starting Cash: $100,000")
        
        # Print key statistics
        if hasattr(result, 'stats') and result.stats:
            print(f"   Final Value: ${result.stats.get('final_value', 'N/A'):,.2f}")
            print(f"   Total Return: {result.stats.get('total_return_pct', 0):.2f}%")
            print(f"   Max Drawdown: {result.stats.get('max_drawdown_pct', 0):.2f}%")
            print(f"   Number of Trades: {len(result.trades) if result.trades else 0}")
            
            if result.trades and len(result.trades) > 0:
                print(f"   Win Rate: {result.stats.get('win_rate_pct', 0):.1f}%")
                print(f"   Sharpe Ratio: {result.stats.get('sharpe_ratio', 0):.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_data_analysis():
    """Analyze the available data first."""
    try:
        from data.database import db
        
        print("📈 Available Data Analysis:")
        
        # Get BTCUSDT data sample
        data = db.get_ohlcv_data("BTCUSDT", "1h", limit=100)
        if data is not None and not data.empty:
            print(f"   BTCUSDT 1h: {len(data)} recent candles")
            print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
            print(f"   Date range: {data.index[0]} to {data.index[-1]}")
            print(f"   Sample data:")
            print(data[['open', 'high', 'low', 'close', 'volume']].tail(3))
        else:
            print("   No BTCUSDT data found")
            
    except Exception as e:
        print(f"❌ Data analysis failed: {e}")


if __name__ == "__main__":
    print("🚀 Simple Backtesting Example")
    print("=" * 50)
    
    # First, analyze the data
    run_data_analysis()
    
    print("\n" + "=" * 50)
    
    # Run the backtest
    success = run_simple_backtest()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Simple backtest completed successfully!")
        print("\n💡 Next steps:")
        print("   1. Try different MA periods (fast_period, slow_period)")
        print("   2. Test on different date ranges")
        print("   3. Use different symbols (ETHUSDT)")
        print("   4. Modify the strategy logic")
    else:
        print("💥 Backtest failed - check error messages above")
