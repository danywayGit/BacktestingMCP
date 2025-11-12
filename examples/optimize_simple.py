"""
Simple optimization example for the AI-generated strategy.
Tests fewer parameter combinations for faster results.
"""
import sys
from pathlib import Path
from datetime import datetime
import signal

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from backtesting import Backtest
from src.strategies.generated.testrsistrategy import TestRSIStrategy
from src.data.database import db


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\n⚠️  Optimization interrupted by user (Ctrl+C)")
    print("Exiting...")
    sys.exit(0)


def optimize_test_rsi_strategy():
    """Run optimization on TestRSIStrategy with fewer combinations."""
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Fetch data
    print("Fetching BTC data for 2023...")
    data = db.get_market_data(
        'BTCUSDT',
        timeframe='1h',
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31)
    )
    
    if data.empty:
        print("No data found! Run data download first.")
        return None
    
    # Data is already a DataFrame with proper columns and datetime index
    df = data[['open', 'high', 'low', 'close', 'volume']].copy()
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Initialize backtest
    bt = Backtest(
        df,
        TestRSIStrategy,
        cash=100000,
        commission=0.001,  # 0.1% commission
    )
    
    # Run optimization with fewer combinations
    print("\nRunning optimization...")
    print("Testing 8 parameter combinations (2x2x2)")
    print("Parameters:")
    print("  - rsi_period: [14, 20]")
    print("  - ema_period: [100, 200]")
    print("  - rsi_oversold/overbought: [30/70, 25/75]")
    print("Press Ctrl+C at any time to stop.\n")
    
    try:
        stats = bt.optimize(
            rsi_period=[14, 20],           # 2 values
            ema_period=[100, 200],         # 2 values
            rsi_oversold=[25, 30],         # 2 values
            rsi_overbought=[70, 75],       # 2 values
            maximize='Sharpe Ratio',
            constraint=lambda p: p.rsi_oversold < p.rsi_overbought,
            return_heatmap=False
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Optimization interrupted by user")
        return None
    
    return stats


if __name__ == '__main__':
    print("=" * 70)
    print("SIMPLIFIED PARAMETER OPTIMIZATION - TestRSIStrategy")
    print("=" * 70)
    
    try:
        stats = optimize_test_rsi_strategy()
        
        if stats is None:
            print("\n⚠️  Optimization was interrupted or failed")
            sys.exit(0)
        
        print("\n" + "=" * 70)
        print("OPTIMIZATION RESULTS - BEST PARAMETERS")
        print("=" * 70)
        print(stats)
        print("\n" + "=" * 70)
        
        # Extract optimized parameters
        print("\nOptimal Parameters:")
        print(f"  RSI Period: {stats._strategy.rsi_period}")
        print(f"  EMA Period: {stats._strategy.ema_period}")
        print(f"  RSI Oversold: {stats._strategy.rsi_oversold}")
        print(f"  RSI Overbought: {stats._strategy.rsi_overbought}")
        print(f"\nPerformance:")
        print(f"  Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
        print(f"  Return: {stats['Return [%]']:.2f}%")
        print(f"  Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")
        print(f"  Win Rate: {stats['Win Rate [%]']:.2f}%")
        print(f"  Number of Trades: {stats['# Trades']}")
        
        print("\n✅ Optimization completed successfully!")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
