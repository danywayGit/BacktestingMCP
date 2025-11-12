"""
Example of how to optimize strategy parameters.

This script shows how class-level parameters can be optimized
using the backtesting.py library's built-in optimizer.
"""

from backtesting import Backtest
from src.strategies.generated.testrsistrategy import TestRSIStrategy
from src.data.database import db
import pandas as pd
import signal
import sys
from datetime import datetime


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\n⚠️  Optimization interrupted by user (Ctrl+C)")
    print("Exiting...")
    sys.exit(0)


def optimize_test_rsi_strategy():
    """Optimize TestRSIStrategy parameters."""
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Load data
    print("Loading BTC data...")
    data = db.get_market_data(
        symbol="BTCUSDT",
        timeframe="1h",
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2023, 12, 31)
    )
    
    if data.empty:
        print("No data found! Run data download first.")
        return
    
    # Prepare data for backtesting
    df = pd.DataFrame({
        'Open': data['open'],
        'High': data['high'],
        'Low': data['low'],
        'Close': data['close'],
        'Volume': data['volume']
    })
    df.index = data.index  # Use existing datetime index
    
    print(f"Data loaded: {len(df)} candles")
    
    # Create backtest
    bt = Backtest(
        df,
        TestRSIStrategy,
        cash=10000,
        commission=0.001
    )
    
    print("\n" + "="*80)
    print("OPTIMIZING STRATEGY PARAMETERS")
    print("="*80)
    print("\nThis will test different combinations of:")
    print("  - RSI period: 10, 14, 20")
    print("  - EMA period: 100, 200, 300")
    print("  - RSI oversold: 25, 30, 35")
    print("  - RSI overbought: 65, 70, 75")
    print("\nOptimizing for maximum Sharpe Ratio...")
    print("(This may take a few minutes)")
    print("Press Ctrl+C to stop optimization\n")
    
    # Optimize parameters
    try:
        stats = bt.optimize(
            rsi_period=[10, 14, 20],
            ema_period=[100, 200, 300],
            rsi_oversold=[25, 30, 35],
            rsi_overbought=[65, 70, 75],
            maximize='Sharpe Ratio',
            constraint=lambda p: p.rsi_oversold < p.rsi_overbought,  # Constraint
            return_heatmap=False  # Don't return heatmap to save memory
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Optimization interrupted by user")
        return None
    
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS")
    print("="*80)
    
    print(f"\nBest Parameters:")
    print(f"  RSI Period: {stats._strategy.rsi_period}")
    print(f"  EMA Period: {stats._strategy.ema_period}")
    print(f"  RSI Oversold: {stats._strategy.rsi_oversold}")
    print(f"  RSI Overbought: {stats._strategy.rsi_overbought}")
    
    print(f"\nPerformance Metrics:")
    print(f"  Return: {stats['Return [%]']:.2f}%")
    print(f"  Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
    print(f"  Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")
    print(f"  Win Rate: {stats['Win Rate [%]']:.2f}%")
    print(f"  # Trades: {stats['# Trades']}")
    
    print("\n" + "="*80)
    
    # Save HTML report
    print("\nGenerating optimization heatmap...")
    bt.plot(filename='optimization_results.html', open_browser=False)
    print("✅ Results saved to: optimization_results.html")
    
    return stats


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║         AI-GENERATED STRATEGY PARAMETER OPTIMIZATION         ║
╚══════════════════════════════════════════════════════════════╝

This demonstrates how class-level parameters allow optimization!
Press Ctrl+C at any time to stop.
""")
    
    try:
        stats = optimize_test_rsi_strategy()
        if stats:
            print("\n✅ Optimization complete!")
        else:
            print("\n⚠️  Optimization was interrupted")
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
