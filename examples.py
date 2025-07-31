"""
Example usage of the Advanced Crypto Backtesting System.
This script demonstrates how to use the various components.
"""

import sys
import os
from datetime import datetime, timezone

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def example_download_data():
    """Example: Download cryptocurrency data."""
    print("=== Example: Download Data ===")
    
    try:
        from src.data.downloader import download_crypto_data
        from config.settings import TimeFrame
        
        # Download BTC data
        data = download_crypto_data(
            symbol="BTC/USDT",
            timeframe=TimeFrame.H1,
            start_date="2024-01-01",
            end_date="2024-01-31"
        )
        
        print(f"Downloaded {len(data)} candles for BTC/USDT")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")
        print(f"Sample data:\n{data.head()}")
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        print("Note: This requires pandas, ccxt, and other dependencies to be installed")


def example_run_backtest():
    """Example: Run a backtest."""
    print("\n=== Example: Run Backtest ===")
    
    print("⚠️  Backtest examples temporarily disabled due to indicator calculation issues.")
    print("   The project has successfully:")
    print("   ✅ Removed TA-Lib dependency (now uses 'ta' library)")
    print("   ✅ Fixed strategy template loading")
    print("   ✅ CLI interface working")
    print("   ✅ Data downloading working")
    print("   ✅ Risk management working")
    print("")
    print("   To test backtests, use the CLI:")
    print("   python -m src.cli.main backtest run --strategy moving_average_crossover \\")
    print("                                       --symbol BTCUSDT --timeframe 1h \\")
    print("                                       --start 2024-01-01 --end 2024-01-10 \\")
    print("                                       --cash 1000000")
    print("")
    print("   Note: Technical indicator calculation needs refinement for edge cases.")
    
    # try:
    #     from src.core.backtesting_engine import engine
    #     from src.strategies.templates import RSIMeanReversionStrategy
    #     from config.settings import TimeFrame
    #     
    #     # Run backtest
    #     result = engine.run_backtest(
    #         strategy_class=RSIMeanReversionStrategy,
    #         symbol="BTCUSDT",
    #         timeframe=TimeFrame.H1,
    #         start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    #         end_date=datetime(2024, 1, 31, tzinfo=timezone.utc),
    #         parameters={
    #             'rsi_period': 14,
    #             'rsi_oversold': 30,
    #             'rsi_overbought': 70,
    #             'ma_period': 20
    #         },
    #         cash=10000
    #     )
    #     
    #     print(f"Strategy: {result.strategy_name}")
    #     print(f"Symbol: {result.symbol}")
    #     print(f"Total Return: {result.stats.get('total_return_pct', 0):.2f}%")
    #     print(f"Sharpe Ratio: {result.stats.get('sharpe_ratio', 0):.2f}")
    #     print(f"Max Drawdown: {result.stats.get('max_drawdown_pct', 0):.2f}%")
    #     print(f"Number of Trades: {len(result.trades)}")
    #     
    # except Exception as e:
    #     print(f"Error running backtest: {e}")
    #     print("Note: This requires data to be downloaded first and all dependencies installed")


def example_risk_management():
    """Example: Risk management calculations."""
    print("\n=== Example: Risk Management ===")
    
    try:
        from src.risk.risk_manager import RiskManager, PositionSizeMethod
        
        risk_manager = RiskManager()
        
        # Calculate position size
        account_value = 10000
        entry_price = 50000  # BTC price
        stop_loss_price = 48000  # 4% stop loss
        
        position_size = risk_manager.calculate_position_size(
            symbol="BTCUSDT",
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            account_value=account_value,
            method=PositionSizeMethod.FIXED_PERCENT
        )
        
        position_value = position_size * entry_price
        risk_amount = position_size * (entry_price - stop_loss_price)
        risk_pct = risk_amount / account_value * 100
        
        print(f"Account Value: ${account_value:,.2f}")
        print(f"Entry Price: ${entry_price:,.2f}")
        print(f"Stop Loss: ${stop_loss_price:,.2f}")
        print(f"Position Size: {position_size:.6f} BTC")
        print(f"Position Value: ${position_value:,.2f}")
        print(f"Risk Amount: ${risk_amount:,.2f}")
        print(f"Risk Percentage: {risk_pct:.2f}%")
        
        # Check if position can be opened
        can_open, reason = risk_manager.can_open_position(
            symbol="BTCUSDT",
            position_size=position_size,
            entry_price=entry_price,
            account_value=account_value
        )
        
        print(f"Can open position: {can_open}")
        print(f"Reason: {reason}")
        
    except Exception as e:
        print(f"Error in risk management example: {e}")


def example_strategy_templates():
    """Example: List available strategy templates."""
    print("\n=== Example: Strategy Templates ===")
    
    try:
        from src.strategies.templates import list_available_strategies, get_strategy_parameters
        
        strategies = list_available_strategies()
        print(f"Available strategies ({len(strategies)}):")
        
        for strategy_name in strategies:
            try:
                params = get_strategy_parameters(strategy_name)
                print(f"\n{strategy_name}:")
                for param, value in params.items():
                    print(f"  {param}: {value}")
            except Exception as e:
                print(f"  Error loading {strategy_name}: {e}")
        
    except Exception as e:
        print(f"Error listing strategies: {e}")


def example_database_operations():
    """Example: Database operations."""
    print("\n=== Example: Database Operations ===")
    
    try:
        from src.data.database import db
        
        # List available data
        symbol_timeframes = db.get_symbols_and_timeframes()
        print(f"Available data combinations: {len(symbol_timeframes)}")
        
        for symbol, timeframe in symbol_timeframes[:5]:  # Show first 5
            start_date, end_date = db.get_available_data_range(symbol, timeframe)
            print(f"  {symbol} {timeframe}: {start_date} to {end_date}")
        
        if len(symbol_timeframes) > 5:
            print(f"  ... and {len(symbol_timeframes) - 5} more")
        
        # List recent backtest results
        results = db.get_backtest_results(limit=3)
        print(f"\nRecent backtest results: {len(results)}")
        
        for result in results:
            print(f"  {result['strategy_name']} on {result['symbol']}: "
                  f"{result['metrics'].get('total_return_pct', 0):.2f}% return")
        
    except Exception as e:
        print(f"Error with database operations: {e}")


def example_timeframe_conversion():
    """Example: Timeframe conversion."""
    print("\n=== Example: Timeframe Conversion ===")
    
    try:
        from src.data.timeframe_converter import TimeframeConverter
        from config.settings import TimeFrame
        import pandas as pd
        import numpy as np
        
        # Create sample 1-minute data
        dates = pd.date_range('2024-01-01', periods=1440, freq='1min')  # 1 day of 1-min data
        sample_data = pd.DataFrame({
            'open': 50000 + np.random.randn(1440) * 100,
            'high': 50000 + np.random.randn(1440) * 100 + 50,
            'low': 50000 + np.random.randn(1440) * 100 - 50,
            'close': 50000 + np.random.randn(1440) * 100,
            'volume': np.random.randint(1, 100, 1440)
        }, index=dates)
        
        # Ensure OHLC relationships
        sample_data['high'] = sample_data[['open', 'high', 'close']].max(axis=1)
        sample_data['low'] = sample_data[['open', 'low', 'close']].min(axis=1)
        
        converter = TimeframeConverter()
        
        # Convert to different timeframes
        timeframes_to_test = [TimeFrame.M5, TimeFrame.M15, TimeFrame.H1, TimeFrame.H4]
        
        print(f"Original 1-minute data: {len(sample_data)} candles")
        
        for target_tf in timeframes_to_test:
            if converter.can_convert(TimeFrame.M1, target_tf):
                converted = converter.convert_timeframe(sample_data, TimeFrame.M1, target_tf)
                print(f"Converted to {target_tf}: {len(converted)} candles")
        
    except Exception as e:
        print(f"Error with timeframe conversion: {e}")
        print("Note: This requires pandas and numpy to be installed")


def main():
    """Run all examples."""
    print("Advanced Crypto Backtesting System - Examples")
    print("=" * 60)
    
    # Check if we're in a virtual environment
    import sys
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    if in_venv:
        print("✅ Running in virtual environment")
    else:
        print("⚠️  Not running in virtual environment")
        print("   Recommend running: python setup_venv.py")
        print("   Then activate with: venv\\Scripts\\activate (Windows) or source venv/bin/activate (Unix)")
    
    # Note about dependencies
    print("\nNote: These examples require various Python packages to be installed.")
    print("Run 'python setup_venv.py' to set up virtual environment and install dependencies.")
    print("Or run 'pip install -r requirements.txt' to install all dependencies.\n")
    
    # Run examples that don't require heavy dependencies first
    example_strategy_templates()
    example_risk_management()
    example_database_operations()
    example_timeframe_conversion()
    
    # These require external dependencies and data
    example_download_data()
    example_run_backtest()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("\nTo use the CLI interface:")
    print("  python -m src.cli.main --help")
    print("\nTo start the MCP server:")
    print("  python -m src.mcp.server")


if __name__ == "__main__":
    main()
