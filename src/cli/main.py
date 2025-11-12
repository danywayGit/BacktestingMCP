"""
Command Line Interface for the backtesting system.
"""

import click
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
import json
from typing import List, Dict, Any
import logging

from ..core.backtesting_engine import engine
from ..data.downloader import downloader, download_crypto_data, update_all_crypto_data
from ..data.database import db
from ..strategies.templates import get_strategy_class, list_available_strategies, get_strategy_parameters
from config.settings import settings, TimeFrame, Direction, CRYPTO_PAIRS

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose):
    """Advanced Crypto Backtesting System with MCP Integration."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.group()
def data():
    """Data management commands."""
    pass


@data.command()
@click.option('--symbol', '-s', required=True, help='Trading symbol (e.g., BTC/USDT)')
@click.option('--timeframe', '-t', default='1h', type=click.Choice(['1m', '5m', '15m', '30m', '1h', '4h', '12h', '1d', '1w']))
@click.option('--start', required=True, help='Start date (YYYY-MM-DD)')
@click.option('--end', help='End date (YYYY-MM-DD, default: today)')
@click.option('--exchange', default='binance', help='Exchange name')
@click.option('--force', is_flag=True, help='Force re-download existing data')
def download(symbol, timeframe, start, end, exchange, force):
    """Download market data for a symbol."""
    try:
        logger.info(f"Downloading {symbol} {timeframe} data from {start} to {end or 'today'}")
        
        data = download_crypto_data(
            symbol=symbol,
            timeframe=TimeFrame(timeframe),
            start_date=start,
            end_date=end,
            exchange=exchange,
            force_update=force
        )
        
        click.echo(f"Downloaded {len(data)} candles for {symbol}")
        click.echo(f"Date range: {data.index[0]} to {data.index[-1]}")
        
    except Exception as e:
        click.echo(f"Error downloading data: {e}", err=True)


@data.command()
@click.option('--symbols', '-s', multiple=True, help='Symbols to download (can specify multiple)')
@click.option('--all-major', is_flag=True, help='Download all major crypto pairs')
@click.option('--timeframe', '-t', default='1h', type=click.Choice(['1m', '5m', '15m', '30m', '1h', '4h', '12h', '1d', '1w']))
@click.option('--start', required=True, help='Start date (YYYY-MM-DD)')
@click.option('--end', help='End date (YYYY-MM-DD, default: today)')
@click.option('--exchange', default='binance', help='Exchange name')
def download_multiple(symbols, all_major, timeframe, start, end, exchange):
    """Download data for multiple symbols."""
    try:
        if all_major:
            symbol_list = [f"{pair[:3]}/{pair[3:]}" for pair in CRYPTO_PAIRS[:10]]  # Top 10
        else:
            symbol_list = list(symbols)
        
        if not symbol_list:
            click.echo("Please specify symbols or use --all-major flag")
            return
        
        logger.info(f"Downloading data for {len(symbol_list)} symbols")
        
        for symbol in symbol_list:
            try:
                data = download_crypto_data(
                    symbol=symbol,
                    timeframe=TimeFrame(timeframe),
                    start_date=start,
                    end_date=end,
                    exchange=exchange
                )
                click.echo(f"✓ {symbol}: {len(data)} candles")
                
            except Exception as e:
                click.echo(f"✗ {symbol}: {e}", err=True)
                continue
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@data.command()
@click.option('--exchange', default='binance', help='Exchange name')
def update(exchange):
    """Update all existing data with latest candles."""
    try:
        logger.info("Updating all existing data...")
        results = update_all_crypto_data(exchange)
        
        for symbol, timeframe_results in results.items():
            click.echo(f"{symbol}:")
            for timeframe, count in timeframe_results.items():
                click.echo(f"  {timeframe}: +{count} candles")
        
    except Exception as e:
        click.echo(f"Error updating data: {e}", err=True)


@data.command()
def list_data():
    """List available data in the database."""
    try:
        symbol_timeframes = db.get_symbols_and_timeframes()
        
        if not symbol_timeframes:
            click.echo("No data found in database")
            return
        
        # Group by symbol
        data_summary = {}
        for symbol, timeframe in symbol_timeframes:
            if symbol not in data_summary:
                data_summary[symbol] = []
            
            # Get data range
            start_date, end_date = db.get_available_data_range(symbol, timeframe)
            data_summary[symbol].append({
                'timeframe': timeframe,
                'start': start_date.strftime('%Y-%m-%d') if start_date else 'N/A',
                'end': end_date.strftime('%Y-%m-%d') if end_date else 'N/A'
            })
        
        # Display summary
        for symbol, timeframes in data_summary.items():
            click.echo(f"\n{symbol}:")
            for tf_info in timeframes:
                click.echo(f"  {tf_info['timeframe']}: {tf_info['start']} to {tf_info['end']}")
        
    except Exception as e:
        click.echo(f"Error listing data: {e}", err=True)


@cli.group()
def strategy():
    """Strategy management commands."""
    pass


@strategy.command()
def list_strategies():
    """List available strategy templates."""
    strategies = list_available_strategies()
    
    click.echo("Available strategy templates:")
    for strategy_name in strategies:
        try:
            params = get_strategy_parameters(strategy_name)
            click.echo(f"\n{strategy_name}:")
            for param, value in params.items():
                click.echo(f"  {param}: {value}")
        except Exception as e:
            click.echo(f"  Error loading parameters: {e}")


@strategy.command()
@click.argument('strategy_name')
def show_parameters(strategy_name):
    """Show parameters for a specific strategy."""
    try:
        params = get_strategy_parameters(strategy_name)
        
        click.echo(f"Parameters for {strategy_name}:")
        for param, value in params.items():
            click.echo(f"  {param}: {value} ({type(value).__name__})")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@strategy.command()
@click.option('--description', '-d', required=True, help='Natural language description of the strategy')
@click.option('--name', '-n', required=True, help='Name for the new strategy (e.g., RSIOversoldStrategy)')
@click.option('--provider', type=click.Choice(['openai', 'anthropic', 'ollama', 'auto']), default='auto', help='AI provider to use')
@click.option('--model', help='Specific model to use (optional)')
@click.option('--output', '-o', help='Output file path (optional)')
@click.option('--register', is_flag=True, help='Automatically register in STRATEGY_REGISTRY')
def create(description, name, provider, model, output, register):
    """Create a trading strategy from natural language description using AI."""
    try:
        from ..ai.strategy_generator import StrategyGenerator
        from pathlib import Path
        
        click.echo(f"🤖 Generating strategy '{name}' using {provider}...")
        click.echo(f"Description: {description}")
        
        # Generate strategy
        generator = StrategyGenerator(provider=provider)
        result = generator.generate_strategy(description, name, model)
        
        click.echo(f"\n✅ Strategy generated successfully!")
        click.echo(f"Provider: {result['provider']}")
        click.echo(f"Model: {result['model']}")
        
        # Validate code
        is_valid, error = generator.validate_strategy_code(result['code'])
        if not is_valid:
            click.echo(f"\n⚠️  Warning: Code validation failed: {error}", err=True)
            click.echo("You may need to manually fix the generated code.")
        
        # Show preview
        click.echo("\n" + "-" * 70)
        click.echo("GENERATED CODE PREVIEW:")
        click.echo("-" * 70)
        lines = result['code'].split('\n')
        for i, line in enumerate(lines[:20], 1):
            click.echo(f"{i:3d}: {line}")
        if len(lines) > 20:
            click.echo(f"... ({len(lines) - 20} more lines)")
        
        # Save strategy
        if output:
            output_path = Path(output)
        else:
            output_path = None
        
        filepath = generator.save_strategy(result['code'], name, output_path)
        click.echo(f"\n💾 Strategy saved to: {filepath}")
        
        # Registration instructions
        if register:
            click.echo("\n⚠️  Auto-registration not yet implemented.")
            click.echo("Please manually add to src/strategies/templates.py:")
            click.echo(f"  1. Add import: from .generated.{name.lower()} import {name}")
            click.echo(f"  2. Add to STRATEGY_REGISTRY: '{name.lower()}': {name}")
        else:
            click.echo("\nNext steps:")
            click.echo(f"  1. Review the generated code in {filepath}")
            click.echo(f"  2. Test it thoroughly before using with real capital")
            click.echo(f"  3. Register in src/strategies/templates.py STRATEGY_REGISTRY")
            click.echo(f"  4. Run: python -m src.cli.main strategy list-strategies")
        
    except ImportError as e:
        click.echo(f"\n❌ Error: {e}", err=True)
        click.echo("\nTo use AI strategy generation, install required packages:")
        click.echo("  For OpenAI: pip install openai")
        click.echo("  For Anthropic: pip install anthropic")
        click.echo("  For Ollama: pip install ollama")
    except Exception as e:
        click.echo(f"\n❌ Error: {e}", err=True)
        logger.exception("Strategy creation failed")


@cli.group()
def backtest():
    """Backtesting commands."""
    pass


@backtest.command()
@click.option('--strategy', '-s', required=True, help='Strategy name')
@click.option('--symbol', required=True, help='Trading symbol (e.g., BTCUSDT)')
@click.option('--timeframe', '-t', default='1h', type=click.Choice(['1m', '5m', '15m', '30m', '1h', '4h', '12h', '1d', '1w']))
@click.option('--start', required=True, help='Start date (YYYY-MM-DD)')
@click.option('--end', required=True, help='End date (YYYY-MM-DD)')
@click.option('--cash', default=10000, type=float, help='Starting cash amount')
@click.option('--commission', default=0.001, type=float, help='Trading commission (0.001 = 0.1%)')
@click.option('--parameters', '-p', help='Strategy parameters as JSON string')
@click.option('--direction', type=click.Choice(['long', 'short', 'both']), default='both', help='Trading direction')
@click.option('--save', is_flag=True, help='Save results to database')
def run(strategy, symbol, timeframe, start, end, cash, commission, parameters, direction, save):
    """Run a backtest for a single symbol."""
    try:
        # Parse parameters
        strategy_params = {}
        if parameters:
            strategy_params = json.loads(parameters)
        
        # Add direction to parameters
        strategy_params['direction'] = Direction(direction)
        
        # Parse dates
        start_date = datetime.strptime(start, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        end_date = datetime.strptime(end, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        
        # Get strategy class
        strategy_class = get_strategy_class(strategy)
        
        # Run backtest
        logger.info(f"Running backtest: {strategy} on {symbol} {timeframe}")
        result = engine.run_backtest(
            strategy_class=strategy_class,
            symbol=symbol,
            timeframe=TimeFrame(timeframe),
            start_date=start_date,
            end_date=end_date,
            parameters=strategy_params,
            cash=cash,
            commission=commission
        )
        
        # Display results
        click.echo(f"\nBacktest Results for {strategy} on {symbol}")
        click.echo("=" * 50)
        
        for metric, value in result.stats.items():
            if isinstance(value, float):
                click.echo(f"{metric}: {value:.4f}")
            else:
                click.echo(f"{metric}: {value}")
        
        click.echo(f"\nTotal trades: {len(result.trades)}")
        
        if result.trades:
            # Show first few trades
            click.echo("\nSample trades:")
            for i, trade in enumerate(result.trades[:5]):
                click.echo(f"  {i+1}: {trade['direction']} at {trade['entry_price']:.4f}, "
                          f"return: {trade['return_pct']:.2f}%")
            
            if len(result.trades) > 5:
                click.echo(f"  ... and {len(result.trades) - 5} more trades")
        
        if save:
            click.echo(f"\n✓ Results saved to database with ID: {result.strategy_name}")
        
    except Exception as e:
        click.echo(f"Error running backtest: {e}", err=True)


@backtest.command()
@click.option('--strategy', '-s', required=True, help='Strategy name')
@click.option('--symbols', multiple=True, help='Trading symbols (can specify multiple)')
@click.option('--all-major', is_flag=True, help='Test all major crypto pairs')
@click.option('--timeframe', '-t', default='1h', type=click.Choice(['1m', '5m', '15m', '30m', '1h', '4h', '12h', '1d', '1w']))
@click.option('--start', required=True, help='Start date (YYYY-MM-DD)')
@click.option('--end', required=True, help='End date (YYYY-MM-DD)')
@click.option('--cash', default=10000, type=float, help='Starting cash per symbol')
@click.option('--parameters', '-p', help='Strategy parameters as JSON string')
@click.option('--sort-by', default='total_return_pct', help='Sort results by metric')
def multi_symbol(strategy, symbols, all_major, timeframe, start, end, cash, parameters, sort_by):
    """Run backtests on multiple symbols."""
    try:
        # Determine symbol list
        if all_major:
            symbol_list = CRYPTO_PAIRS[:20]  # Top 20
        else:
            symbol_list = list(symbols)
        
        if not symbol_list:
            click.echo("Please specify symbols or use --all-major flag")
            return
        
        # Parse parameters
        strategy_params = {}
        if parameters:
            strategy_params = json.loads(parameters)
        
        # Parse dates
        start_date = datetime.strptime(start, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        end_date = datetime.strptime(end, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        
        # Get strategy class
        strategy_class = get_strategy_class(strategy)
        
        # Run backtests
        logger.info(f"Running backtests on {len(symbol_list)} symbols")
        results = engine.run_multi_symbol_backtest(
            strategy_class=strategy_class,
            symbols=symbol_list,
            timeframe=TimeFrame(timeframe),
            start_date=start_date,
            end_date=end_date,
            parameters=strategy_params,
            cash_per_symbol=cash
        )
        
        # Sort and display results
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].stats.get(sort_by, 0),
            reverse=True
        )
        
        click.echo(f"\nMulti-Symbol Backtest Results (sorted by {sort_by})")
        click.echo("=" * 80)
        click.echo(f"{'Symbol':<10} {'Return %':<10} {'Sharpe':<8} {'Max DD %':<10} {'Trades':<8} {'Win Rate %':<12}")
        click.echo("-" * 80)
        
        for symbol, result in sorted_results:
            stats = result.stats
            click.echo(f"{symbol:<10} "
                      f"{stats.get('total_return_pct', 0):<10.2f} "
                      f"{stats.get('sharpe_ratio', 0):<8.2f} "
                      f"{stats.get('max_drawdown_pct', 0):<10.2f} "
                      f"{stats.get('num_trades', 0):<8.0f} "
                      f"{stats.get('win_rate_pct', 0):<12.2f}")
        
    except Exception as e:
        click.echo(f"Error running multi-symbol backtest: {e}", err=True)


@cli.group()
def results():
    """View backtest results."""
    pass


@results.command()
@click.option('--strategy', help='Filter by strategy name')
@click.option('--symbol', help='Filter by symbol')
@click.option('--limit', default=10, help='Number of results to show')
def list_results(strategy, symbol, limit):
    """List recent backtest results."""
    try:
        results = db.get_backtest_results(
            strategy_name=strategy,
            symbol=symbol,
            limit=limit
        )
        
        if not results:
            click.echo("No backtest results found")
            return
        
        click.echo("Recent Backtest Results")
        click.echo("=" * 100)
        click.echo(f"{'ID':<4} {'Strategy':<20} {'Symbol':<10} {'TF':<4} {'Return %':<10} {'Sharpe':<8} {'Date':<12}")
        click.echo("-" * 100)
        
        for result in results:
            metrics = result['metrics']
            click.echo(f"{result['id']:<4} "
                      f"{result['strategy_name']:<20} "
                      f"{result['symbol']:<10} "
                      f"{result['timeframe']:<4} "
                      f"{metrics.get('total_return_pct', 0):<10.2f} "
                      f"{metrics.get('sharpe_ratio', 0):<8.2f} "
                      f"{result['created_at'][:10]:<12}")
        
    except Exception as e:
        click.echo(f"Error listing results: {e}", err=True)


@cli.command()
def info():
    """Show system information."""
    click.echo("Advanced Crypto Backtesting System")
    click.echo("=" * 40)
    click.echo(f"Database path: {settings.data.database_path}")
    click.echo(f"Default exchange: {settings.data.exchange}")
    click.echo(f"Account risk: {settings.risk.account_risk_pct}%")
    click.echo(f"Available strategies: {len(list_available_strategies())}")
    
    # Data summary
    try:
        symbol_timeframes = db.get_symbols_and_timeframes()
        symbols = list(set([st[0] for st in symbol_timeframes]))
        timeframes = list(set([st[1] for st in symbol_timeframes]))
        
        click.echo(f"Stored symbols: {len(symbols)}")
        click.echo(f"Available timeframes: {', '.join(timeframes)}")
        
    except Exception as e:
        click.echo(f"Error getting data info: {e}")


def main():
    """Main CLI entry point."""
    cli()


if __name__ == '__main__':
    main()
