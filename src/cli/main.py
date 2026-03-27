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
from ..strategies.scanner import evaluate_scan, SCAN_TYPES
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


def _auto_register_strategy(strategy_name: str) -> None:
    """
    Automatically register a generated strategy in templates.py.
    Adds import statement and STRATEGY_REGISTRY entry.
    """
    import re
    from pathlib import Path
    
    # Path to templates.py
    templates_path = Path(__file__).parent.parent / 'strategies' / 'templates.py'
    
    if not templates_path.exists():
        raise FileNotFoundError(f"templates.py not found at {templates_path}")
    
    content = templates_path.read_text(encoding='utf-8')
    
    # Check if already registered
    strategy_key = strategy_name.lower()
    if f"'{strategy_key}'" in content or f'"{strategy_key}"' in content:
        raise ValueError(f"Strategy '{strategy_key}' is already registered")
    
    # Add import after existing generated imports
    # Look for pattern: from .generated.xxx import XXX
    import_line = f"from .generated.{strategy_name.lower()} import {strategy_name}"
    
    # Find the last generated import or the main imports section
    generated_import_pattern = r'(from \.generated\.[\w]+ import [\w]+)'
    matches = list(re.finditer(generated_import_pattern, content))
    
    if matches:
        # Insert after last generated import
        last_match = matches[-1]
        insert_pos = last_match.end()
        content = content[:insert_pos] + f"\n{import_line}" + content[insert_pos:]
    else:
        # Insert after the DCA imports
        dca_import_pattern = r'(from \.dca_strategies import [\w, ]+)'
        dca_match = re.search(dca_import_pattern, content)
        if dca_match:
            insert_pos = dca_match.end()
            content = content[:insert_pos] + f"\n{import_line}" + content[insert_pos:]
        else:
            raise ValueError("Could not find insertion point for import")
    
    # Add to STRATEGY_REGISTRY
    registry_pattern = r"(STRATEGY_REGISTRY\s*=\s*\{[^}]+)"
    registry_match = re.search(registry_pattern, content, re.DOTALL)
    
    if not registry_match:
        raise ValueError("Could not find STRATEGY_REGISTRY in templates.py")
    
    # Find the last entry in the registry
    registry_content = registry_match.group(1)
    # Insert before the closing brace
    new_entry = f"    '{strategy_key}': {strategy_name},\n"
    
    # Find where to insert (before the closing })
    registry_end = content.find('}', registry_match.end() - 1)
    if registry_end == -1:
        raise ValueError("Could not find end of STRATEGY_REGISTRY")
    
    # Check if there's a trailing comma
    before_brace = content[:registry_end].rstrip()
    if not before_brace.endswith(','):
        content = before_brace + ',\n' + new_entry + content[registry_end:]
    else:
        content = content[:registry_end] + new_entry + content[registry_end:]
    
    # Write back
    templates_path.write_text(content, encoding='utf-8')


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
        
        # Registration
        if register:
            try:
                _auto_register_strategy(name)
                click.echo(f"\n✅ Strategy '{name}' registered successfully!")
                click.echo(f"Run: python -m src.cli.main strategy list-strategies")
            except Exception as reg_error:
                click.echo(f"\n⚠️  Auto-registration failed: {reg_error}")
                click.echo("Please manually add to src/strategies/templates.py:")
                click.echo(f"  1. Add import: from .generated.{name.lower()} import {name}")
                click.echo(f"  2. Add to STRATEGY_REGISTRY: '{name.lower()}': {name}")
        else:
            click.echo("\nNext steps:")
            click.echo(f"  1. Review the generated code in {filepath}")
            click.echo(f"  2. Test it thoroughly before using with real capital")
            click.echo(f"  3. Register with: python -m src.cli.main strategy create -d '...' -n {name} --register")
            click.echo(f"  4. Or manually add to src/strategies/templates.py STRATEGY_REGISTRY")
        
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
        s = result.stats
        pnl_dollar = s.get('final_equity', cash) - cash
        W = 62

        click.echo()
        click.echo("=" * W)
        click.echo(f"  BACKTEST RESULTS  —  {strategy.upper()}  on  {symbol}  {timeframe}")
        click.echo(f"  Period : {start}  to  {end}")
        click.echo(f"  Cash   : ${cash:,.0f}   Commission: {commission*100:.2f}%")
        click.echo("=" * W)

        click.echo("  RETURNS")
        click.echo(f"    {'P&L ($)':<28}  ${pnl_dollar:>+12,.2f}")
        click.echo(f"    {'P&L (%)':<28}  {s.get('total_return_pct', 0):>+12.2f} %")
        click.echo(f"    {'Buy & Hold return':<28}  {s.get('buy_hold_return_pct', 0):>+12.2f} %")
        click.echo(f"    {'Annualized return':<28}  {s.get('annualized_return_pct', 0):>+12.2f} %")
        click.echo(f"    {'Final equity':<28}  ${s.get('final_equity', cash):>12,.2f}")
        click.echo(f"    {'Peak equity':<28}  ${s.get('peak_equity', cash):>12,.2f}")

        click.echo("  RISK & QUALITY")
        click.echo(f"    {'Sharpe ratio':<28}  {s.get('sharpe_ratio', 0):>13.4f}")
        click.echo(f"    {'Sortino ratio':<28}  {s.get('sortino_ratio', 0):>13.4f}")
        click.echo(f"    {'Calmar ratio':<28}  {s.get('calmar_ratio', 0):>13.4f}")
        click.echo(f"    {'SQN':<28}  {s.get('sqn', 0):>13.4f}")
        click.echo(f"    {'Volatility (ann.)':<28}  {s.get('volatility_pct', 0):>12.2f} %")
        click.echo(f"    {'Max drawdown':<28}  {s.get('max_drawdown_pct', 0):>12.2f} %")
        click.echo(f"    {'Avg drawdown':<28}  {s.get('avg_drawdown_pct', 0):>12.2f} %")
        click.echo(f"    {'Max DD duration':<28}  {s.get('max_drawdown_duration', 'N/A'):>13}")

        click.echo("  TRADE STATISTICS")
        click.echo(f"    {'# Trades':<28}  {int(s.get('num_trades', 0)):>13,}")
        click.echo(f"    {'Win rate':<28}  {s.get('win_rate_pct', 0):>12.2f} %")
        click.echo(f"    {'Profit factor':<28}  {s.get('profit_factor', 0):>13.4f}")
        click.echo(f"    {'Expectancy / trade':<28}  {s.get('expectancy_pct', 0):>12.2f} %")
        click.echo(f"    {'Avg trade':<28}  {s.get('avg_trade_pct', 0):>12.2f} %")
        click.echo(f"    {'Best trade':<28}  {s.get('best_trade_pct', 0):>+12.2f} %")
        click.echo(f"    {'Worst trade':<28}  {s.get('worst_trade_pct', 0):>+12.2f} %")
        click.echo(f"    {'Avg trade duration':<28}  {s.get('avg_trade_duration', 'N/A'):>13}")
        click.echo(f"    {'Exposure time':<28}  {s.get('exposure_time_pct', 0):>12.2f} %")

        click.echo("=" * W)

        if result.trades:
            n = len(result.trades)
            show = min(10, n)
            click.echo(f"  LAST {show} TRADES  (of {n} total)")
            click.echo(f"  {'#':<4} {'Dir':<6} {'Entry':>12} {'Exit':>12} {'Return %':>10}  Duration")
            click.echo("  " + "-" * 58)
            for i, trade in enumerate(result.trades[-show:], n - show + 1):
                click.echo(
                    f"  {i:<4} {trade.get('direction','?'):<6} "
                    f"{trade.get('entry_price', 0):>12.4f} "
                    f"{trade.get('exit_price',  0):>12.4f} "
                    f"{trade.get('return_pct',  0):>+9.2f}%  "
                    f"{trade.get('duration', '')}"
                )
            click.echo("=" * W)

        if save:
            click.echo(f"\n  Results saved (strategy: {result.strategy_name})")
        click.echo()
        
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


@backtest.command('compare-breakouts')
@click.option('--symbol', required=True, help='Trading symbol (e.g., BTCUSDT)')
@click.option('--timeframe', '-t', default='1h',
              type=click.Choice(['1m', '5m', '15m', '30m', '1h', '4h', '12h', '1d', '1w']))
@click.option('--start', required=True, help='Start date (YYYY-MM-DD)')
@click.option('--end', required=True, help='End date (YYYY-MM-DD)')
@click.option('--cash', default=10000, type=float, show_default=True, help='Starting cash')
@click.option('--commission', default=0.001, type=float, show_default=True,
              help='Commission per trade (0.001 = 0.1%%)')
@click.option('--sort-by', default='sharpe_ratio', show_default=True,
              type=click.Choice(['sharpe_ratio', 'total_return_pct', 'profit_factor', 'win_rate_pct', 'num_trades']))
def compare_breakouts(symbol, timeframe, start, end, cash, commission, sort_by):
    """Run and rank all built-in momentum breakout strategies on the same dataset."""
    try:
        breakout_strategies = [
            'unusual_volume_breakout',
            'new_local_high_breakout',
            'resistance_breakout',
            'ascending_triangle_breakout',
        ]

        start_date = datetime.strptime(start, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        end_date = datetime.strptime(end, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        tf = TimeFrame(timeframe)

        rows = []
        for strategy_name in breakout_strategies:
            strategy_class = get_strategy_class(strategy_name)
            result = engine.run_backtest(
                strategy_class=strategy_class,
                symbol=symbol,
                timeframe=tf,
                start_date=start_date,
                end_date=end_date,
                cash=cash,
                commission=commission,
            )
            s = result.stats
            rows.append({
                'strategy': strategy_name,
                'total_return_pct': float(s.get('total_return_pct', 0) or 0),
                'sharpe_ratio': float(s.get('sharpe_ratio', 0) or 0),
                'profit_factor': float(s.get('profit_factor', 0) or 0),
                'win_rate_pct': float(s.get('win_rate_pct', 0) or 0),
                'num_trades': int(s.get('num_trades', 0) or 0),
                'max_drawdown_pct': float(s.get('max_drawdown_pct', 0) or 0),
            })

        reverse = True
        rows_sorted = sorted(rows, key=lambda r: r.get(sort_by, 0), reverse=reverse)

        click.echo()
        click.echo(f"Breakout Comparison on {symbol} {timeframe}  ({start} -> {end})")
        click.echo('=' * 100)
        click.echo(f"{'Rank':<6}{'Strategy':<30}{'Return %':>10}{'Sharpe':>10}{'PF':>10}{'Win %':>10}{'Trades':>10}{'MaxDD %':>12}")
        click.echo('-' * 100)

        for idx, row in enumerate(rows_sorted, 1):
            click.echo(
                f"{idx:<6}{row['strategy']:<30}"
                f"{row['total_return_pct']:>10.2f}"
                f"{row['sharpe_ratio']:>10.3f}"
                f"{row['profit_factor']:>10.3f}"
                f"{row['win_rate_pct']:>10.2f}"
                f"{row['num_trades']:>10d}"
                f"{row['max_drawdown_pct']:>12.2f}"
            )
        click.echo('=' * 100)
        click.echo()

    except Exception as e:
        click.echo(f"Error running breakout comparison: {e}", err=True)
        logger.exception('Breakout comparison failed')


@backtest.command('walk-forward')
@click.option('--strategy', '-s', required=True, help='Strategy name')
@click.option('--symbol', required=True, help='Trading symbol (e.g., BTCUSDT)')
@click.option('--timeframe', '-t', default='1h',
              type=click.Choice(['1m', '5m', '15m', '30m', '1h', '4h', '12h', '1d', '1w']))
@click.option('--start', required=True, help='Start date (YYYY-MM-DD)')
@click.option('--end', required=True, help='End date (YYYY-MM-DD)')
@click.option('--train-ratio', default=0.7, type=float, show_default=True,
              help='Fraction of the period used for training (0.0–1.0)')
@click.option('--cash', default=10000, type=float, show_default=True, help='Starting cash')
@click.option('--commission', default=0.001, type=float, show_default=True,
              help='Commission per trade (0.001 = 0.1%%)')
@click.option('--parameters', '-p', help='Fixed strategy parameters as JSON string')
def walk_forward(strategy, symbol, timeframe, start, end, train_ratio, cash, commission, parameters):
    """Walk-forward validation: compare train vs test metrics to detect overfitting."""
    try:
        raw_params = json.loads(parameters) if parameters else {}
        # Coerce string values to int/float (typed JSON vs. copy-paste from optimization)
        strategy_params = {}
        for k, v in raw_params.items():
            if isinstance(v, str):
                try:
                    strategy_params[k] = int(v) if '.' not in v else float(v)
                except ValueError:
                    strategy_params[k] = v
            else:
                strategy_params[k] = v
        start_date = datetime.strptime(start, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        end_date   = datetime.strptime(end,   '%Y-%m-%d').replace(tzinfo=timezone.utc)
        strategy_class = get_strategy_class(strategy)

        click.echo(f"\nWalk-Forward Validation: {strategy} on {symbol} {timeframe}")
        click.echo(f"Full period : {start} to {end}  "
                   f"(train {train_ratio*100:.0f}% / test {100 - train_ratio*100:.0f}%)")
        click.echo("Running backtests...")

        train_result, test_result, split_date = engine.run_walk_forward(
            strategy_class=strategy_class,
            symbol=symbol,
            timeframe=TimeFrame(timeframe),
            start_date=start_date,
            end_date=end_date,
            train_ratio=train_ratio,
            parameters=strategy_params,
            cash=cash,
            commission=commission,
        )

        split_str = split_date.strftime('%Y-%m-%d')
        click.echo(f"\nTrain: {start} to {split_str}")
        click.echo(f"Test : {split_str} to {end}\n")

        # Metrics to compare: (stats_key, display_label, lower_is_better)
        # lower_is_better=None means "informational only, no degradation calc"
        display_metrics = [
            ('total_return_pct',    'Total Return %',  False),
            ('buy_hold_return_pct', 'Buy & Hold %',    False),
            ('sharpe_ratio',        'Sharpe Ratio',    False),
            ('sortino_ratio',       'Sortino Ratio',   False),
            ('profit_factor',       'Profit Factor',   False),
            ('max_drawdown_pct',    'Max Drawdown %',  True),
            ('win_rate_pct',        'Win Rate %',      False),
            ('sqn',                 'SQN',             False),
            ('num_trades',          'Num Trades',      None),
        ]

        click.echo(f"{'Metric':<22} {'Train':>10} {'Test':>10} {'Change':>10}  Status")
        click.echo("-" * 62)

        issues = 0
        for key, label, lower_is_better in display_metrics:
            train_val = train_result.stats.get(key, 0)
            test_val  = test_result.stats.get(key, 0)

            if lower_is_better is None or not isinstance(train_val, (int, float)):
                tv = int(train_val) if isinstance(train_val, float) else train_val
                vv = int(test_val)  if isinstance(test_val,  float) else test_val
                click.echo(f"{label:<22} {tv:>10} {vv:>10}")
                continue

            degr = ((test_val - train_val) / abs(train_val) * 100) if train_val != 0 else 0.0

            # A result is bad if the metric degrades by >40% in the wrong direction
            is_bad = (lower_is_better and degr > 40) or (not lower_is_better and degr < -40)
            status = "WARN" if is_bad else "OK"
            if is_bad:
                issues += 1

            click.echo(f"{label:<22} {train_val:>10.2f} {test_val:>10.2f} {degr:>+9.1f}%  {status}")

        click.echo("-" * 62)
        if issues == 0:
            click.echo("\nVerdict: ACCEPTABLE — metrics hold within threshold on out-of-sample data.")
        elif issues <= 2:
            click.echo(f"\nVerdict: MARGINAL — {issues} metric(s) degrade >40% on test data. Use cautiously.")
        else:
            click.echo(f"\nVerdict: LIKELY OVERFIT — {issues} metrics collapse on test data. Simplify strategy.")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        logger.exception("Walk-forward failed")


@backtest.command()
@click.option('--strategy', '-s', required=True, help='Strategy name')
@click.option('--symbol', required=True, help='Trading symbol (e.g., BTCUSDT)')
@click.option('--timeframe', '-t', default='1h',
              type=click.Choice(['1m', '5m', '15m', '30m', '1h', '4h', '12h', '1d', '1w']))
@click.option('--start', required=True, help='Start date (YYYY-MM-DD)')
@click.option('--end', required=True, help='End date (YYYY-MM-DD)')
@click.option('--objective', '-o', default='sharpe_ratio', show_default=True,
              type=click.Choice([
                  'sharpe_ratio', 'total_return_pct', 'sortino_ratio',
                  'calmar_ratio', 'profit_factor', 'win_rate_pct',
                  'sqn', 'max_drawdown_pct',
              ]),
              help='Metric to maximise (max_drawdown_pct is minimised)')
@click.option('--param-grid', required=True,
              help='JSON parameter grid, e.g. \'{"rsi_period":[10,14,20],"rsi_oversold":[25,30,35]}\'')
@click.option('--cash', default=10000, type=float, show_default=True, help='Starting cash')
@click.option('--commission', default=0.001, type=float, show_default=True,
              help='Commission per trade (0.001 = 0.1%%)')
@click.option('--top-n', default=10, type=int, show_default=True,
              help='Number of top results to display')
@click.option('--max-tries', default=None, type=int,
              help='Randomly sample this many combinations instead of exhaustive search')
def optimize(strategy, symbol, timeframe, start, end, objective, param_grid, cash, commission, top_n, max_tries):
    """Optimize strategy parameters against a target metric (Sharpe, SQN, Profit Factor, etc.)."""
    try:
        from ..core.backtesting_engine import OPTIMIZATION_OBJECTIVES

        grid = json.loads(param_grid)
        start_date = datetime.strptime(start, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        end_date   = datetime.strptime(end,   '%Y-%m-%d').replace(tzinfo=timezone.utc)
        strategy_class = get_strategy_class(strategy)

        n_combos = 1
        for v in grid.values():
            if isinstance(v, list):
                n_combos *= len(v)

        click.echo(f"\nOptimising: {strategy} on {symbol} {timeframe}")
        click.echo(f"Period    : {start} to {end}")
        click.echo(f"Objective : {objective}")
        click.echo(f"Grid      : " + ", ".join(f"{k}={v}" for k, v in grid.items()))
        click.echo(f"Combos    : {n_combos}")
        click.echo("Running...\n")

        best_stats, best_params, top_results, _ = engine.run_optimization(
            strategy_class=strategy_class,
            symbol=symbol,
            timeframe=TimeFrame(timeframe),
            start_date=start_date,
            end_date=end_date,
            param_grid=grid,
            objective=objective,
            cash=cash,
            commission=commission,
            top_n=top_n,
            max_tries=max_tries,
        )

        click.echo("Best parameters:")
        for param, val in best_params.items():
            click.echo(f"  {param} = {val}")

        click.echo("\nBest result metrics:")
        bt_to_display = [
            ('Return [%]',         'total_return_pct'),
            ('Sharpe Ratio',       'sharpe_ratio'),
            ('Sortino Ratio',      'sortino_ratio'),
            ('Profit Factor',      'profit_factor'),
            ('Max. Drawdown [%]',  'max_drawdown_pct'),
            ('Win Rate [%]',       'win_rate_pct'),
            ('SQN',                'sqn'),
            ('# Trades',           'num_trades'),
        ]
        for bt_key, display_key in bt_to_display:
            val = best_stats.get(bt_key)
            if val is not None:
                try:
                    click.echo(f"  {display_key:<22}: {float(val):.4f}")
                except (TypeError, ValueError):
                    click.echo(f"  {display_key:<22}: {val}")

        click.echo(f"\nTop {min(top_n, len(top_results))} combinations (by {objective}):")
        click.echo(f"  {'Rank':<5} {'Score':>10}  Parameters")
        click.echo("  " + "-" * 60)
        for rank, (idx, score) in enumerate(top_results.items(), 1):
            param_names = list(grid.keys())
            if isinstance(idx, tuple):
                param_str = ", ".join(f"{k}={v}" for k, v in zip(param_names, idx))
            else:
                param_str = f"{param_names[0]}={idx}" if param_names else str(idx)
            click.echo(f"  {rank:<5} {score:>10.4f}  {param_str}")

        # Print best params as clean JSON for easy copy-paste into walk-forward
        import json as _json
        def _to_native(v):
            # Convert numpy scalars (int64, float32, etc.) to plain Python types
            if hasattr(v, 'item'):
                v = v.item()
            if isinstance(v, float) and v == int(v):
                return int(v)
            return v
        best_json = _json.dumps({k: _to_native(v) for k, v in best_params.items()})
        click.echo(f"\n{'='*62}")
        click.echo("  BEST PARAMETERS (copy-paste into walk-forward option 6):")
        click.echo(f"  {best_json}")
        click.echo(f"{'='*62}")
        click.echo("\nTip: Use 'backtest walk-forward' with these params to validate out-of-sample.")

    except json.JSONDecodeError:
        click.echo(
            "Error: --param-grid must be valid JSON, e.g. '{\"rsi_period\":[10,14,20]}'",
            err=True,
        )
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        logger.exception("Optimisation failed")


@cli.group()
def scan():
    """Run breakout scanners to find candidate symbols."""
    pass


@scan.command('run')
@click.option('--scan', 'scan_name', required=True,
              type=click.Choice(['all'] + SCAN_TYPES),
              help='Scan type to evaluate')
@click.option('--symbols', multiple=True, help='Trading symbols (can specify multiple)')
@click.option('--all-major', is_flag=True, help='Scan all major crypto pairs')
@click.option('--timeframe', '-t', default='1h',
              type=click.Choice(['1m', '5m', '15m', '30m', '1h', '4h', '12h', '1d', '1w']))
@click.option('--start', required=True, help='Start date (YYYY-MM-DD)')
@click.option('--end', required=True, help='End date (YYYY-MM-DD)')
@click.option('--parameters', '-p', help='Optional scan parameters as JSON string')
def run_scan(scan_name, symbols, all_major, timeframe, start, end, parameters):
    """Evaluate one or all breakout scans on the latest candle for selected symbols."""
    try:
        if all_major:
            symbol_list = CRYPTO_PAIRS[:20]
        else:
            symbol_list = list(symbols)

        if not symbol_list:
            click.echo('Please specify symbols or use --all-major flag')
            return

        scan_params = json.loads(parameters) if parameters else {}
        start_date = datetime.strptime(start, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        end_date = datetime.strptime(end, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        tf = TimeFrame(timeframe)

        click.echo()
        click.echo(f"Scanner: {scan_name}")
        click.echo(f"Period : {start} to {end}")
        click.echo(f"TF     : {timeframe}")
        click.echo('-' * 90)

        hits = []
        for symbol in symbol_list:
            try:
                data = engine.get_data(symbol, tf, start_date, end_date)
                if data.empty:
                    click.echo(f"{symbol:<10} no_data")
                    continue

                result = evaluate_scan(data, scan_name, scan_params)
                if scan_name == 'all':
                    triggered_scans = [name for name, details in result.items() if details.get('triggered')]
                    if triggered_scans:
                        hits.append((symbol, ','.join(triggered_scans)))
                        click.echo(f"{symbol:<10} HIT  {','.join(triggered_scans)}")
                    else:
                        click.echo(f"{symbol:<10} no_match")
                else:
                    if result.get('triggered'):
                        hits.append((symbol, scan_name))
                        close_price = result.get('close', float('nan'))
                        breakout_level = result.get('breakout_level', float('nan'))
                        rel_volume = result.get('relative_volume', float('nan'))
                        click.echo(
                            f"{symbol:<10} HIT  close={close_price:.4f} "
                            f"breakout={breakout_level:.4f} rel_vol={rel_volume:.2f}"
                        )
                    else:
                        reason = result.get('reason', 'conditions_not_met')
                        click.echo(f"{symbol:<10} no_match ({reason})")
            except Exception as symbol_error:
                click.echo(f"{symbol:<10} error: {symbol_error}")

        click.echo('-' * 90)
        click.echo(f"Hits: {len(hits)} / {len(symbol_list)}")
        if hits:
            click.echo('Matched symbols:')
            for sym, detail in hits:
                click.echo(f"  {sym}: {detail}")
        click.echo()

    except json.JSONDecodeError:
        click.echo("Error: --parameters must be valid JSON", err=True)
    except Exception as e:
        click.echo(f"Error running scanner: {e}", err=True)
        logger.exception('Scanner run failed')


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
        
        W = 110
        click.echo()
        click.echo("=" * W)
        click.echo("  RECENT BACKTEST RESULTS")
        click.echo("=" * W)
        hdr = (f"  {'ID':<4} {'Strategy':<22} {'Symbol':<10} {'TF':<5}"
               f" {'Return %':>9} {'P&L ($)':>12} {'Max DD%':>8}"
               f" {'Win%':>6} {'Sharpe':>7} {'SQN':>7}"
               f" {'Trades':>7} {'PF':>7}  {'Date':<10}")
        click.echo(hdr)
        click.echo("  " + "-" * (W - 2))

        for result in results:
            m = result['metrics']
            cash_used = m.get('initial_cash', 10000)
            pnl = m.get('final_equity', cash_used) - cash_used
            click.echo(
                f"  {result['id']:<4} "
                f"{result['strategy_name']:<22} "
                f"{result['symbol']:<10} "
                f"{result['timeframe']:<5}"
                f" {m.get('total_return_pct', 0):>+9.2f}"
                f" {pnl:>+12,.2f}"
                f" {m.get('max_drawdown_pct', 0):>8.2f}"
                f" {m.get('win_rate_pct', 0):>6.1f}"
                f" {m.get('sharpe_ratio', 0):>7.3f}"
                f" {m.get('sqn', 0):>7.3f}"
                f" {int(m.get('num_trades', 0)):>7,}"
                f" {m.get('profit_factor', 0):>7.3f}"
                f"  {result['created_at'][:10]:<10}"
            )
        click.echo("=" * W)
        click.echo()
        
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
