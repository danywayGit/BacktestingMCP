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
from ..edge_scanner import composite as edge_composite
from ..edge_scanner import store as edge_store
from ..edge_scanner import alerts as edge_alerts
from ..edge_scanner import patterns as edge_pattern_scanner
from ..edge_scanner import multi_version_scan as edge_multi
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


@strategy.command('list-models')
def list_models():
    """List available Ollama models with recommendations for RTX 4090."""
    try:
        from ..ai.strategy_generator import StrategyGenerator

        generator = StrategyGenerator(provider='ollama')
        models = generator.list_ollama_models()

        if not models:
            click.echo("\n⚠️  No Ollama models found. Is Ollama running?")
            click.echo("  Start it with: ollama serve")
            return

        best = generator._select_best_ollama_model()
        click.echo(f"\nInstalled Ollama models ({len(models)}):")
        click.echo(f"{'#':>3}  {'Model':<45} {'Size':>8}  Note")
        click.echo("-" * 75)
        for i, m in enumerate(models, 1):
            tag = "⭐ recommended" if m['name'] == best else ""
            click.echo(f"{i:>3}  {m['name']:<45} {m['size_gb']:>6.1f} GB  {tag}")
        click.echo(f"\nAuto-selected model: {best}")
        click.echo("Use --model to override: strategy create --provider ollama --model <name>")

    except ImportError as e:
        click.echo(f"\n❌ Error: {e}", err=True)
        click.echo("Install ollama package: pip install ollama")
    except Exception as e:
        click.echo(f"\n❌ Error: {e}", err=True)


def _auto_register_strategy(strategy_name: str) -> None:
    """
    Automatically register a generated strategy in templates.py.
    Adds import statement and STRATEGY_REGISTRY entry.
    Names are sanitized to valid Python identifiers.
    """
    import re
    from pathlib import Path
    from ..ai.strategy_generator import StrategyGenerator

    # Sanitize names to valid Python identifiers
    class_name = StrategyGenerator._sanitize_class_name(strategy_name)
    module_name = StrategyGenerator._sanitize_module_name(strategy_name)
    registry_key = module_name  # lowercase, underscored
    
    # Path to templates.py
    templates_path = Path(__file__).parent.parent / 'strategies' / 'templates.py'
    
    if not templates_path.exists():
        raise FileNotFoundError(f"templates.py not found at {templates_path}")
    
    content = templates_path.read_text(encoding='utf-8')
    
    # Check if already registered
    if f"'{registry_key}'" in content or f'"{registry_key}"' in content:
        raise ValueError(f"Strategy '{registry_key}' is already registered")
    
    # Add import after existing generated imports
    import_line = f"from .generated.{module_name} import {class_name}"
    
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
    new_entry = f"    '{registry_key}': {class_name},\n"
    
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
@click.option('--cash', default=1_000_000, type=float, help='Starting cash amount')
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
@click.option('--cash', default=1_000_000, type=float, help='Starting cash per symbol')
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
@click.option('--cash', default=1_000_000, type=float, show_default=True, help='Starting cash')
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
@click.option('--cash', default=1_000_000, type=float, show_default=True, help='Starting cash')
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
@click.option('--cash', default=1_000_000, type=float, show_default=True, help='Starting cash')
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
def edge():
    """Composite edge scanner: discover, log, track, and report signals."""
    pass


@edge.command('scan')
@click.option('--timeframe', '-t', default='1h',
              type=click.Choice(['1m', '5m', '15m', '30m', '1h', '4h', '12h', '1d', '1w']))
@click.option('--lookback-days', default=30, help='Days of OHLCV history to scan for TA confirmation')
@click.option('--per-side', default=20, help='Number of altFINS bullish/bearish candidates to pull per side')
@click.option('--horizon-hours', default=24, help='Hours ahead to check the outcome of any logged signal')
@click.option('--log/--no-log', default=True, help='Persist actionable signals for forward tracking')
@click.option('--multi/--single', default=True,
              help='Run all config versions in parallel (default) or active config only')
@click.option('--version', default=None, help='Run a specific config version only (e.g. v1.1)')
def edge_scan(timeframe, lookback_days, per_side, horizon_hours, log, multi, version):
    """Run one composite scan cycle.

    By default runs ALL 14 config versions in parallel (one API call).
    Use --single to run only the active config, or --version v1.2 for a specific one.
    """
    from src.edge_scanner.scoring_config import ALL_CONFIGS, ACTIVE_CONFIG

    tf = TimeFrame(timeframe)

    # Single specific version
    if version:
        if version not in ALL_CONFIGS:
            click.echo(f"Unknown version '{version}'. Run 'edge configs' to see available.", err=True)
            return
        configs_to_run = [ALL_CONFIGS[version]]
        multi = False

    # Single mode — active config only
    elif not multi:
        configs_to_run = [ACTIVE_CONFIG]

    # Multi mode — all versions in parallel (default)
    else:
        configs_to_run = list(ALL_CONFIGS.values())

    if not multi and len(configs_to_run) == 1:
        # Legacy single-version output (detailed, all candidates shown)
        cfg = configs_to_run[0]
        scores = edge_composite.run_composite_scan(
            timeframe=tf, lookback_days=lookback_days, per_side_size=per_side, config=cfg
        )
        click.echo()
        click.echo(f"Composite edge scan ({timeframe}) config={cfg.version} — {len(scores)} candidates")
        click.echo('-' * 90)
        for s in scores:
            if s.direction is None:
                continue
            click.echo(
                f"{s.symbol:<8} {s.direction:<6} score={s.composite_score:>6.2f} "
                f"close={s.last_close if s.last_close is not None else float('nan'):.4f} "
                f"{s.components}"
            )
        click.echo('-' * 90)
        if log:
            logged = edge_store.log_signals(scores, tf, horizon_hours=horizon_hours)
            click.echo(f"Logged {logged} actionable signal(s).")
        sent = edge_alerts.send_alerts(scores)
        if sent:
            click.echo(f"Sent {sent} Telegram alert(s) to @CryptoAlertsTradingView.")

    else:
        # Multi-version parallel scan
        click.echo(f"\n⚡ Multi-version scan: {len(configs_to_run)} configs in parallel...")
        result = edge_multi.run_parallel_scan(
            timeframe=tf,
            lookback_days=lookback_days,
            per_side_size=per_side,
            horizon_hours=horizon_hours,
            configs=configs_to_run,
            log_signals=log,
            send_alerts=True,
        )

        # Print per-version summary table
        click.echo(f"\n{'Version':<8} {'LONG':>5} {'SHORT':>6} {'Total':>6}  Top signal")
        click.echo('-' * 70)
        for v in sorted(result.signals_by_version.keys()):
            sigs = result.signals_by_version[v]
            longs  = sum(1 for s in sigs if s.direction == "LONG")
            shorts = sum(1 for s in sigs if s.direction == "SHORT")
            top = sigs[0] if sigs else None
            top_str = f"{top.symbol} {top.direction} {top.composite_score:+.2f}" if top else "—"
            active = " ✅" if v == ACTIVE_CONFIG.version else ""
            click.echo(f"{v+active:<10} {longs:>5} {shorts:>6} {len(sigs):>6}  {top_str}")
        click.echo('-' * 70)
        click.echo(f"\nTotal logged: {result.total_logged} signals across {len(configs_to_run)} versions")
        if result.total_alerts_sent:
            click.echo(f"Sent {result.total_alerts_sent} Telegram alert(s) from active config ({ACTIVE_CONFIG.version})")

    click.echo()


@edge.command('track')
def edge_track():
    """Resolve any logged signals whose tracking horizon has elapsed."""
    resolved = edge_store.resolve_due_signals()
    click.echo(f"Resolved {resolved} signal(s).")


@edge.command('report')
@click.option('--group-by', default='symbol',
              type=click.Choice(['symbol', 'hour', 'direction', 'config', 'coin_type']))
@click.option('--min-n', default=5, help='Minimum resolved signals required to show a group')
@click.option('--since-days', default=90, help='Lookback window for resolved signals')
@click.option('--breakeven/--no-breakeven', default=True,
              help="Show breakeven WR based on each config R:R ratio")
def edge_report(group_by, min_n, since_days, breakeven):
    """Show win-rate / avg return of resolved signals, grouped by segment.

    Use --group-by config to compare performance across scoring config versions.
    Use --group-by coin_type to see which asset categories work best.
    """
    summary = edge_store.performance_report(group_by=group_by, min_n=min_n, since_days=since_days)
    if summary.empty:
        click.echo("No resolved signals matching the criteria yet.")
        return
    click.echo(summary.to_string(index=False))


@edge.command('auto-evolve')
@click.option('--dry-run/--no-dry-run', default=True,
              help='Preview without registering (default: True)')
def edge_auto_evolve(dry_run):
    """Use LLM to auto-generate an improved ScoringConfig based on evolution stats."""
    from ..edge_scanner.llm_evolver import auto_evolve_with_llm
    click.echo("Running LLM-driven evolution...")
    result = auto_evolve_with_llm(dry_run=dry_run)
    click.echo(f"Action: {result.get('action')}")
    click.echo(f"Reason: {result.get('reason')}")
    if result.get('new_config'):
        click.echo(f"Suggested config: {json.dumps(result['new_config'], indent=2)}")


@edge.command('gems')
@click.option('--pages', default=5, help='Number of CoinGecko pages to scan (250 coins each)')
@click.option('--start-page', default=3, help='Start from this page (1=top 250)')
@click.option('--top', default=20, help='Number of top gems to show')
def edge_gems(pages, start_page, top):
    """Scan for spot gem candidates with strong tokenomics for 3-6 month holds."""
    from ..edge_scanner.gem_scanner import scan_gems, format_gem_report
    click.echo(f"Scanning {pages} CoinGecko pages...")
    candidates = scan_gems(pages=pages, start_page=start_page)
    report = format_gem_report(candidates, top_n=top)
    click.echo(report)


@edge.command('daily-summary')
def edge_daily_summary():
    """Generate a daily summary for Telegram — top signals, resolutions, win-rates."""
    from datetime import datetime, timezone, timedelta
    from ..data.database import db
    import pandas as pd

    since = datetime.now(timezone.utc) - timedelta(days=1)
    since = datetime.now(timezone.utc) - timedelta(days=1)
    rows = db.get_resolved_edge_signals(resolved_since=since)
    if not rows:
        msg = (
            "📊 *Daily Edge Scanner — {date}*"
            "\n\nNo signals resolved in the last 24h."
            "\n(PENDING: {pending} | Total DB: {total})"
        )
        from ..data.database import db as _db
        import sqlite3
        conn = sqlite3.connect(_db.db_path)
        pending = conn.execute("SELECT COUNT(*) FROM edge_signals WHERE status='PENDING'").fetchone()[0]
        total = conn.execute("SELECT COUNT(*) FROM edge_signals").fetchone()[0]
        conn.close()
        click.echo(msg.format(
            date=datetime.now(timezone.utc).strftime('%Y-%m-%d'),
            pending=pending,
            total=total,
        ))
        return

    df = pd.DataFrame(rows)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["is_win"] = (df["outcome"] == "WIN").astype(int)

    # Overall stats
    total = len(df)
    wins = df["is_win"].sum()
    losses = (df["outcome"] == "LOSS").sum()
    flats = (df["outcome"] == "FLAT").sum()
    win_rate = (wins / total * 100) if total > 0 else 0
    avg_ret = df["forward_return_pct"].mean() if total > 0 else 0
    avg_win = df[df["outcome"] == "WIN"]["forward_return_pct"].mean() if wins > 0 else 0
    avg_loss = df[df["outcome"] == "LOSS"]["forward_return_pct"].mean() if losses > 0 else 0
    bars_win = df[df["outcome"] == "WIN"]["time_to_resolve_hours"].mean() if wins > 0 else 0
    bars_loss = df[df["outcome"] == "LOSS"]["time_to_resolve_hours"].mean() if losses > 0 else 0

    # Per-config stats (only configs with >= 2 signals)
    config_stats = df.groupby("config_version").agg(
        n=("outcome", "size"),
        wr=("is_win", "mean"),
        ret=("forward_return_pct", "mean"),
        avg_win=("forward_return_pct", lambda x: x[df.loc[x.index, "outcome"] == "WIN"].mean()),
        avg_loss=("forward_return_pct", lambda x: x[df.loc[x.index, "outcome"] == "LOSS"].mean()),
        bars_win=("time_to_resolve_hours", lambda x: x[df.loc[x.index, "outcome"] == "WIN"].mean()),
        bars_loss=("time_to_resolve_hours", lambda x: x[df.loc[x.index, "outcome"] == "LOSS"].mean()),
    ).reset_index()
    config_stats["wr"] = (config_stats["wr"] * 100).round(1)
    config_stats["ret"] = config_stats["ret"].round(3)
    config_stats = config_stats[config_stats["n"] >= 2].sort_values("wr", ascending=False)

    # Per-direction stats
    dir_stats = df.groupby("direction").agg(
        n=("outcome", "size"),
        wr=("is_win", "mean"),
        ret=("forward_return_pct", "mean")
    ).reset_index()
    dir_stats["wr"] = (dir_stats["wr"] * 100).round(1)
    dir_stats["ret"] = dir_stats["ret"].round(3)

    # Build message
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    lines = [
        f"📊 *Daily Edge Scanner — {today}*",
        f"Resolved in last 24h: *{total}* signals",
        f"🟢 WIN: {wins} | 🔴 LOSS: {losses} | ⚪ FLAT: {flats}",
        f"Win-rate: *{win_rate:.1f}%* | 🟢 Avg win: *{avg_win:+.2f}%* ({bars_win:.0f}h) | 🔴 Avg loss: *{avg_loss:+.2f}%* ({bars_loss:.0f}h)",
        "",
    ]

    if not dir_stats.empty:
        lines.append("*By direction:*")
        for _, row in dir_stats.iterrows():
            lines.append(f"  {row['direction']}: {row['n']} sigs, {row['wr']:.1f}% WR, {row['ret']:.2f}% avg")

    if not config_stats.empty:
        lines.append("")
        lines.append("*By config (n≥2):*")
        for _, row in config_stats.iterrows():
            active = " ✅" if row["config_version"] == "v1.0" else ""
            aw = f"🟢{row['avg_win']:+.2f}%" if pd.notna(row.get('avg_win')) else ""
            al = f"🔴{row['avg_loss']:+.2f}%" if pd.notna(row.get('avg_loss')) else ""
            bw = f"{row['bars_win']:.0f}h" if pd.notna(row.get('bars_win')) else ""
            bl = f"{row['bars_loss']:.0f}h" if pd.notna(row.get('bars_loss')) else ""
            lines.append(f"  {row['config_version']}{active}: {row['n']} sigs, {row['wr']:.1f}% WR | 🟢{aw}/🔴{al} | {bw}/{bl}")

    lines.append("")
    lines.append("🤖 *Edge Scanner — auto-generated at 09:00 UTC*")

    click.echo("\n".join(lines))


@edge.command('configs')
def edge_configs():
    """List all scoring config versions and their status."""
    from src.edge_scanner.scoring_config import ALL_CONFIGS, ACTIVE_CONFIG
    configs = db.list_scoring_configs()

    # Seed DB with all known configs if not already there
    for cfg in ALL_CONFIGS.values():
        db.save_scoring_config(cfg)
    if not db.get_active_scoring_config_version():
        db.activate_scoring_config(ACTIVE_CONFIG.version)
    configs = db.list_scoring_configs()

    click.echo(f"\n{'Version':<8} {'Active':<8} {'Description':<60} {'Since'}")
    click.echo('-' * 100)
    for c in configs:
        active_marker = "✅ YES" if c['is_active'] else "   no"
        since = c['activated_at'][:10] if c['activated_at'] else "—"
        click.echo(f"{c['version']:<8} {active_marker:<8} {(c['description'] or '')[:58]:<60} {since}")

    click.echo(f"\nActive config: {db.get_active_scoring_config_version() or 'none'}")
    click.echo("\nTo activate a different version:")
    click.echo("  python -m src.cli.main edge activate-config --version v1.1")


@edge.command('activate-config')
@click.option('--version', required=True, help='Config version to activate (e.g. v1.1)')
def edge_activate_config(version):
    """Switch the active scoring config version.

    All subsequent scans will use the new config.
    Previous signals retain their original config_version for accurate win-rate attribution.
    """
    from src.edge_scanner.scoring_config import ALL_CONFIGS
    if version not in ALL_CONFIGS:
        click.echo(f"Unknown version '{version}'. Available: {list(ALL_CONFIGS.keys())}", err=True)
        return
    cfg = ALL_CONFIGS[version]
    db.save_scoring_config(cfg)
    db.activate_scoring_config(version)
    click.echo(f"✅ Activated scoring config {version}: {cfg.description}")
    click.echo("New scans will use these weights:")
    click.echo(f"  trend={cfg.trend_weight}  volume={cfg.volume_relative_weight}  "
               f"signal_feed={cfg.signal_feed_weight}  scanner={cfg.scanner_hit_weight}  "
               f"onchain={cfg.onchain_netflow_weight}")
    if cfg.min_market_cap_usd > 0:
        click.echo(f"  min_market_cap=${cfg.min_market_cap_usd:,.0f}")
    if cfg.min_volume_relative > 0:
        click.echo(f"  min_volume_relative={cfg.min_volume_relative}x")
    if "ANY" not in cfg.coin_type_filter:
        click.echo(f"  coin_type_filter={cfg.coin_type_filter}")
    if cfg.exclude_coin_types:
        click.echo(f"  exclude_coin_types={cfg.exclude_coin_types}")


@edge.command('patterns')
@click.option('--lookback', default='last 24 hours', help='Time window (e.g. "last 7 days")')
@click.option('--symbols', '-s', default=None, help='Comma-separated symbols to filter (e.g. BTC,ETH,SOL)')
@click.option('--output', type=click.Choice(['table', 'alert']), default='table')
def edge_patterns(lookback, symbols, output):
    """Scan altFINS for chart patterns: wedges, channels, breakouts, support/resistance."""
    from src.edge_scanner.patterns import run_pattern_scan, format_pattern_alert

    sym_list = symbols.split(",") if symbols else None
    result = run_pattern_scan(lookback=lookback, symbol_filter=sym_list)

    if result.total_signals == 0:
        click.echo("No chart patterns detected in the selected time window.")
        return

    if output == 'alert':
        click.echo(format_pattern_alert(result))
        return

    # Table output
    click.echo(f"\n📐 Chart Pattern Scan — {result.total_signals} signals, {len(result.by_pattern)} patterns")
    click.echo(f"🟢 Bullish: {result.by_direction['BULLISH']}  |  🔴 Bearish: {result.by_direction['BEARISH']}")
    click.echo('-' * 80)
    for pattern_key, signals in sorted(result.by_pattern.items(), key=lambda x: -len(x[1])):
        name = signals[0].pattern_name if signals else pattern_key
        bullish = sum(1 for s in signals if s.direction == "BULLISH")
        bearish = sum(1 for s in signals if s.direction == "BEARISH")
        dir_emoji = "🟢" if bullish >= bearish else "🔴"
        click.echo(f"\n{dir_emoji} {name} ({len(signals)} — {bullish}B/{bearish}S)")
        for s in signals[:4]:
            price_str = f" ${s.last_price:.4f}" if s.last_price else ""
            click.echo(f"    {s.symbol:12s} {s.direction:8s}{price_str}")


@edge.command('fund-rate')
@click.option('--poll', is_flag=True, help='Poll funding rates from Binance API and cache them')
@click.option('--show', is_flag=True, help='Show cached funding rates')
@click.option('--symbols', default=None, help='Comma-separated symbols (default: show all with data)')
def edge_fund_rate(poll, show, symbols):
    """Poll or view Binance funding rate data for funding mean-reversion scoring (V8.0).

    Funding rates are cached in-memory with a 5-minute TTL. Use --poll to
    refresh data before running a scan cycle.
    """
    from ..integrations import binance_funding

    if poll:
        # Poll funding rates for all tradeable symbols
        sym_list = symbols.split(",") if symbols else [
            "BTC", "ETH", "SOL", "DOGE", "ADA", "LINK", "AVAX",
            "DOT", "MATIC", "UNI", "ATOM", "NEAR", "APT", "SUI",
            "SEI", "TIA", "INJ", "OP", "ARB",
        ]
        click.echo(f"Polling funding rates for {len(sym_list)} symbols...")
        results = binance_funding.poll_all_funding(sym_list)
        oi_results = binance_funding.poll_all_open_interest(sym_list)
        click.echo(f"Funding data: {len(results)}/{len(sym_list)} symbols refreshed")
        click.echo(f"OI data: {len(oi_results)}/{len(sym_list)} symbols refreshed")
        if results:
            extremes = {s: d for s, d in results.items() if abs(d.get("funding_rate", 0)) > 0.006}
            if extremes:
                click.echo(f"\nExtreme funding rates ({len(extremes)} symbols):")
                for s, d in sorted(extremes.items(), key=lambda x: -abs(x[1].get("funding_rate", 0)))[:10]:
                    fr = d["funding_rate"] * 100
                    click.echo(f"  {s:<8} {fr:>+7.3f}%  mark=${d.get('mark_price', 0):.4f}")
        return

    if show:
        # Show what's in the cache / could be fetched
        sym_list = symbols.split(",") if symbols else ["BTC", "ETH", "SOL"]
        click.echo(f"{'Symbol':<8} {'Funding%':>10} {'Momentum':>10} {'OI':>10}")
        click.echo("-" * 42)
        for s in sym_list:
            fr = binance_funding.get_funding_rate(s)
            mom = binance_funding.get_funding_momentum(s)
            oi = binance_funding.get_oi_change(s)
            fr_str = f"{fr*100:>+7.3f}%" if fr is not None else "   N/A  "
            mom_str = f"{mom*10000:>+7.3f}" if mom is not None else "   N/A  "
            oi_str = f"{oi*100:>+7.2f}%" if oi is not None else "   N/A  "
            click.echo(f"{s:<8} {fr_str:>10} {mom_str:>10} {oi_str:>10}")
        return

    click.echo("Use --poll to fetch data or --show to view cached data.")


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
