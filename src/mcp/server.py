"""
MCP Server for the backtesting system.
Provides tools for strategy management, backtesting, and data access.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

from ..core.backtesting_engine import engine
from ..data.database import db
from ..data.downloader import downloader, download_crypto_data
from ..strategies.templates import (
    get_strategy_class, 
    list_available_strategies, 
    get_strategy_parameters
)
from config.settings import settings, TimeFrame, Direction

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create MCP server instance
server = Server("backtesting-mcp")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available MCP tools."""
    return [
        types.Tool(
            name="download_crypto_data",
            description="Download cryptocurrency market data for a symbol and timeframe",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Trading symbol (e.g., BTC/USDT or BTCUSDT)"
                    },
                    "timeframe": {
                        "type": "string",
                        "enum": ["1m", "5m", "15m", "30m", "1h", "4h", "12h", "1d", "1w"],
                        "description": "Data timeframe"
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date in YYYY-MM-DD format"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date in YYYY-MM-DD format (optional)"
                    },
                    "force_update": {
                        "type": "boolean",
                        "description": "Force re-download existing data",
                        "default": False
                    }
                },
                "required": ["symbol", "timeframe", "start_date"]
            }
        ),
        
        types.Tool(
            name="list_available_data",
            description="List all available market data in the database",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        
        types.Tool(
            name="list_strategies",
            description="List all available trading strategies",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        
        types.Tool(
            name="get_strategy_info",
            description="Get detailed information about a specific strategy including parameters",
            inputSchema={
                "type": "object",
                "properties": {
                    "strategy_name": {
                        "type": "string",
                        "description": "Name of the strategy"
                    }
                },
                "required": ["strategy_name"]
            }
        ),
        
        types.Tool(
            name="run_backtest",
            description="Run a backtest for a trading strategy",
            inputSchema={
                "type": "object",
                "properties": {
                    "strategy_name": {
                        "type": "string",
                        "description": "Name of the strategy to test"
                    },
                    "symbol": {
                        "type": "string",
                        "description": "Trading symbol (e.g., BTCUSDT)"
                    },
                    "timeframe": {
                        "type": "string",
                        "enum": ["1m", "5m", "15m", "30m", "1h", "4h", "12h", "1d", "1w"],
                        "description": "Data timeframe"
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date in YYYY-MM-DD format"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date in YYYY-MM-DD format"
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Strategy parameters as key-value pairs",
                        "default": {}
                    },
                    "cash": {
                        "type": "number",
                        "description": "Starting cash amount",
                        "default": 10000
                    },
                    "commission": {
                        "type": "number",
                        "description": "Trading commission (0.001 = 0.1%)",
                        "default": 0.001
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["long", "short", "both"],
                        "description": "Trading direction",
                        "default": "both"
                    }
                },
                "required": ["strategy_name", "symbol", "timeframe", "start_date", "end_date"]
            }
        ),
        
        types.Tool(
            name="run_multi_symbol_backtest",
            description="Run backtests on multiple symbols with the same strategy",
            inputSchema={
                "type": "object",
                "properties": {
                    "strategy_name": {
                        "type": "string",
                        "description": "Name of the strategy to test"
                    },
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of trading symbols"
                    },
                    "timeframe": {
                        "type": "string",
                        "enum": ["1m", "5m", "15m", "30m", "1h", "4h", "12h", "1d", "1w"],
                        "description": "Data timeframe"
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date in YYYY-MM-DD format"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date in YYYY-MM-DD format"
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Strategy parameters as key-value pairs",
                        "default": {}
                    },
                    "cash_per_symbol": {
                        "type": "number",
                        "description": "Starting cash amount per symbol",
                        "default": 10000
                    }
                },
                "required": ["strategy_name", "symbols", "timeframe", "start_date", "end_date"]
            }
        ),
        
        types.Tool(
            name="get_backtest_results",
            description="Retrieve stored backtest results",
            inputSchema={
                "type": "object",
                "properties": {
                    "strategy_name": {
                        "type": "string",
                        "description": "Filter by strategy name (optional)"
                    },
                    "symbol": {
                        "type": "string",
                        "description": "Filter by symbol (optional)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 10
                    }
                }
            }
        ),
        
        types.Tool(
            name="create_strategy_from_description",
            description="Create a trading strategy from natural language description",
            inputSchema={
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Natural language description of the trading strategy"
                    },
                    "strategy_name": {
                        "type": "string",
                        "description": "Name for the new strategy"
                    }
                },
                "required": ["description", "strategy_name"]
            }
        ),
        
        types.Tool(
            name="optimize_strategy",
            description="Optimize strategy parameters to find best settings",
            inputSchema={
                "type": "object",
                "properties": {
                    "strategy_name": {
                        "type": "string",
                        "description": "Name of the strategy to optimize"
                    },
                    "symbol": {
                        "type": "string",
                        "description": "Trading symbol"
                    },
                    "timeframe": {
                        "type": "string",
                        "enum": ["1m", "5m", "15m", "30m", "1h", "4h", "12h", "1d", "1w"],
                        "description": "Data timeframe"
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date in YYYY-MM-DD format"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date in YYYY-MM-DD format"
                    },
                    "parameter_ranges": {
                        "type": "object",
                        "description": "Parameter ranges to optimize (e.g., {'rsi_period': [10, 20, 30]})"
                    },
                    "optimization_metric": {
                        "type": "string",
                        "description": "Metric to optimize (e.g., 'total_return_pct', 'sharpe_ratio')",
                        "default": "total_return_pct"
                    },
                    "n_trials": {
                        "type": "integer",
                        "description": "Number of optimization trials",
                        "default": 100
                    }
                },
                "required": ["strategy_name", "symbol", "timeframe", "start_date", "end_date", "parameter_ranges"]
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls."""
    try:
        if name == "download_crypto_data":
            return await download_crypto_data_tool(arguments)
        elif name == "list_available_data":
            return await list_available_data_tool(arguments)
        elif name == "list_strategies":
            return await list_strategies_tool(arguments)
        elif name == "get_strategy_info":
            return await get_strategy_info_tool(arguments)
        elif name == "run_backtest":
            return await run_backtest_tool(arguments)
        elif name == "run_multi_symbol_backtest":
            return await run_multi_symbol_backtest_tool(arguments)
        elif name == "get_backtest_results":
            return await get_backtest_results_tool(arguments)
        elif name == "create_strategy_from_description":
            return await create_strategy_from_description_tool(arguments)
        elif name == "optimize_strategy":
            return await optimize_strategy_tool(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")
    
    except Exception as e:
        logger.error(f"Error in tool {name}: {e}")
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]


async def download_crypto_data_tool(args: dict) -> list[types.TextContent]:
    """Download cryptocurrency data."""
    symbol = args["symbol"]
    timeframe = args["timeframe"]
    start_date = args["start_date"]
    end_date = args.get("end_date")
    force_update = args.get("force_update", False)
    
    try:
        data = download_crypto_data(
            symbol=symbol,
            timeframe=TimeFrame(timeframe),
            start_date=start_date,
            end_date=end_date,
            force_update=force_update
        )
        
        result = {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "records_downloaded": len(data),
            "date_range": {
                "start": data.index[0].isoformat() if not data.empty else None,
                "end": data.index[-1].isoformat() if not data.empty else None
            }
        }
        
        return [types.TextContent(
            type="text",
            text=f"Successfully downloaded {len(data)} records for {symbol} {timeframe}\n" +
                 f"Date range: {result['date_range']['start']} to {result['date_range']['end']}\n" +
                 json.dumps(result, indent=2)
        )]
        
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"Error downloading data: {str(e)}"
        )]


async def list_available_data_tool(args: dict) -> list[types.TextContent]:
    """List available market data."""
    try:
        symbol_timeframes = db.get_symbols_and_timeframes()
        
        if not symbol_timeframes:
            return [types.TextContent(
                type="text",
                text="No market data found in database"
            )]
        
        # Group by symbol
        data_summary = {}
        for symbol, timeframe in symbol_timeframes:
            if symbol not in data_summary:
                data_summary[symbol] = []
            
            start_date, end_date = db.get_available_data_range(symbol, timeframe)
            data_summary[symbol].append({
                "timeframe": timeframe,
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None,
                "days": (end_date - start_date).days if start_date and end_date else 0
            })
        
        result = {
            "total_symbols": len(data_summary),
            "data": data_summary
        }
        
        # Format output
        output = f"Available Market Data ({len(data_summary)} symbols):\n\n"
        for symbol, timeframes in data_summary.items():
            output += f"{symbol}:\n"
            for tf_info in timeframes:
                output += f"  {tf_info['timeframe']}: {tf_info['start_date'][:10]} to {tf_info['end_date'][:10]} ({tf_info['days']} days)\n"
            output += "\n"
        
        output += "\nDetailed JSON:\n" + json.dumps(result, indent=2)
        
        return [types.TextContent(type="text", text=output)]
        
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"Error listing data: {str(e)}"
        )]


async def list_strategies_tool(args: dict) -> list[types.TextContent]:
    """List available strategies."""
    try:
        strategies = list_available_strategies()
        
        output = f"Available Trading Strategies ({len(strategies)}):\n\n"
        
        for strategy_name in strategies:
            try:
                params = get_strategy_parameters(strategy_name)
                output += f"{strategy_name}:\n"
                
                # Get strategy class for description
                strategy_class = get_strategy_class(strategy_name)
                if hasattr(strategy_class, '__doc__') and strategy_class.__doc__:
                    description = strategy_class.__doc__.strip().split('\n')[0]
                    output += f"  Description: {description}\n"
                
                output += "  Parameters:\n"
                for param, value in params.items():
                    output += f"    {param}: {value} ({type(value).__name__})\n"
                output += "\n"
                
            except Exception as e:
                output += f"  Error loading parameters: {e}\n\n"
        
        return [types.TextContent(type="text", text=output)]
        
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"Error listing strategies: {str(e)}"
        )]


async def get_strategy_info_tool(args: dict) -> list[types.TextContent]:
    """Get detailed strategy information."""
    strategy_name = args["strategy_name"]
    
    try:
        # Get strategy class and parameters
        strategy_class = get_strategy_class(strategy_name)
        params = get_strategy_parameters(strategy_name)
        
        # Extract documentation
        doc = strategy_class.__doc__ or "No description available"
        
        result = {
            "name": strategy_name,
            "description": doc.strip(),
            "parameters": params,
            "class_name": strategy_class.__name__
        }
        
        output = f"Strategy: {strategy_name}\n\n"
        output += f"Description:\n{doc}\n\n"
        output += f"Parameters ({len(params)}):\n"
        
        for param, value in params.items():
            output += f"  {param}: {value} ({type(value).__name__})\n"
        
        output += f"\nJSON Details:\n{json.dumps(result, indent=2)}"
        
        return [types.TextContent(type="text", text=output)]
        
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"Error getting strategy info: {str(e)}"
        )]


async def run_backtest_tool(args: dict) -> list[types.TextContent]:
    """Run a backtest."""
    try:
        strategy_name = args["strategy_name"]
        symbol = args["symbol"]
        timeframe = args["timeframe"]
        start_date = args["start_date"]
        end_date = args["end_date"]
        parameters = args.get("parameters", {})
        cash = args.get("cash", 10000)
        commission = args.get("commission", 0.001)
        direction = args.get("direction", "both")
        
        # Parse dates
        start_dt = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        
        # Add direction to parameters
        parameters['direction'] = Direction(direction)
        
        # Get strategy class
        strategy_class = get_strategy_class(strategy_name)
        
        # Run backtest
        result = engine.run_backtest(
            strategy_class=strategy_class,
            symbol=symbol,
            timeframe=TimeFrame(timeframe),
            start_date=start_dt,
            end_date=end_dt,
            parameters=parameters,
            cash=cash,
            commission=commission
        )
        
        # Format output
        output = f"Backtest Results: {strategy_name} on {symbol} {timeframe}\n"
        output += "=" * 60 + "\n\n"
        
        output += f"Period: {start_date} to {end_date}\n"
        output += f"Starting Cash: ${cash:,.2f}\n"
        output += f"Commission: {commission*100:.3f}%\n\n"
        
        output += "Performance Metrics:\n"
        for metric, value in result.stats.items():
            if isinstance(value, float):
                output += f"  {metric}: {value:.4f}\n"
            else:
                output += f"  {metric}: {value}\n"
        
        output += f"\nTotal Trades: {len(result.trades)}\n"
        
        if result.trades:
            output += "\nSample Trades (first 5):\n"
            for i, trade in enumerate(result.trades[:5]):
                output += f"  {i+1}: {trade['direction']} entry={trade['entry_price']:.4f} "
                output += f"exit={trade['exit_price']:.4f} return={trade['return_pct']:.2f}%\n"
            
            if len(result.trades) > 5:
                output += f"  ... and {len(result.trades) - 5} more trades\n"
        
        # Add JSON data
        output += f"\nDetailed JSON Results:\n{json.dumps(result.to_dict(), indent=2, default=str)}"
        
        return [types.TextContent(type="text", text=output)]
        
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"Error running backtest: {str(e)}"
        )]


async def run_multi_symbol_backtest_tool(args: dict) -> list[types.TextContent]:
    """Run multi-symbol backtest."""
    try:
        strategy_name = args["strategy_name"]
        symbols = args["symbols"]
        timeframe = args["timeframe"]
        start_date = args["start_date"]
        end_date = args["end_date"]
        parameters = args.get("parameters", {})
        cash_per_symbol = args.get("cash_per_symbol", 10000)
        
        # Parse dates
        start_dt = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        
        # Get strategy class
        strategy_class = get_strategy_class(strategy_name)
        
        # Run multi-symbol backtest
        results = engine.run_multi_symbol_backtest(
            strategy_class=strategy_class,
            symbols=symbols,
            timeframe=TimeFrame(timeframe),
            start_date=start_dt,
            end_date=end_dt,
            parameters=parameters,
            cash_per_symbol=cash_per_symbol
        )
        
        # Sort results by return
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].stats.get('total_return_pct', 0),
            reverse=True
        )
        
        # Format output
        output = f"Multi-Symbol Backtest Results: {strategy_name}\n"
        output += "=" * 80 + "\n\n"
        
        output += f"Strategy: {strategy_name}\n"
        output += f"Symbols: {', '.join(symbols)}\n"
        output += f"Timeframe: {timeframe}\n"
        output += f"Period: {start_date} to {end_date}\n"
        output += f"Cash per symbol: ${cash_per_symbol:,.2f}\n\n"
        
        # Summary table
        output += f"{'Symbol':<12} {'Return %':<10} {'Sharpe':<8} {'Max DD %':<10} {'Trades':<8} {'Win Rate %':<12}\n"
        output += "-" * 80 + "\n"
        
        total_return = 0
        successful_tests = 0
        
        for symbol, result in sorted_results:
            stats = result.stats
            output += f"{symbol:<12} "
            output += f"{stats.get('total_return_pct', 0):<10.2f} "
            output += f"{stats.get('sharpe_ratio', 0):<8.2f} "
            output += f"{stats.get('max_drawdown_pct', 0):<10.2f} "
            output += f"{stats.get('num_trades', 0):<8.0f} "
            output += f"{stats.get('win_rate_pct', 0):<12.2f}\n"
            
            if stats.get('total_return_pct', 0) != 0:
                total_return += stats.get('total_return_pct', 0)
                successful_tests += 1
        
        if successful_tests > 0:
            avg_return = total_return / successful_tests
            output += f"\nAverage Return: {avg_return:.2f}%\n"
        
        # Add detailed JSON
        output += f"\nDetailed Results JSON:\n"
        results_dict = {symbol: result.to_dict() for symbol, result in results.items()}
        output += json.dumps(results_dict, indent=2, default=str)
        
        return [types.TextContent(type="text", text=output)]
        
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"Error running multi-symbol backtest: {str(e)}"
        )]


async def get_backtest_results_tool(args: dict) -> list[types.TextContent]:
    """Get stored backtest results."""
    try:
        strategy_name = args.get("strategy_name")
        symbol = args.get("symbol")
        limit = args.get("limit", 10)
        
        results = db.get_backtest_results(
            strategy_name=strategy_name,
            symbol=symbol,
            limit=limit
        )
        
        if not results:
            return [types.TextContent(
                type="text",
                text="No backtest results found matching the criteria"
            )]
        
        output = f"Stored Backtest Results ({len(results)} found):\n\n"
        
        # Summary table
        output += f"{'ID':<4} {'Strategy':<20} {'Symbol':<10} {'TF':<4} {'Return %':<10} {'Sharpe':<8} {'Date':<12}\n"
        output += "-" * 80 + "\n"
        
        for result in results:
            metrics = result['metrics']
            output += f"{result['id']:<4} "
            output += f"{result['strategy_name']:<20} "
            output += f"{result['symbol']:<10} "
            output += f"{result['timeframe']:<4} "
            output += f"{metrics.get('total_return_pct', 0):<10.2f} "
            output += f"{metrics.get('sharpe_ratio', 0):<8.2f} "
            output += f"{result['created_at'][:10]:<12}\n"
        
        # Add detailed JSON for first result
        if results:
            output += f"\nDetailed data for first result:\n"
            output += json.dumps(results[0], indent=2, default=str)
        
        return [types.TextContent(type="text", text=output)]
        
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"Error getting backtest results: {str(e)}"
        )]


async def create_strategy_from_description_tool(args: dict) -> list[types.TextContent]:
    """Create strategy from natural language description using AI."""
    description = args["description"]
    strategy_name = args["strategy_name"]
    
    try:
        from ..ai.strategy_generator import StrategyGenerator
        
        # Generate strategy
        generator = StrategyGenerator(provider="auto")
        result = generator.generate_strategy(description, strategy_name)
        
        # Validate
        is_valid, error = generator.validate_strategy_code(result['code'])
        validation_status = "✅ Valid" if is_valid else f"⚠️ Validation failed: {error}"
        
        # Save
        filepath = generator.save_strategy(result['code'], strategy_name)
        
        response = f"""✅ Strategy '{strategy_name}' generated successfully!

Provider: {result['provider']}
Model: {result['model']}
Validation: {validation_status}
Saved to: {filepath}

GENERATED CODE:
{'-' * 70}
{result['code']}
{'-' * 70}

Next steps:
1. Review the generated code in {filepath}
2. Test it thoroughly before using with real capital
3. Register in src/strategies/templates.py STRATEGY_REGISTRY:
   - Add import: from .generated.{strategy_name.lower()} import {strategy_name}
   - Add to registry: '{strategy_name.lower()}': {strategy_name}
4. Run: python -m src.cli.main strategy list-strategies
"""
        
        return [types.TextContent(type="text", text=response)]
    
    except ImportError as e:
        return [types.TextContent(
            type="text",
            text=f"❌ Error: {str(e)}\n\n"
                 f"To use AI strategy generation, install required packages:\n"
                 f"  For OpenAI: pip install openai\n"
                 f"  For Anthropic: pip install anthropic\n"
                 f"  For Ollama: pip install ollama\n\n"
                 f"Set environment variables:\n"
                 f"  OPENAI_API_KEY or ANTHROPIC_API_KEY"
        )]
    
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"❌ Strategy generation failed: {str(e)}\n\n"
                 f"Please check your API credentials and try again."
        )]


async def optimize_strategy_tool(args: dict) -> list[types.TextContent]:
    """Optimize strategy parameters using GPU acceleration."""
    import asyncio
    
    strategy_name = args["strategy_name"]
    symbol = args["symbol"]
    parameter_ranges = args.get("parameter_ranges", {})
    start_date = args.get("start_date", "2021-01-01")
    end_date = args.get("end_date", "2025-12-01")
    timeframe = args.get("timeframe", "1h")
    top_n = args.get("top_n", 10)
    
    try:
        from ..optimization.gpu_optimizer import run_dca_optimization, GPU_AVAILABLE
        
        if not GPU_AVAILABLE:
            return [types.TextContent(
                type="text",
                text="⚠️ GPU acceleration not available. CuPy is required.\n"
                     "Install with: pip install cupy-cuda12x"
            )]
        
        # Run optimization in thread pool to avoid blocking
        def run_optimization():
            return run_dca_optimization(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe,
                param_ranges=parameter_ranges,
                top_n=top_n
            )
        
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, run_optimization)
        
        if 'error' in results:
            return [types.TextContent(
                type="text",
                text=f"❌ Optimization failed: {results['error']}"
            )]
        
        # Format results
        output_lines = [
            f"✅ GPU-Accelerated Optimization Complete!",
            f"",
            f"📊 **Summary**",
            f"- Symbol: {results['symbol']}",
            f"- Timeframe: {results['timeframe']}",
            f"- Period: {results['start_date']} to {results['end_date']}",
            f"- Data points: {results.get('data_points', 'N/A'):,}",
            f"- Valid combinations tested: {results['valid_combinations']:,}",
            f"- Optimization time: {results['optimization_time_seconds']:.2f}s",
            f"- Throughput: {results['throughput_tests_per_second']:.1f} tests/sec",
            f"",
            f"📈 **Statistics**",
            f"- Best return: {results['stats']['best_return_pct']:.2f}%",
            f"- Worst return: {results['stats']['worst_return_pct']:.2f}%",
            f"- Mean return: {results['stats']['mean_return_pct']:.2f}%",
            f"- Median return: {results['stats']['median_return_pct']:.2f}%",
            f"",
            f"🏆 **Top {len(results['top_results'])} Results**",
        ]
        
        for i, result in enumerate(results['top_results'], 1):
            output_lines.append(
                f"\n#{i}: Return: {result['total_return_pct']:.2f}%, "
                f"Trades: {result['total_trades']}, "
                f"RSI: {result['rsi_period']}, EMA: {result['ema_period']}"
            )
        
        return [types.TextContent(
            type="text",
            text="\n".join(output_lines)
        )]
        
    except ImportError as e:
        return [types.TextContent(
            type="text",
            text=f"❌ GPU optimizer not available: {str(e)}\n"
                 f"Ensure CuPy and Numba are installed."
        )]
    except Exception as e:
        logger.exception("Optimization failed")
        return [types.TextContent(
            type="text",
            text=f"❌ Optimization error: {str(e)}"
        )]


async def main():
    """Main server entry point."""
    # Initialize server options
    options = InitializationOptions(
        server_name="backtesting-mcp",
        server_version="0.1.0",
        capabilities=server.get_capabilities(
            notification_options=NotificationOptions(),
            experimental_capabilities={}
        )
    )
    
    # Run the server
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="backtesting-mcp",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the server
    asyncio.run(main())
