"""
BacktestingMCP — Public API

Quick imports so you don't need to dig through sub-modules:

    from src import backtest_engine, database, list_strategies
    from src import GPUOptimizer, check_gpu_status
    from src import StrategyGenerator
"""

# Core engine
from src.core.backtesting_engine import engine as backtest_engine, BacktestResult

# Data
from src.data.database import CryptoDatabase, db as database
from src.data.downloader import DataDownloader, downloader as data_downloader

# Strategies
from src.strategies.templates import (
    list_available_strategies,
    get_strategy_class,
    get_strategy_parameters,
)

# GPU optimisation (graceful if CuPy not installed)
from src.optimization import (
    GPUOptimizer,
    optimize_with_gpu,
    check_gpu_status,
    GPU_AVAILABLE,
)

# AI strategy generation
from src.ai.strategy_generator import StrategyGenerator

__all__ = [
    # Core
    "backtest_engine",
    "BacktestResult",
    # Data
    "CryptoDatabase",
    "database",
    "DataDownloader",
    "data_downloader",
    # Strategies
    "list_available_strategies",
    "get_strategy_class",
    "get_strategy_parameters",
    # Optimization
    "GPUOptimizer",
    "optimize_with_gpu",
    "check_gpu_status",
    "GPU_AVAILABLE",
    # AI
    "StrategyGenerator",
]
