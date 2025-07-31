"""
Configuration module for the backtesting system.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class TimeFrame(str, Enum):
    """Supported timeframes."""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    H12 = "12h"
    D1 = "1d"
    W1 = "1w"


class Direction(str, Enum):
    """Trading directions."""
    LONG = "long"
    SHORT = "short"
    BOTH = "both"


@dataclass
class RiskConfig:
    """Risk management configuration."""
    account_risk_pct: float = 1.0  # % of account to risk per trade
    max_daily_loss_pct: float = 3.0  # Max daily loss %
    max_weekly_loss_pct: float = 5.0  # Max weekly loss %
    max_monthly_loss_pct: float = 10.0  # Max monthly loss %
    max_positions: int = 5  # Max concurrent positions
    correlation_limit: float = 0.7  # Max correlation between assets
    default_stop_loss_pct: float = 2.0  # Default stop loss %
    default_take_profit_pct: float = 4.0  # Default take profit %
    reward_risk_ratio: float = 2.0  # Reward to risk ratio


@dataclass
class TradingConfig:
    """Trading configuration."""
    direction: Direction = Direction.BOTH
    trading_days: List[int] = field(default_factory=lambda: list(range(7)))  # 0=Monday, 6=Sunday
    trading_hours: Optional[List[int]] = None  # Hours of day to trade (0-23)
    commission: float = 0.001  # Trading commission (0.1%)
    slippage: float = 0.0005  # Slippage (0.05%)


@dataclass
class DataConfig:
    """Data configuration."""
    database_path: str = "data/crypto.db"
    exchange: str = "binance"
    default_timeframe: TimeFrame = TimeFrame.H1
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    start_date: str = "2022-01-01"
    end_date: str = "2024-12-31"


@dataclass
class MCPConfig:
    """MCP server configuration."""
    host: str = "localhost"
    port: int = 8000
    name: str = "backtesting-mcp"
    version: str = "0.1.0"


@dataclass
class OptimizationConfig:
    """Optimization configuration."""
    method: str = "optuna"  # optuna, grid, random
    n_trials: int = 100
    timeout: Optional[int] = None  # seconds
    n_jobs: int = 1  # parallel jobs


@dataclass
class Settings:
    """Main settings class."""
    risk: RiskConfig = field(default_factory=RiskConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    
    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    strategies_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "src" / "strategies")
    
    def __post_init__(self):
        """Ensure directories exist."""
        self.data_dir.mkdir(exist_ok=True)
        self.strategies_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()


# Environment variable overrides
def load_from_env():
    """Load configuration from environment variables."""
    if db_path := os.getenv("CRYPTO_DB_PATH"):
        settings.data.database_path = db_path
    
    if exchange := os.getenv("CRYPTO_EXCHANGE"):
        settings.data.exchange = exchange
    
    if risk_pct := os.getenv("ACCOUNT_RISK_PCT"):
        settings.risk.account_risk_pct = float(risk_pct)
    
    if commission := os.getenv("TRADING_COMMISSION"):
        settings.trading.commission = float(commission)
    
    if mcp_port := os.getenv("MCP_PORT"):
        settings.mcp.port = int(mcp_port)


# Load environment variables on import
load_from_env()


# Timeframe mappings for pandas
TIMEFRAME_MAPPING = {
    TimeFrame.M1: "1T",
    TimeFrame.M5: "5T",
    TimeFrame.M15: "15T",
    TimeFrame.M30: "30T",
    TimeFrame.H1: "1H",
    TimeFrame.H4: "4H",
    TimeFrame.H12: "12H",
    TimeFrame.D1: "1D",
    TimeFrame.W1: "1W",
}


# CCXT timeframe mappings
CCXT_TIMEFRAME_MAPPING = {
    TimeFrame.M1: "1m",
    TimeFrame.M5: "5m",
    TimeFrame.M15: "15m",
    TimeFrame.M30: "30m",
    TimeFrame.H1: "1h",
    TimeFrame.H4: "4h",
    TimeFrame.H12: "12h",
    TimeFrame.D1: "1d",
    TimeFrame.W1: "1w",
}


# Common cryptocurrency pairs
CRYPTO_PAIRS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT",
    "SOLUSDT", "DOTUSDT", "DOGEUSDT", "AVAXUSDT", "SHIBUSDT",
    "MATICUSDT", "LTCUSDT", "LINKUSDT", "ATOMUSDT", "ETCUSDT",
    "XLMUSDT", "NEARUSDT", "ALGOUSDT", "VETUSDT", "FILUSDT"
]


# Performance metrics to calculate
PERFORMANCE_METRICS = [
    "total_return",
    "annualized_return", 
    "volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "max_drawdown",
    "avg_drawdown",
    "win_rate",
    "avg_win",
    "avg_loss",
    "profit_factor",
    "expectancy",
    "sqn",  # System Quality Number
]
