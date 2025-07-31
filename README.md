# Advanced Crypto Backtesting System with MCP Integration

A comprehensive cryptocurrency backtesting platform that combines multiple frameworks with Model Context Protocol (MCP) server integration for enhanced trading strategy development and testing.

## Features

### Core Backtesting
- **Framework**: Built on backtesting.py with plans for multi-framework support
- **Multi-timeframe**: Support for 1m, 5m, 15m, 1h, 4h, 1d and custom timeframes
- **Data Compression**: Convert between timeframes (1m → 5m, 15m, etc.)
- **Multi-pair Testing**: Backtest strategies across multiple cryptocurrency pairs

### Data Management
- **SQLite Database**: Efficient local storage with incremental updates
- **Binance Integration**: Historical data download with CCXT support
- **Incremental Updates**: Add new data without re-downloading existing periods
- **Data Validation**: Ensure data quality and consistency

### Strategy Development
- **Natural Language Processing**: Convert text descriptions to trading strategies
- **Template-based Generation**: Pre-built strategy templates
- **AI-Assisted Creation**: Integration with VS Code Copilot and Claude via MCP
- **Parameter Optimization**: Automated strategy parameter tuning

### Risk Management
- **Position Sizing**: Risk-based position calculation (1%-5% account risk)
- **Stop Loss/Take Profit**: Percentage or indicator-based levels
- **Risk Reward Ratios**: Configurable risk/reward targets
- **Daily/Weekly/Monthly Limits**: Maximum loss limits with trade suspension
- **Correlation Management**: BTC correlation and multi-pair position limits

### Trading Controls
- **Day Selection**: Trade on specific days of the week
- **Direction Control**: Long only, short only, or both
- **Time Filters**: Optional hourly trading restrictions
- **Portfolio Limits**: Maximum concurrent positions

### Performance Analytics
- **Comprehensive Metrics**: Return, Sharpe, Sortino, Calmar ratios
- **Win/Loss Analysis**: Separate stats for long/short positions
- **Temporal Analysis**: Performance by day of week and hour
- **Drawdown Analysis**: Maximum and average drawdown periods
- **Risk Metrics**: Value at Risk (VaR) and Expected Shortfall

### MCP Server Integration
- **Strategy Management**: Create and modify strategies via MCP
- **Backtesting Control**: Run and monitor backtests remotely
- **Data Access**: Query market data and results
- **Real-time Monitoring**: Live backtest progress and results

## Project Structure

```
BacktestingMCP/
├── src/
│   ├── core/
│   │   ├── backtesting_engine.py    # Main backtesting logic
│   │   ├── data_manager.py          # Data storage and retrieval
│   │   ├── strategy_base.py         # Base strategy class
│   │   └── risk_manager.py          # Risk management logic
│   ├── data/
│   │   ├── downloader.py            # Binance/CCXT data download
│   │   ├── database.py              # SQLite database operations
│   │   └── timeframe_converter.py   # Timeframe compression
│   ├── strategies/
│   │   ├── templates/               # Strategy templates
│   │   ├── generator.py             # NLP strategy generation
│   │   └── optimizer.py             # Parameter optimization
│   ├── risk/
│   │   ├── position_sizer.py        # Position sizing logic
│   │   ├── limits.py                # Risk limits enforcement
│   │   └── correlation.py           # Correlation analysis
│   ├── analytics/
│   │   ├── metrics.py               # Performance calculations
│   │   ├── reports.py               # Report generation
│   │   └── visualization.py         # Charts and plots
│   ├── mcp/
│   │   ├── server.py                # MCP server implementation
│   │   ├── tools/                   # MCP tools
│   │   └── handlers/                # Request handlers
│   └── cli/
│       ├── main.py                  # Command line interface
│       └── commands/                # CLI commands
├── data/
│   └── crypto.db                    # SQLite database
├── config/
│   ├── settings.py                  # Configuration
│   └── strategies.yaml              # Strategy templates
├── tests/
│   └── ...                          # Test files
├── requirements.txt                 # Python dependencies
├── pyproject.toml                   # Project configuration
└── README.md                        # This file
```

## 🎯 **Project Status**

✅ **Completed Features:**
- Virtual environment setup and dependency management
- Database schema and data storage (SQLite)
- Cryptocurrency data downloading (CCXT/Binance)
- Strategy template system (6 templates available)
- Risk management calculations
- CLI interface for all operations
- MCP server framework
- Technical analysis using `ta` library (TA-Lib optional)

⚠️ **In Progress:**
- Technical indicator edge case handling (some NaN value issues)
- Full backtest execution (CLI works, programmatic API needs refinement)

🚀 **Ready to Use:**
- Data downloading: `python -m src.cli.main data download --symbol BTCUSDT --timeframe 1h`
- Strategy listing: Available templates work correctly
- Risk calculations: Position sizing and risk management operational
- Database operations: Data storage and retrieval working

## Quick Start

### 1. **Set Up Virtual Environment** (Recommended):
   ```bash
   # Run the virtual environment setup script
   python setup_venv.py
   
   # Activate the virtual environment
   # Windows:
   venv\Scripts\activate
   # or simply run: activate.bat
   
   # macOS/Linux:
   source venv/bin/activate
   # or simply run: ./activate.sh
   ```

### 2. **Alternative: Install Dependencies Globally**:
   ```bash
   pip install -r requirements.txt
   ```

### 3. **Download Initial Data**:
   ```bash
   python -m src.cli.main download-data --symbol BTCUSDT --timeframe 1m --start 2022-01-01 --end 2024-12-31
   ```

### 4. **Download Initial Data**:
   ```bash
   # Make sure virtual environment is activated first
   python -m src.cli.main download-data --symbol BTCUSDT --timeframe 1m --start 2022-01-01 --end 2024-12-31
   ```

### 5. **Create a Strategy**:
   ```bash
   python -m src.cli.main create-strategy --description "Buy when RSI is oversold and price is above 20-day MA"
   ```

### 6. **Run Backtest**:
   ```bash
   python -m src.cli.main backtest --strategy my_strategy --symbol BTCUSDT --timeframe 1h --start 2023-01-01 --end 2023-12-31
   ```

### 7. **Start MCP Server**:
   ```bash
   python -m src.mcp.server
   ```

## Development Setup

### Virtual Environment (Recommended)
This project is designed to work with Python virtual environments for better dependency management and isolation.

1. **Create and activate virtual environment**:
   ```bash
   # Run the automated setup
   python setup_venv.py
   
   # Or manually:
   python -m venv venv
   
   # Activate on Windows:
   venv\Scripts\activate
   
   # Activate on macOS/Linux:
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   # Core dependencies
   pip install pandas numpy sqlalchemy click pydantic python-dotenv rich pyyaml
   
   # Trading and backtesting
   pip install ccxt backtesting plotly scikit-learn scipy
   
   # Technical analysis (ta library - easier to install than TA-Lib)
   pip install ta
   
   # Web interface (optional)
   pip install fastapi uvicorn streamlit
   
   # MCP server
   pip install mcp
   
   # Or install all at once:
   pip install -r requirements.txt
   ```

### ⚠️ TA-Lib Installation (Optional)

TA-Lib is an optional dependency that requires C library installation on Windows and can be difficult to set up. The project uses the `ta` library by default, which provides the same functionality and is much easier to install.

If you want to use TA-Lib instead of the `ta` library:

**Windows**:
1. Download pre-compiled wheels from: https://github.com/cgohlke/talib-build/releases
2. Install: `pip install TA_Lib-0.4.XX-cp3XX-cp3XX-win_amd64.whl`

**macOS**:
```bash
brew install ta-lib
pip install TA-Lib
```

**Linux**:
```bash
sudo apt-get install libta-lib-dev
pip install TA-Lib
```

The project automatically detects if TA-Lib is available and falls back to the `ta` library if not.

3. **Development tools** (optional):
   ```bash
   pip install pytest black isort mypy flake8
   ```

### IDE Configuration

**VS Code**: The setup script automatically creates `.vscode/settings.json` with proper Python interpreter and PYTHONPATH configuration.

**PyCharm**: Set the project interpreter to `venv/bin/python` (or `venv\Scripts\python.exe` on Windows).

### Environment Variables
Copy and customize the generated `.env` file for your configuration:

```bash
# Database
CRYPTO_DB_PATH=data/crypto.db

# Exchange settings  
CRYPTO_EXCHANGE=binance

# Risk management
ACCOUNT_RISK_PCT=1.0
TRADING_COMMISSION=0.001

# MCP Server
MCP_PORT=8000
```

## Configuration

Edit `config/settings.py` to customize:
- Default risk parameters
- Database location
- API credentials
- MCP server settings

## Strategy Templates

The system includes several strategy templates:
- **RSI Mean Reversion**: Buy oversold, sell overbought
- **Moving Average Crossover**: Trend following with MA signals
- **Bollinger Bands**: Mean reversion with volatility bands
- **MACD Momentum**: Trend following with MACD signals
- **Support/Resistance**: Level-based trading

## Risk Management Examples

```python
# 1% account risk with 2:1 reward ratio
risk_config = {
    'account_risk_pct': 1.0,
    'reward_risk_ratio': 2.0,
    'max_daily_loss_pct': 3.0,
    'max_weekly_loss_pct': 5.0,
    'max_monthly_loss_pct': 10.0,
    'max_positions': 5,
    'correlation_limit': 0.7
}
```

## License

MIT License - see LICENSE file for details.
