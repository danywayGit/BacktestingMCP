# Quick Start Guide

This guide will help you get started with the Advanced Crypto Backtesting System quickly.

## Prerequisites

- Python 3.9 or later
- Windows, macOS, or Linux
- Internet connection for downloading market data

## Installation

### 1. Automatic Setup (Recommended)

Run the setup script to install everything automatically:

```bash
python setup.py
```

This will:
- Check your Python version
- Install all required dependencies
- Create necessary directories
- Set up configuration files
- Run basic tests

### 2. Manual Setup

If the automatic setup doesn't work, follow these steps:

```bash
# Install core dependencies
pip install pandas numpy sqlalchemy click pydantic python-dotenv rich

# Install trading dependencies (may require special setup)
pip install ccxt backtesting ta-lib plotly scikit-learn

# Install optional web dependencies
pip install fastapi uvicorn streamlit

# Install MCP dependencies
pip install mcp
```

## First Steps

### 1. Download Sample Data

```bash
# Using the CLI
python -m src.cli.main data download --symbol BTC/USDT --timeframe 1h --start 2024-01-01 --end 2024-01-31

# Or using the example script
python examples.py
```

### 2. Run Your First Backtest

```bash
python -m src.cli.main backtest run \
  --strategy rsi_mean_reversion \
  --symbol BTCUSDT \
  --timeframe 1h \
  --start 2024-01-01 \
  --end 2024-01-31 \
  --cash 10000
```

### 3. View Available Strategies

```bash
python -m src.cli.main strategy list-strategies
```

### 4. Multi-Symbol Backtest

```bash
python -m src.cli.main backtest multi-symbol \
  --strategy moving_average_crossover \
  --symbols BTCUSDT ETHUSDT ADAUSDT \
  --timeframe 1h \
  --start 2024-01-01 \
  --end 2024-01-31
```

## MCP Server Usage

### Start the Server

```bash
python -m src.mcp.server
```

### Available MCP Tools

1. **download_crypto_data** - Download market data
2. **list_available_data** - List stored data
3. **list_strategies** - List available strategies
4. **run_backtest** - Run a backtest
5. **run_multi_symbol_backtest** - Test multiple symbols
6. **get_backtest_results** - Retrieve results
7. **optimize_strategy** - Parameter optimization (planned)

### Example MCP Usage with Claude Desktop

Add this to your Claude Desktop MCP configuration:

```json
{
  "mcpServers": {
    "backtesting": {
      "command": "python",
      "args": ["-m", "src.mcp.server"],
      "cwd": "/path/to/BacktestingMCP"
    }
  }
}
```

## Common Use Cases

### 1. Test a Strategy on Multiple Timeframes

```bash
# Test RSI strategy on different timeframes
for tf in 15m 1h 4h 1d; do
  python -m src.cli.main backtest run \
    --strategy rsi_mean_reversion \
    --symbol BTCUSDT \
    --timeframe $tf \
    --start 2024-01-01 \
    --end 2024-12-31
done
```

### 2. Download Data for Multiple Symbols

```bash
python -m src.cli.main data download-multiple \
  --symbols BTC/USDT ETH/USDT ADA/USDT \
  --timeframe 1h \
  --start 2022-01-01 \
  --end 2024-12-31
```

### 3. View Results

```bash
# List recent results
python -m src.cli.main results list-results --limit 10

# Filter by strategy
python -m src.cli.main results list-results --strategy rsi_mean_reversion

# Filter by symbol
python -m src.cli.main results list-results --symbol BTCUSDT
```

## Configuration

Edit the `.env` file to customize settings:

```bash
# Risk management
ACCOUNT_RISK_PCT=1.0
TRADING_COMMISSION=0.001

# Database location
CRYPTO_DB_PATH=data/crypto.db

# Exchange
CRYPTO_EXCHANGE=binance
```

## Strategy Customization

### Modify Existing Strategy Parameters

```bash
python -m src.cli.main backtest run \
  --strategy rsi_mean_reversion \
  --symbol BTCUSDT \
  --timeframe 1h \
  --start 2024-01-01 \
  --end 2024-01-31 \
  --parameters '{"rsi_period": 21, "rsi_oversold": 25, "rsi_overbought": 75}'
```

### Available Strategy Parameters

```bash
# See parameters for a specific strategy
python -m src.cli.main strategy show-parameters rsi_mean_reversion
```

## Data Management

### Check Available Data

```bash
python -m src.cli.main data list-data
```

### Update Existing Data

```bash
python -m src.cli.main data update
```

### Download Historical Data

```bash
# Download 2 years of data
python -m src.cli.main data download \
  --symbol BTC/USDT \
  --timeframe 1h \
  --start 2022-01-01 \
  --end 2024-12-31
```

## Risk Management Features

The system includes comprehensive risk management:

- **Position Sizing**: Based on account risk percentage
- **Daily/Weekly/Monthly Loss Limits**: Automatic trade suspension
- **Maximum Positions**: Limit concurrent trades
- **Correlation Limits**: Prevent over-concentration
- **Stop Loss/Take Profit**: Automatic risk controls

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **TA-Lib Installation**: 
   - Windows: Download from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
   - macOS: `brew install ta-lib`
   - Linux: Install ta-lib development package

3. **Database Errors**: Check that the data directory exists and is writable

4. **No Data Found**: Download data first using the CLI or examples

### Getting Help

```bash
# CLI help
python -m src.cli.main --help

# Specific command help
python -m src.cli.main backtest run --help

# System information
python -m src.cli.main info
```

## Next Steps

1. **Explore Examples**: Run `python examples.py` to see all features
2. **Create Custom Strategies**: Modify templates in `src/strategies/templates.py`
3. **Set Up MCP**: Integrate with Claude Desktop or VS Code Copilot
4. **Optimize Parameters**: Use the parameter optimization features
5. **Multi-Timeframe Analysis**: Test strategies across different timeframes

## Performance Tips

1. **Use appropriate timeframes**: Lower timeframes require more memory
2. **Limit data range**: Start with shorter periods for testing
3. **Batch operations**: Use multi-symbol backtests for efficiency
4. **Database optimization**: The SQLite database is optimized for read performance

## Support

- Check the README.md for detailed documentation
- Look at examples.py for code samples
- Review strategy templates for implementation patterns
- Use the CLI help system for command-specific guidance
