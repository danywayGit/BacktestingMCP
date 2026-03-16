# BacktestingMCP

GPU-accelerated cryptocurrency backtesting platform with CLI, MCP server, and AI strategy generation.
Optimized for NVIDIA RTX 4090 (CUDA 12.x) using CuPy + Numba (~1,145 optimization tests/sec).

---

## Features

- **6 built-in strategy templates** — MA Crossover, RSI Mean Reversion, EMA+RSI, Bollinger Bands, ATR Breakout, Momentum
- **DCA strategies** — fixed monthly or signal-based Dollar-Cost Averaging
- **GPU optimization** — CuPy + Numba JIT for parameter sweeps, automatic CPU fallback
- **Data management** — Binance/CCXT historical data download, SQLite storage, incremental updates
- **Risk management** — ATR/volatility/Kelly position sizing, daily/weekly loss limits, correlation tracking
- **AI strategy generation** — natural language → Python strategy code (OpenAI, Anthropic, Ollama)
- **MCP server** — 6 tools for AI assistants (VS Code Copilot, Claude Desktop, etc.)
- **Full CLI** — data, backtest, strategy, results commands

---

## Quick Start

### 1. Set up the environment

```powershell
# Windows
python setup_venv.py
.\venv\Scripts\Activate.ps1
```
```bash
# macOS / Linux
python setup_venv.py
source venv/bin/activate
```

### 2. Launch the interactive menu

```bash
python run.py
```

```
==================================================
  BacktestingMCP
==================================================
  1  Download market data
  2  List available data
  3  List strategies
  4  Run a backtest
  5  GPU optimization
  6  Generate AI strategy
  7  Start MCP server
  8  Check GPU status
  9  Inspect database
  t  Run tutorial walkthrough
  q  Quit
==================================================
```

### 3. Or use the CLI directly

```bash
# Download BTC/USDT hourly data for 2024
python -m src.cli.main data download --symbol BTC/USDT --timeframe 1h --start 2024-01-01 --end 2024-12-31

# Run a backtest
python -m src.cli.main backtest run \
  --strategy moving_average_crossover \
  --symbol BTCUSDT --timeframe 1h \
  --start 2024-01-01 --end 2024-06-01 \
  --cash 10000

# Multi-symbol comparison
python -m src.cli.main backtest multi-symbol \
  --strategy rsi_mean_reversion \
  --symbols BTCUSDT ETHUSDT BNBUSDT \
  --timeframe 1h --start 2024-01-01 --end 2024-06-01
```

---

## CLI Reference

```
python -m src.cli.main <group> <command> [options]
```

| Group | Command | What it does |
|-------|---------|-------------|
| `data` | `download` | Download data for a symbol |
| `data` | `download-multiple` | Download multiple symbols |
| `data` | `update` | Incrementally refresh all data |
| `data` | `list-data` | Show available data (symbol / timeframe / date range) |
| `strategy` | `list-strategies` | Show all available strategies |
| `strategy` | `show-parameters` | Show parameters for a strategy |
| `strategy` | `create` | Generate a strategy from natural language (AI) |
| `backtest` | `run` | Run a single backtest |
| `backtest` | `multi-symbol` | Backtest across multiple symbols |
| `results` | `list-results` | View recent backtest results |

---

## Examples

See [examples/README.md](examples/README.md) for full descriptions.

| Script | Purpose |
|--------|---------|
| `examples/00_tutorial.py` | Walkthrough of all features |
| `examples/01_basic_backtest.py` | Simple MA Crossover on real BTC data |
| `examples/02_gpu_optimization.py` | **Recommended** — GPU DCA sweep (~1,145 tests/sec) |
| `examples/03_ema_crossover.py` | EMA+RSI optimization on 4H timeframe |
| `examples/04_ai_strategy_gen.py` | Generate strategy from natural language |

Run any example from the project root:

```bash
python examples/01_basic_backtest.py
```

---

## Python API

```python
from src import backtest_engine, Database, list_strategies
from src import GPUOptimizer, check_gpu_status, StrategyGenerator

# List strategies
print(list_strategies())

# Run a backtest
result = backtest_engine.run_backtest(
    strategy_class=MyStrategy,
    symbol="BTCUSDT",
    timeframe="1h",
    start_date="2024-01-01",
    end_date="2024-06-01",
)

# Check GPU
check_gpu_status()
```

---

## Project Structure

```
BacktestingMCP/
├── run.py                   # Interactive launcher (start here)
├── src/
│   ├── __init__.py          # Public API (from src import ...)
│   ├── core/                # Backtesting engine (backtesting.py)
│   ├── data/                # Downloader + SQLite database
│   ├── strategies/          # 6 templates + DCA strategies + AI-generated
│   ├── optimization/        # GPU optimizer (CuPy + Numba)
│   ├── risk/                # Position sizing, limits, correlation
│   ├── ai/                  # AI strategy generator
│   ├── mcp/                 # MCP server
│   └── cli/                 # CLI (main.py)
├── examples/                # Numbered example scripts (00–04)
│   └── archive/             # Older / superseded scripts
├── scripts/                 # DCA comparison utilities
├── tools/                   # Developer utilities
│   ├── check_gpu.py         # Verify GPU/CUDA setup
│   ├── check_db.py          # Inspect database
│   ├── param_sensitivity.py # Parameter sensitivity analysis
│   └── signal_dca_test.py   # DCA signal testing
├── config/
│   └── settings.py          # All configuration (timeframes, risk, MCP, etc.)
├── docs/
│   ├── GPU_GUIDE.md         # GPU setup and performance guide
│   ├── DCA_STRATEGIES_GUIDE.md
│   ├── PARAMETER_FORMAT_GUIDE.md
│   └── OLLAMA_RTX4090_SETUP.md
└── data/
    └── crypto.db            # SQLite (created on first data download)
```

---

## MCP Server

Exposes 6 tools for AI assistants (Claude Desktop, VS Code Copilot, etc.):

```bash
python -m src.mcp.server   # starts on localhost:8000
```

Tools: `download_crypto_data`, `list_available_data`, `list_strategies`,
`get_strategy_info`, `run_backtest`, `create_strategy`.

---

## GPU Setup

See [docs/GPU_GUIDE.md](docs/GPU_GUIDE.md) for full details.

```bash
# Install CuPy for CUDA 12.x
pip install cupy-cuda12x

# Verify
python tools/check_gpu.py
```

All GPU scripts fall back to NumPy/CPU automatically when CuPy is unavailable.

---

## Configuration

Edit `config/settings.py` to adjust:
- `RiskConfig` — account risk %, daily/weekly/monthly loss limits, position limits
- `TradingConfig` — direction (long/short/both), commission, slippage, day/hour filters
- `DataConfig` — database path, exchange, default symbols
- `MCPConfig` — server host/port
- `OptimizationConfig` — method (optuna/grid/random), trial count

---

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA 12.x (optional, for GPU acceleration)
- See `requirements.txt` for full dependency list

Key packages: `backtesting`, `vectorbt`, `cupy-cuda12x`, `numba==0.56.4`, `numpy==1.23.5`, `pandas`, `ta`, `ccxt`
