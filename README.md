# BacktestingMCP

GPU-accelerated cryptocurrency backtesting platform with CLI, MCP server, and AI strategy generation.
Optimized for NVIDIA RTX 4090 (CUDA 12.x) using CuPy + Numba (~1,145 optimization tests/sec).

---

## Features

- **6 built-in strategy templates** — MA Crossover, RSI Mean Reversion, EMA+RSI, Bollinger Bands, ATR Breakout, Momentum
- **DCA strategies** — fixed monthly or signal-based Dollar-Cost Averaging
- **Breakout scanners** — scan symbols for unusual-volume and breakout setups
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
  a  Run market scanner
  b  Compare breakout strategies
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

# Optimize parameters (target Sharpe ratio)
python -m src.cli.main backtest optimize \
  --strategy rsi_mean_reversion --symbol BTCUSDT --timeframe 1h \
  --start 2020-01-01 --end 2022-12-31 \
  --objective sharpe_ratio \
  --param-grid '{"rsi_period":[10,14,20],"rsi_oversold":[25,30,35]}'

# Walk-forward validation (70% train / 30% test)
python -m src.cli.main backtest walk-forward \
  --strategy rsi_mean_reversion --symbol BTCUSDT --timeframe 1h \
  --start 2020-01-01 --end 2025-12-31 \
  --train-ratio 0.7

# Multi-symbol comparison
python -m src.cli.main backtest multi-symbol \
  --strategy rsi_mean_reversion \
  --symbols BTCUSDT ETHUSDT BNBUSDT \
  --timeframe 1h --start 2024-01-01 --end 2024-06-01

# Scan symbols for breakout candidates
python -m src.cli.main scan run \
  --scan all \
  --symbols BTCUSDT ETHUSDT SOLUSDT \
  --timeframe 1h --start 2024-01-01 --end 2024-06-01

# Compare all built-in breakout strategies on one symbol
python -m src.cli.main backtest compare-breakouts \
  --symbol BTCUSDT --timeframe 1h \
  --start 2024-01-01 --end 2024-06-01 \
  --sort-by sharpe_ratio
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
| `backtest` | `optimize` | Sweep a parameter grid, target Sharpe / SQN / Profit Factor / etc. |
| `backtest` | `walk-forward` | Train/test split — detect overfitting |
| `backtest` | `multi-symbol` | Backtest across multiple symbols |
| `backtest` | `compare-breakouts` | Run and rank all built-in breakout strategies on one symbol |
| `scan` | `run` | Run one or all breakout scans on latest candle(s) for selected symbols |
| `results` | `list-results` | View recent backtest results |

### Scan command quick notes

- Available scan types: `unusual_volume_breakout`, `new_local_high_breakout`, `resistance_breakout`, `ascending_triangle_breakout`, `all`
- Use `--symbols` for specific symbols or `--all-major` to scan the built-in major list
- Optional `--parameters` accepts JSON to override scan thresholds

Scanner profile guidance:

- Apply predefined parameter bundles for scanner thresholds and breakout strictness.
- Aggressive finds more signals with looser filters.
- Conservative finds fewer signals with stricter confirmation.
- Normal sits between the two.

Example with parameter overrides:

```bash
python -m src.cli.main scan run \
  --scan unusual_volume_breakout \
  --symbols BTCUSDT \
  --timeframe 1h --start 2024-01-01 --end 2024-06-01 \
  --parameters '{"volume_multiplier": 1.8, "breakout_lookback": 30}'
```

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
from src import backtest_engine, database, list_available_strategies
from src import GPUOptimizer, check_gpu_status, StrategyGenerator

# List strategies
print(list_available_strategies())

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
│   ├── GPU_GUIDE.md              # GPU setup and performance guide
│   ├── STRATEGY_WORKFLOW_FAQ.md  # Strategy creation → optimization → validation FAQ
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
