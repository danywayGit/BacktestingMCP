# Copilot Instructions for BacktestingMCP

## Environment Setup - CRITICAL

**ALWAYS activate the virtual environment before running ANY Python commands or scripts:**

```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Windows CMD
.\venv\Scripts\activate.bat
```

```bash
# Linux/macOS
source ./venv/bin/activate
```

The venv contains all required dependencies including CuPy, Numba, VectorBT, and other specialized packages. **Never install packages globally.**

---

## Project Overview

A GPU-accelerated cryptocurrency backtesting platform with:
- **GPU Optimization**: NVIDIA GPU with CUDA support (tested on RTX 4090 24GB), CuPy, Numba JIT
- **Database**: SQLite at `data/crypto.db` with historical BTC/USDT candles
- **MCP Integration**: Model Context Protocol server for AI-assisted development
- **Frameworks**: backtesting.py, VectorBT, pandas, ta (technical analysis)

---

## Key Architecture

```
src/
├── core/           # Backtesting engine (backtesting.py framework)
├── strategies/     # Strategy classes (templates.py, dca_strategies.py)
├── optimization/   # GPU-accelerated optimization (gpu_optimizer.py)
├── data/           # Data download/database (CCXT/Binance integration)
├── ai/             # AI strategy generation (OpenAI, Anthropic, Ollama)
├── mcp/            # MCP server implementation
├── risk/           # Position sizing, limits, correlation
└── cli/            # Command-line interface
```

---

## GPU Optimization Patterns

### Use CuPy for Vectorized Calculations
```python
import cupy as cp

# GPU-accelerated indicator calculation
prices_gpu = cp.asarray(prices)  # Transfer to GPU
# ... compute on GPU ...
result = cp.asnumpy(result_gpu)  # Transfer back to CPU
```

### Use Numba JIT for Sequential Portfolio Simulation
```python
from numba import njit, prange

@njit(parallel=True)
def simulate_portfolios(params, signals, prices):
    # Parallel CPU execution - GPU can't handle sequential state dependencies
    for i in prange(len(params)):
        # Portfolio simulation with cash state tracking
        ...
```

### Batch Size for 24GB VRAM
- Safe batch size: **25,000 parameter combinations per GPU batch**
- Monitor with `cp.get_default_memory_pool().used_bytes()`

---

## Database Access

```python
from src.data.database import Database

db = Database("data/crypto.db")
df = db.get_ohlcv("BTC/USDT", "1h", start_date, end_date)
# Returns: DataFrame with Open, High, Low, Close, Volume columns
```

---

## Strategy Development

### Existing Strategies
- `src/strategies/templates.py` - 6 template strategies with registry
- `src/strategies/dca_strategies.py` - DCAMonthlyStrategy, DCASignalStrategy

### Creating New Strategies
Extend `BaseStrategy` from backtesting.py:
```python
from backtesting import Strategy

class MyStrategy(Strategy):
    param1 = 14  # Optimizable parameter
    
    def init(self):
        self.indicator = self.I(calculate_indicator, self.data.Close, self.param1)
    
    def next(self):
        if self.indicator[-1] < threshold:
            self.buy()
```

---

## CLI Commands

```bash
# Always activate venv first!
python -m src.cli.main data download --symbol BTC/USDT --timeframe 1h --start 2024-01-01
python -m src.cli.main backtest run --strategy moving_average_crossover --symbol BTCUSDT --timeframe 1h --start 2024-01-01 --end 2024-06-01
python -m src.cli.main strategy list-strategies
```

---

## Optimization Scripts

Located in `examples/`:
- `optimize_dca_cupy.py` - Pure CuPy/Numba optimization (recommended, 1,145 tests/sec)
- `optimize_dca_vectorbt.py` - VectorBT-based (CPU-bound, high RAM usage)

---

## Dependencies

Key packages (pinned versions for compatibility):
- `numpy==1.23.5` - Pinned for Numba compatibility
- `numba==0.56.4` - JIT compilation
- `cupy-cuda12x` - GPU acceleration (CUDA 12.x)
- `vectorbt==0.28.1` - Portfolio backtesting framework

---

## Important Notes

1. **Always use venv** - Never install packages globally
2. **GPU memory management** - Clear CuPy memory pools between large operations
3. **Portfolio simulation limits** - Sequential state dependencies prevent full GPU parallelization
4. **Technical analysis** - Use `ta` library (not TA-Lib) for easier installation
