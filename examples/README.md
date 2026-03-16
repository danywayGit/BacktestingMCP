# Examples

Five numbered scripts — run them in order to learn the system, or jump straight to the one you need.

| Script | What it does | GPU needed? |
|--------|-------------|-------------|
| [00_tutorial.py](00_tutorial.py) | Walks through data download, strategy listing, and CLI commands. Start here. | No |
| [01_basic_backtest.py](01_basic_backtest.py) | Runs a Simple MA Crossover backtest on real BTC data. | No |
| [02_gpu_optimization.py](02_gpu_optimization.py) | **Recommended optimizer** — Pure CuPy DCA optimization (~1,145 tests/sec on RTX 4090). | Yes (falls back to CPU) |
| [03_ema_crossover.py](03_ema_crossover.py) | Optimizes EMA Crossover + RSI strategy on 4H BTCUSDT/ETHUSDT/SOLUSDT. | Yes (falls back to CPU) |
| [04_ai_strategy_gen.py](04_ai_strategy_gen.py) | Generates a new strategy from natural language using OpenAI, Anthropic, or Ollama. | No |

## Running an example

```powershell
# From the project root, with venv activated:
.\venv\Scripts\Activate.ps1
python examples\01_basic_backtest.py
```

Or use the interactive launcher:
```powershell
python run.py
```

## Archived scripts

Older or superseded variants are in [archive/](archive/). They still work but have been replaced by the scripts above:

| Script | Why archived |
|--------|-------------|
| `optimize_dca_gpu.py` | Superseded by `02_gpu_optimization.py` (pure CuPy is faster) |
| `optimize_dca_numba.py` | Numba CUDA approach — fast but inaccurate results |
| `optimize_dca_parallel.py` | CPU multiprocessing — no GPU utilization |
| `optimize_dca_signal.py` | Basic 81-combination CPU version |
| `optimize_dca_signal_enhanced.py` | 3,072-combination CPU version (~30-60 min) |
| `optimize_dca_vectorbt.py` | VectorBT framework — high RAM usage |
| `optimize_simple.py` | backtesting.py built-in optimizer only |
| `optimize_example.py` | Basic optimization example |
| `test_backtest.py` | Ad-hoc test script |
