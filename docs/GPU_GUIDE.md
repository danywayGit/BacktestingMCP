# GPU Acceleration Guide

GPU acceleration via CuPy + Numba gives ~10-50x speedup for indicator calculations and optimization sweeps on an NVIDIA RTX 4090 (CUDA 12.x).

---

## Verified Setup

```
CuPy version  : 13.6.0
GPU           : NVIDIA GeForce RTX 4090
Compute cap.  : 8.9
CUDA version  : 12.9
VRAM          : 24 GB
```

Run `python tools/check_gpu.py` at any time to confirm your GPU is detected.

---

## GPU Module: `src/optimization/gpu_optimizer.py`

### Key classes

| Class | Purpose |
|-------|---------|
| `GPUIndicators` | Vectorized EMA, RSI, SMA, MACD, Bollinger Bands on GPU |
| `GPUOptimizer` | Hybrid controller: GPU for indicator math, CPU for strategy state |
| `GPUBacktester` | Vectorized backtest wrapper |

### Key functions

```python
from src.optimization import check_gpu_status, optimize_with_gpu, GPU_AVAILABLE

check_gpu_status()          # Print GPU info and VRAM usage
optimize_with_gpu(...)      # High-level optimization entry point
```

---

## How the Hybrid Approach Works

```
┌──────────── GPU Phase (fast math) ────────────┐
│  1. Transfer price array to VRAM               │
│  2. Compute ALL indicator variants at once      │
│     • EMA (multiple periods)                   │
│     • RSI (multiple periods)                   │
│     • SMA, MACD, Bollinger Bands               │
│  3. Store results in GPU memory (~5 sec)        │
└───────────────────────────────────────────────┘
            ↓ transfer slices to CPU
┌──────────── CPU Phase (accurate state) ───────┐
│  4. For each parameter combination:            │
│     • Use pre-calculated GPU indicators        │
│     • Run strategy buy/sell logic (stateful)   │
│     • Calculate Sharpe, drawdown, etc.         │
└───────────────────────────────────────────────┘
```

Sequential state (cash tracking, position sizing) cannot be trivially parallelised on GPU, hence the hybrid design.

---

## Performance Benchmarks

| Method | Speed | Accuracy | Script |
|--------|-------|----------|--------|
| Pure CuPy + Numba JIT | **~1,145 tests/sec** | ✅ 100% | `02_gpu_optimization.py` |
| Hybrid GPU indicators + CPU strategy | 10–30 tests/sec | ✅ 100% | `03_ema_crossover.py` |
| Numba CUDA only | Very fast | ❌ ~80% | archived |
| CPU parallel (multiprocessing) | ~10 tests/sec | ✅ 100% | archived |
| CPU single-threaded | ~2 tests/sec | ✅ 100% | archived |

---

## Memory Management

Safe batch size for 24 GB VRAM: **25,000 parameter combinations per GPU batch**.

```python
import cupy as cp

# Check current VRAM usage
pool = cp.get_default_memory_pool()
used_gb  = pool.used_bytes() / 1024**3
total_gb = pool.total_bytes() / 1024**3
print(f"VRAM: {used_gb:.2f} / {total_gb:.2f} GB used")

# Free unused cached memory between large runs
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()
```

---

## CPU Fallback

Every GPU script falls back to NumPy automatically when CuPy is unavailable:

```python
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    import numpy as cp   # transparent fallback
    GPU_AVAILABLE = False
```

---

## Installing CuPy (CUDA 12.x)

```bash
pip install cupy-cuda12x
```

For other CUDA versions: https://docs.cupy.dev/en/stable/install.html

---

## GPU MCP Server — expose the 4090 to Hermes over LAN

`src/mcp/gpu_server.py` is a standalone MCP server that exposes the GPU's
optimization/backtest tools to an agent (Hermes) over the network. Hermes can
then call `gpu_health`, `run_exit_geometry_sweep`, `run_sweep_async`,
`sweep_status`, `sweep_result` directly — running sweeps on the GPU and getting
the results back.

### Re-run this server (Windows 11 + RTX 4090)

**Step 1 — deps** (one-time): the venv needs FastMCP + the supported `mcp` library:

```bat
pip install "fastmcp" "mcp[server]"
```

**Step 2 — start the server.** ⚠️ CRITICAL: you must run it from a directory
OUTSIDE the repo AND clear `PYTHONPATH`, because the repo's own `src/mcp/`
shadow-clashes with the installed `mcp` library (a circular-import error:
`cannot import name 'Server'`). From your Windows home directory:

```bat
cd C:\Users\<you>
set PYTHONPATH=
python C:\Users\<you>\Documents\Git\DanywayGit\BacktestingMCP\src\mcp\gpu_server.py
```

Expected output (keep the window open):
```
INFO: Uvicorn running on http://0.0.0.0:8765 (Press CTRL+C to quit)
```
Serves Streamable HTTP MCP at `http://0.0.0.0:8765/mcp`.

> **Path gotcha:** the repo has its own `src/mcp/server.py`. If `src/` is on
> `sys.path` (cwd = repo, or a `PYTHONPATH` / editable-install `.pth` points at
> `src`), then `import mcp` loads the repo file instead of the real library and
> FastMCP fails with the circular-import error above. cd out + `set PYTHONPATH=`
> fixes it. Do NOT run from the repo root.

### Connect Hermes to it

On the Hermes box (or any MCP client that can reach `192.168.1.7`):

```bash
hermes mcp add gpu4090 --url http://192.168.1.7:8765/mcp
# answers: auth = n (none), enable all = y
hermes mcp test gpu4090        # expect ✓ Connected + 5 tools
```

Or add to `~/.hermes/config.yaml`:
```yaml
mcp_servers:
  gpu4090:
    url: http://192.168.1.7:8765/mcp
    enabled: true
```

Then `/reload-mcp` (or new session) and the GPU tools are usable as normal tools.

### Tools exposed

| Tool | Purpose |
|------|---------|
| `gpu_health` | GPU name / VRAM / numba availability (fast probe) |
| `run_exit_geometry_sweep` | vectorized `atr_stop_mult × rr_ratio × max_hold` sweep on an OHLC close series; returns ranked combos by EV(R) |
| `run_sweep_async` | submit a long sweep as a background job → returns `job_id` |
| `sweep_status` / `sweep_result` | poll / fetch a completed job's results (persisted to `results/gpu_mcp/`) |

**Safety:** this server exposes *compute* tools only (no trading/order actions).
It reads OHLC and returns ranked results — it never touches exchange accounts.
