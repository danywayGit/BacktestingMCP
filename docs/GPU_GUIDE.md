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
