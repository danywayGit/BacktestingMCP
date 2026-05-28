"""Optimization module with GPU acceleration support."""

try:
    from .gpu_optimizer import (
        GPUIndicators,
        GPUOptimizer,
        optimize_with_gpu,
        check_gpu_status,
        GPU_AVAILABLE
    )
except (ImportError, SystemError) as _e:
    import warnings
    warnings.warn(f"GPU optimizer unavailable (numba/cupy import failed: {_e}). Falling back to CPU.")

    GPU_AVAILABLE = False

    class GPUIndicators:
        pass

    class GPUOptimizer:
        pass

    def optimize_with_gpu(*args, **kwargs):
        raise RuntimeError("GPU optimizer not available — numba failed to load.")

    def check_gpu_status():
        return {"available": False, "reason": str(_e)}


__all__ = [
    'GPUIndicators',
    'GPUOptimizer',
    'optimize_with_gpu',
    'check_gpu_status',
    'GPU_AVAILABLE'
]
