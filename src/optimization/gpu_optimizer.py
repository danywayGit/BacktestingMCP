"""
GPU-Accelerated Optimization Module using CuPy
Leverages RTX 4090 for massive parallel speedup in indicator calculations and optimization.

This module provides GPU-accelerated functions for:
- Technical indicator calculations (EMA, RSI, SMA, etc.)
- Parallel backtesting across parameter combinations
- Hybrid CPU/GPU optimization strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Callable, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
    device = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(device.id)
    gpu_name = props['name'].decode('utf-8')
    print(f"✓ GPU Acceleration Available: {gpu_name}")
    print(f"✓ CUDA Version: {cp.cuda.runtime.runtimeGetVersion()}")
except ImportError:
    GPU_AVAILABLE = False
    cp = np  # Fallback to NumPy
    print("⚠️  CuPy not available - falling back to CPU (NumPy)")
except Exception as e:
    GPU_AVAILABLE = False
    cp = np
    print(f"⚠️  GPU initialization error: {e} - falling back to CPU")


class GPUIndicators:
    """GPU-accelerated technical indicator calculations."""
    
    @staticmethod
    def ema(prices: np.ndarray, period: int, use_gpu: bool = True) -> np.ndarray:
        """
        Calculate Exponential Moving Average on GPU.
        
        Args:
            prices: Array of prices
            period: EMA period
            use_gpu: Whether to use GPU (if available)
        
        Returns:
            EMA values as numpy array
        """
        if use_gpu and GPU_AVAILABLE:
            prices_gpu = cp.asarray(prices)
            ema_gpu = cp.empty_like(prices_gpu)
            alpha = 2.0 / (period + 1)
            
            ema_gpu[0] = prices_gpu[0]
            for i in range(1, len(prices_gpu)):
                ema_gpu[i] = alpha * prices_gpu[i] + (1 - alpha) * ema_gpu[i-1]
            
            return cp.asnumpy(ema_gpu)
        else:
            # CPU fallback
            ema = np.empty(len(prices))
            alpha = 2.0 / (period + 1)
            ema[0] = prices[0]
            for i in range(1, len(prices)):
                ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
            return ema
    
    @staticmethod
    def rsi(prices: np.ndarray, period: int = 14, use_gpu: bool = True) -> np.ndarray:
        """
        Calculate Relative Strength Index using pandas rolling (fastest method).
        
        Args:
            prices: Array of prices
            period: RSI period
            use_gpu: Whether to use GPU (if available)
        
        Returns:
            RSI values as numpy array
        """
        import pandas as pd
        
        # Use pandas for fast rolling calculations (optimized in C)
        prices_series = pd.Series(prices)
        delta = prices_series.diff()
        
        gains = delta.where(delta > 0, 0.0)
        losses = -delta.where(delta < 0, 0.0)
        
        # Use exponential weighted mean (faster than manual loop)
        avg_gain = gains.ewm(span=period, adjust=False).mean()
        avg_loss = losses.ewm(span=period, adjust=False).mean()
        
        # Calculate RSI
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        # Fill NaN with 50
        rsi = rsi.fillna(50.0)
        
        return rsi.values
    
    @staticmethod
    def sma(prices: np.ndarray, period: int, use_gpu: bool = True) -> np.ndarray:
        """
        Calculate Simple Moving Average on GPU.
        
        Args:
            prices: Array of prices
            period: SMA period
            use_gpu: Whether to use GPU (if available)
        
        Returns:
            SMA values as numpy array
        """
        if use_gpu and GPU_AVAILABLE:
            prices_gpu = cp.asarray(prices)
            sma_gpu = cp.convolve(prices_gpu, cp.ones(period)/period, mode='same')
            return cp.asnumpy(sma_gpu)
        else:
            return np.convolve(prices, np.ones(period)/period, mode='same')
    
    @staticmethod
    def batch_calculate_indicators(
        prices: np.ndarray,
        rsi_periods: List[int],
        ema_periods: List[int],
        use_gpu: bool = True
    ) -> Dict[str, Dict[int, np.ndarray]]:
        """
        Calculate multiple indicators in batch on GPU for efficiency.
        
        Args:
            prices: Price array
            rsi_periods: List of RSI periods to calculate
            ema_periods: List of EMA periods to calculate
            use_gpu: Whether to use GPU
        
        Returns:
            Dictionary with 'rsi' and 'ema' keys, each containing period->values mapping
        """
        results = {'rsi': {}, 'ema': {}}
        
        print(f"Calculating {len(rsi_periods)} RSI + {len(ema_periods)} EMA indicators on {'GPU' if use_gpu and GPU_AVAILABLE else 'CPU'}...")
        
        # Calculate all RSI periods
        for period in rsi_periods:
            results['rsi'][period] = GPUIndicators.rsi(prices, period, use_gpu)
        
        # Calculate all EMA periods
        for period in ema_periods:
            results['ema'][period] = GPUIndicators.ema(prices, period, use_gpu)
        
        return results


class GPUOptimizer:
    """
    Hybrid GPU/CPU optimizer for backtesting strategies.
    
    Uses GPU for:
    - Fast indicator pre-calculation (EMA, RSI, etc.)
    - Parallel computation of technical signals
    
    Uses CPU for:
    - Complex strategy logic
    - Sequential state management (cash, positions)
    - Final backtest execution
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize GPU optimizer.
        
        Args:
            use_gpu: Whether to use GPU acceleration (if available)
        """
        self.use_gpu = use_gpu and GPU_AVAILABLE
        
        if self.use_gpu:
            device = cp.cuda.Device()
            props = cp.cuda.runtime.getDeviceProperties(device.id)
            gpu_name = props['name'].decode('utf-8')
            print(f"✓ GPU Optimizer initialized with {gpu_name}")
            # Get GPU memory info
            mempool = cp.get_default_memory_pool()
            print(f"✓ GPU Memory: {mempool.total_bytes() / 1024**3:.2f} GB total")
        else:
            print("⚠️  GPU Optimizer using CPU fallback")
    
    def pre_calculate_indicators(
        self,
        prices: np.ndarray,
        rsi_periods: List[int],
        ema_periods: List[int]
    ) -> Dict[str, Dict[int, np.ndarray]]:
        """
        Pre-calculate all required indicators on GPU.
        This is the main GPU acceleration point.
        
        Args:
            prices: Price data
            rsi_periods: List of RSI periods needed
            ema_periods: List of EMA periods needed
        
        Returns:
            Dictionary of pre-calculated indicators
        """
        start_time = datetime.now()
        
        indicators = GPUIndicators.batch_calculate_indicators(
            prices,
            rsi_periods,
            ema_periods,
            self.use_gpu
        )
        
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"✓ Indicators calculated in {elapsed:.2f}s")
        
        return indicators
    
    def optimize_parameters(
        self,
        prices: np.ndarray,
        param_combinations: List[Dict[str, Any]],
        backtest_function: Callable,
        pre_calculated_indicators: Optional[Dict] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Optimize strategy parameters using hybrid GPU/CPU approach.
        
        Args:
            prices: Price data
            param_combinations: List of parameter dictionaries to test
            backtest_function: Function that runs backtest with given params and indicators
            pre_calculated_indicators: Pre-calculated indicators (from pre_calculate_indicators)
            progress_callback: Optional callback for progress updates (current, total)
        
        Returns:
            List of results for each parameter combination
        """
        total = len(param_combinations)
        print(f"\nOptimizing {total} parameter combinations...")
        print(f"Using {'GPU-accelerated indicators + CPU backtesting' if self.use_gpu else 'CPU-only'}")
        print()
        
        results = []
        
        for i, params in enumerate(param_combinations, 1):
            try:
                # Run backtest with pre-calculated indicators
                result = backtest_function(
                    prices=prices,
                    params=params,
                    indicators=pre_calculated_indicators
                )
                results.append(result)
                
                # Progress update
                if progress_callback:
                    progress_callback(i, total)
                elif i % 100 == 0 or i == total:
                    print(f"Progress: {i}/{total} ({i*100//total}%)")
                
            except Exception as e:
                print(f"Error with params {params}: {e}")
                continue
        
        print(f"\n✓ Optimization complete: {len(results)}/{total} successful")
        return results
    
    def cleanup(self):
        """Clean up GPU memory."""
        if self.use_gpu and GPU_AVAILABLE:
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            print("✓ GPU memory cleaned up")


# Utility functions for easy integration

def optimize_with_gpu(
    prices: np.ndarray,
    param_grid: Dict[str, List],
    backtest_func: Callable,
    rsi_periods: Optional[List[int]] = None,
    ema_periods: Optional[List[int]] = None,
    validation_func: Optional[Callable[[Dict], bool]] = None,
    use_gpu: bool = True
) -> pd.DataFrame:
    """
    High-level function for GPU-accelerated optimization.
    
    Args:
        prices: Price data array
        param_grid: Dictionary of parameter names to lists of values
        backtest_func: Backtest function(prices, params, indicators) -> result_dict
        rsi_periods: RSI periods to pre-calculate (extracted from param_grid if None)
        ema_periods: EMA periods to pre-calculate (extracted from param_grid if None)
        validation_func: Optional function to validate parameter combinations
        use_gpu: Whether to use GPU acceleration
    
    Returns:
        DataFrame with optimization results
    
    Example:
        param_grid = {
            'rsi_period': [14, 21],
            'ema_period': [150, 200],
            'min_score': [4, 5, 6]
        }
        
        results = optimize_with_gpu(
            prices=df['Close'].values,
            param_grid=param_grid,
            backtest_func=my_backtest_function,
            use_gpu=True
        )
    """
    from itertools import product
    
    # Extract indicator periods from param_grid if not provided
    if rsi_periods is None:
        rsi_periods = list(set(param_grid.get('rsi_period', [14])))
    if ema_periods is None:
        ema_periods = list(set(param_grid.get('ema_period', [200])))
    
    # Generate parameter combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combinations = [dict(zip(keys, combo)) for combo in product(*values)]
    
    # Apply validation if provided
    if validation_func:
        param_combinations = [p for p in all_combinations if validation_func(p)]
        print(f"Valid combinations: {len(param_combinations)}/{len(all_combinations)}")
    else:
        param_combinations = all_combinations
    
    # Initialize GPU optimizer
    optimizer = GPUOptimizer(use_gpu=use_gpu)
    
    # Pre-calculate indicators on GPU
    print("\nPre-calculating indicators on GPU...")
    indicators = optimizer.pre_calculate_indicators(
        prices=prices,
        rsi_periods=rsi_periods,
        ema_periods=ema_periods
    )
    
    # Run optimization
    results = optimizer.optimize_parameters(
        prices=prices,
        param_combinations=param_combinations,
        backtest_function=backtest_func,
        pre_calculated_indicators=indicators
    )
    
    # Cleanup
    optimizer.cleanup()
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    return results_df


def check_gpu_status() -> Dict[str, Any]:
    """
    Check GPU availability and status.
    
    Returns:
        Dictionary with GPU information
    """
    info = {
        'gpu_available': GPU_AVAILABLE,
        'backend': 'CuPy' if GPU_AVAILABLE else 'NumPy (CPU)',
    }
    
    if GPU_AVAILABLE:
        try:
            device = cp.cuda.Device()
            props = cp.cuda.runtime.getDeviceProperties(device.id)
            info['gpu_name'] = props['name'].decode('utf-8')
            info['compute_capability'] = device.compute_capability
            info['cuda_version'] = cp.cuda.runtime.runtimeGetVersion()
            
            mempool = cp.get_default_memory_pool()
            info['total_memory_gb'] = mempool.total_bytes() / 1024**3
            info['used_memory_gb'] = mempool.used_bytes() / 1024**3
            info['free_memory_gb'] = (mempool.total_bytes() - mempool.used_bytes()) / 1024**3
        except Exception as e:
            info['error'] = str(e)
    
    return info


# Print GPU status on module import
if __name__ != "__main__":
    status = check_gpu_status()
    if status['gpu_available']:
        print(f"\n{'='*70}")
        print("GPU ACCELERATION ENABLED")
        print(f"{'='*70}")
        print(f"Device: {status['gpu_name']}")
        print(f"Compute Capability: {status['compute_capability']}")
        print(f"CUDA Version: {status['cuda_version']}")
        print(f"GPU Memory: {status['total_memory_gb']:.2f} GB total, {status['free_memory_gb']:.2f} GB free")
        print(f"{'='*70}\n")
