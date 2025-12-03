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


# DCA Strategy Optimization Runner
# Default parameter grid for DCA-style strategies
DEFAULT_DCA_PARAM_GRID = {
    'rsi_period': [14, 18, 21],
    'ema_period': [150, 200, 250],
    'min_signal_score': [3, 4, 5, 6],
    'strong_signal_threshold': [7, 8, 9],
    'extreme_signal_score': [9, 10, 11],
    'rsi_oversold_extreme': [28, 30, 33, 35],
    'rsi_oversold_moderate': [38, 40, 43, 45],
    'ema_distance_extreme': [-22, -20, -18, -16],
    'ema_distance_strong': [-16, -14, -12, -10],
    'strong_signal_multiplier': [1.1, 1.2, 1.3, 1.4, 1.5],
    'extreme_signal_multiplier': [1.3, 1.4, 1.5, 1.6, 1.7]
}


def _validate_dca_params(p: Dict) -> bool:
    """Validate a DCA parameter combination for logical consistency."""
    return (
        p['min_signal_score'] < p['extreme_signal_score'] and
        p['min_signal_score'] < p['strong_signal_threshold'] and
        p['strong_signal_threshold'] < p['extreme_signal_score'] and
        p['rsi_oversold_extreme'] < p['rsi_oversold_moderate'] and
        p['ema_distance_extreme'] < p['ema_distance_strong'] and
        p['ema_distance_strong'] < -8 and
        p['strong_signal_multiplier'] < p['extreme_signal_multiplier']
    )


def _generate_signals_gpu_batch(prices_gpu, indicators: Dict, param_batch: List[Dict]) -> Tuple:
    """Generate buy signals for a batch of parameters on GPU."""
    n_bars = len(prices_gpu)
    n_params = len(param_batch)
    
    signals = cp.zeros((n_bars, n_params), dtype=cp.bool_)
    multipliers = cp.ones((n_bars, n_params), dtype=cp.float32)
    
    for idx, params in enumerate(param_batch):
        rsi = cp.asarray(indicators['rsi'][params['rsi_period']], dtype=cp.float32)
        ema = cp.asarray(indicators['ema'][params['ema_period']], dtype=cp.float32)
        
        scores = cp.zeros(n_bars, dtype=cp.float32)
        
        # EMA distance scoring
        ema_distance_pct = ((prices_gpu - ema) / ema) * 100.0
        scores += cp.where(ema_distance_pct <= params['ema_distance_extreme'], 3, 0)
        scores += cp.where(
            (ema_distance_pct > params['ema_distance_extreme']) & 
            (ema_distance_pct <= params['ema_distance_strong']), 
            2, 0
        )
        scores += cp.where(
            (ema_distance_pct > params['ema_distance_strong']) & 
            (ema_distance_pct <= -8), 
            1, 0
        )
        
        # RSI scoring
        scores += cp.where(rsi < params['rsi_oversold_extreme'], 2, 0)
        scores += cp.where(
            (rsi >= params['rsi_oversold_extreme']) & 
            (rsi < params['rsi_oversold_moderate']), 
            1, 0
        )
        
        # Generate signals
        param_signals = (scores >= params['min_signal_score']) & (prices_gpu < ema)
        
        # Calculate multipliers
        param_mult = cp.ones(n_bars, dtype=cp.float32)
        param_mult = cp.where(scores >= params['extreme_signal_score'], 
                              params['extreme_signal_multiplier'], param_mult)
        param_mult = cp.where(
            (scores >= params['strong_signal_threshold']) & (scores < params['extreme_signal_score']),
            params['strong_signal_multiplier'],
            param_mult
        )
        
        signals[:, idx] = param_signals
        multipliers[:, idx] = param_mult
    
    return signals, multipliers


try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
    
    @jit(nopython=True, parallel=True)
    def _simulate_portfolios_numba(prices, signals, multipliers, n_params, n_bars,
                                    initial_cash, monthly_contribution, commission, bars_per_month):
        """Numba JIT-compiled portfolio simulation. Runs in parallel across CPU cores."""
        returns = np.zeros(n_params, dtype=np.float32)
        final_equities = np.zeros(n_params, dtype=np.float32)
        trade_counts = np.zeros(n_params, dtype=np.int32)
        
        for i in prange(n_params):
            cash = initial_cash
            btc = 0.0
            total_invested = initial_cash
            trades = 0
            
            for bar in range(n_bars):
                if bar % bars_per_month == 0 and bar > 0:
                    cash += monthly_contribution
                    total_invested += monthly_contribution
                
                if signals[bar, i]:
                    buy_amount = min(multipliers[bar, i] * monthly_contribution, cash)
                    cost_with_fee = buy_amount * (1.0 + commission)
                    
                    if cash >= cost_with_fee:
                        btc_bought = buy_amount / prices[bar]
                        btc += btc_bought
                        cash -= cost_with_fee
                        trades += 1
            
            final_price = prices[n_bars - 1]
            final_equity = cash + (btc * final_price)
            total_return_pct = ((final_equity - total_invested) / total_invested) * 100.0
            
            returns[i] = total_return_pct
            final_equities[i] = final_equity
            trade_counts[i] = trades
        
        return returns, final_equities, trade_counts
except ImportError:
    NUMBA_AVAILABLE = False


def run_dca_optimization(
    symbol: str,
    start_date: str,
    end_date: str,
    timeframe: str = '1h',
    param_ranges: Optional[Dict[str, List]] = None,
    db_path: str = "data/crypto.db",
    initial_cash: float = 5000.0,
    monthly_contribution: float = 1000.0,
    commission: float = 0.001,
    batch_size: int = 25000,
    top_n: int = 10
) -> Dict[str, Any]:
    """
    Run GPU-accelerated DCA strategy optimization.
    
    This is the main entry point for optimization from MCP or CLI.
    Uses CuPy for GPU-accelerated signal generation and Numba for 
    parallel portfolio simulation.
    
    Args:
        symbol: Trading symbol (e.g., 'BTC/USDT' or 'BTCUSDT')
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD
        timeframe: Data timeframe (default '1h')
        param_ranges: Custom parameter ranges (uses defaults if None)
        db_path: Path to SQLite database
        initial_cash: Starting capital
        monthly_contribution: Monthly investment amount
        commission: Trading commission rate
        batch_size: GPU batch size for memory management
        top_n: Number of top results to return
        
    Returns:
        Dictionary with optimization results including top_results, stats, etc.
    """
    import sqlite3
    from itertools import product
    
    if not GPU_AVAILABLE:
        return {
            'error': 'CuPy not available. Please install cupy-cuda12x for GPU acceleration.',
            'gpu_available': False
        }
    
    if not NUMBA_AVAILABLE:
        return {
            'error': 'Numba not available. Please install numba for JIT compilation.',
            'gpu_available': True
        }
    
    results = {
        'symbol': symbol,
        'timeframe': timeframe,
        'start_date': start_date,
        'end_date': end_date,
        'gpu_available': GPU_AVAILABLE,
        'top_results': [],
        'total_combinations': 0,
        'valid_combinations': 0,
        'optimization_time_seconds': 0,
        'throughput_tests_per_second': 0,
    }
    
    # Load data from database
    symbol_normalized = symbol.replace('/', '')
    conn = sqlite3.connect(db_path)
    
    start_ts = int(pd.Timestamp(start_date).timestamp())
    end_ts = int(pd.Timestamp(end_date).timestamp())
    
    query = """
        SELECT timestamp, close
        FROM market_data
        WHERE symbol = ?
        AND timeframe = ?
        AND timestamp >= ?
        AND timestamp <= ?
        ORDER BY timestamp
    """
    df = pd.read_sql_query(query, conn, params=(symbol_normalized, timeframe, start_ts, end_ts))
    conn.close()
    
    if len(df) == 0:
        results['error'] = f"No data found for {symbol_normalized} {timeframe} from {start_date} to {end_date}"
        return results
    
    prices_cpu = df['close'].values.astype(np.float32)
    results['data_points'] = len(prices_cpu)
    
    # Transfer to GPU
    prices_gpu = cp.asarray(prices_cpu, dtype=cp.float32)
    
    # Generate parameter combinations
    param_grid = {}
    for key, default_values in DEFAULT_DCA_PARAM_GRID.items():
        param_grid[key] = param_ranges.get(key, default_values) if param_ranges else default_values
    
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combinations = [dict(zip(keys, combo)) for combo in product(*values)]
    results['total_combinations'] = len(all_combinations)
    
    param_combinations = [p for p in all_combinations if _validate_dca_params(p)]
    results['valid_combinations'] = len(param_combinations)
    
    if len(param_combinations) == 0:
        results['error'] = "No valid parameter combinations generated"
        return results
    
    # Pre-calculate indicators on GPU
    rsi_periods = sorted(set(param_grid['rsi_period']))
    ema_periods = sorted(set(param_grid['ema_period']))
    
    indicator_start = datetime.now()
    indicators = GPUIndicators.batch_calculate_indicators(
        prices_cpu, rsi_periods, ema_periods, use_gpu=True
    )
    indicator_time = (datetime.now() - indicator_start).total_seconds()
    
    # Process in batches
    n_batches = (len(param_combinations) + batch_size - 1) // batch_size
    all_results = []
    
    optimization_start = datetime.now()
    bars_per_month = 720  # Approximate for 1h timeframe
    
    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(param_combinations))
        param_batch = param_combinations[batch_start:batch_end]
        
        # Generate signals on GPU
        signals_gpu, multipliers_gpu = _generate_signals_gpu_batch(
            prices_gpu, indicators, param_batch
        )
        
        # Simulate portfolios using Numba
        n_bars = len(prices_gpu)
        n_params = signals_gpu.shape[1]
        
        prices_np = cp.asnumpy(prices_gpu).astype(np.float32)
        signals_np = cp.asnumpy(signals_gpu)
        multipliers_np = cp.asnumpy(multipliers_gpu).astype(np.float32)
        
        returns, final_equities, trade_counts = _simulate_portfolios_numba(
            prices_np, signals_np, multipliers_np, n_params, n_bars,
            initial_cash, monthly_contribution, commission, bars_per_month
        )
        
        # Build results
        for i in range(n_params):
            result = param_batch[i].copy()
            result.update({
                'total_return_pct': float(returns[i]),
                'final_equity': float(final_equities[i]),
                'total_trades': int(trade_counts[i]),
            })
            all_results.append(result)
    
    optimization_time = (datetime.now() - optimization_start).total_seconds()
    total_time = indicator_time + optimization_time
    
    # Sort and get top results
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('total_return_pct', ascending=False)
    
    results['top_results'] = results_df.head(top_n).to_dict('records')
    results['optimization_time_seconds'] = round(total_time, 2)
    results['throughput_tests_per_second'] = round(len(param_combinations) / total_time, 1)
    results['indicator_time_seconds'] = round(indicator_time, 2)
    
    # Statistics
    results['stats'] = {
        'best_return_pct': round(float(results_df['total_return_pct'].max()), 2),
        'worst_return_pct': round(float(results_df['total_return_pct'].min()), 2),
        'mean_return_pct': round(float(results_df['total_return_pct'].mean()), 2),
        'median_return_pct': round(float(results_df['total_return_pct'].median()), 2),
    }
    
    # Cleanup GPU memory
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()
    
    return results
