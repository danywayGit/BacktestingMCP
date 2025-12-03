"""
Pure GPU-Accelerated DCA Strategy Optimization using CuPy
Runs EVERYTHING on GPU to maximize RTX 4090 utilization.

Strategy:
1. Load data once to GPU
2. Calculate indicators on GPU (vectorized)
3. Generate signals on GPU (vectorized for ALL params simultaneously)
4. Simulate portfolios on GPU (vectorized)
"""

import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
import warnings
from itertools import product
from numba import jit, prange
warnings.filterwarnings('ignore')

try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✓ CuPy GPU available")
except ImportError:
    GPU_AVAILABLE = False
    cp = np
    print("⚠️ CuPy not available, using NumPy")


def load_data(db_path, symbol, start_date, end_date, timeframe='1h'):
    """Load OHLCV data from database."""
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
    df = pd.read_sql_query(query, conn, params=(symbol, timeframe, start_ts, end_ts))
    conn.close()
    
    return df['close'].values


def calculate_indicators_gpu(prices_gpu, rsi_periods, ema_periods):
    """Calculate all indicators on GPU simultaneously."""
    print("\nCalculating indicators on GPU...")
    
    indicators = {}
    
    # Calculate all RSI values (vectorized on GPU)
    for period in rsi_periods:
        rsi = calculate_rsi_gpu(prices_gpu, period)
        indicators[f'rsi_{period}'] = rsi
    
    # Calculate all EMA values (vectorized on GPU)
    for period in ema_periods:
        ema = calculate_ema_gpu(prices_gpu, period)
        indicators[f'ema_{period}'] = ema
    
    print(f"✓ Indicators calculated on GPU")
    
    return indicators


def calculate_rsi_gpu(prices, period):
    """Calculate RSI on GPU using pandas (faster than manual loops)."""
    prices_cpu = cp.asnumpy(prices)
    prices_series = pd.Series(prices_cpu)
    delta = prices_series.diff()
    
    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)
    
    avg_gain = gains.ewm(span=period, adjust=False).mean()
    avg_loss = losses.ewm(span=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0)
    
    return cp.asarray(rsi.values, dtype=cp.float32)


def calculate_ema_gpu(prices, period):
    """Calculate EMA on GPU."""
    alpha = 2.0 / (period + 1)
    ema = cp.empty_like(prices)
    ema[0] = prices[0]
    
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
    
    return ema


def generate_signals_gpu_batch(prices_gpu, indicators, param_batch, batch_offset):
    """
    Generate buy signals for a batch of parameters on GPU.
    Returns signals and multipliers as GPU arrays.
    """
    n_bars = len(prices_gpu)
    n_params = len(param_batch)
    
    # Create GPU arrays for this batch
    signals = cp.zeros((n_bars, n_params), dtype=cp.bool_)
    multipliers = cp.ones((n_bars, n_params), dtype=cp.float32)
    
    # Process each parameter combination
    for idx, params in enumerate(param_batch):
        rsi = indicators[f"rsi_{params['rsi_period']}"]
        ema = indicators[f"ema_{params['ema_period']}"]
        
        # Calculate signal scores (vectorized on GPU)
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
        param_mult = cp.where(scores >= params['extreme_signal_score'], params['extreme_signal_multiplier'], param_mult)
        param_mult = cp.where(
            (scores >= params['strong_signal_threshold']) & (scores < params['extreme_signal_score']),
            params['strong_signal_multiplier'],
            param_mult
        )
        
        signals[:, idx] = param_signals
        multipliers[:, idx] = param_mult
    
    return signals, multipliers


@jit(nopython=True, parallel=True)
def simulate_portfolios_numba(prices, signals, multipliers, n_params, n_bars, 
                               initial_cash, monthly_contribution, commission, bars_per_month):
    """
    Simulate portfolios using Numba JIT compilation for MASSIVE speedup.
    Runs in parallel across multiple CPU cores.
    """
    returns = np.zeros(n_params, dtype=np.float32)
    final_equities = np.zeros(n_params, dtype=np.float32)
    trade_counts = np.zeros(n_params, dtype=np.int32)
    
    # Process each parameter in parallel
    for i in prange(n_params):
        cash = initial_cash
        btc = 0.0
        total_invested = initial_cash
        trades = 0
        
        # Process each bar
        for bar in range(n_bars):
            # Monthly contribution
            if bar % bars_per_month == 0 and bar > 0:
                cash += monthly_contribution
                total_invested += monthly_contribution
            
            # Execute buy if signal
            if signals[bar, i]:
                buy_amount = min(multipliers[bar, i] * monthly_contribution, cash)
                cost_with_fee = buy_amount * (1.0 + commission)
                
                if cash >= cost_with_fee:
                    btc_bought = buy_amount / prices[bar]
                    btc += btc_bought
                    cash -= cost_with_fee
                    trades += 1
        
        # Calculate final metrics
        final_price = prices[n_bars - 1]
        final_equity = cash + (btc * final_price)
        total_return_pct = ((final_equity - total_invested) / total_invested) * 100.0
        
        returns[i] = total_return_pct
        final_equities[i] = final_equity
        trade_counts[i] = trades
    
    return returns, final_equities, trade_counts


def simulate_portfolios_gpu(prices_gpu, signals_gpu, multipliers_gpu, param_batch, 
                            initial_cash=5000.0, monthly_contribution=1000.0, commission=0.001):
    """
    Simulate portfolios using Numba (compiled CPU code - much faster than Python loops).
    Converts GPU arrays to CPU for Numba processing.
    """
    n_bars = len(prices_gpu)
    n_params = signals_gpu.shape[1]
    bars_per_month = 720
    
    # Transfer from GPU to CPU for Numba processing
    prices_cpu = cp.asnumpy(prices_gpu).astype(np.float32)
    signals_cpu = cp.asnumpy(signals_gpu)
    multipliers_cpu = cp.asnumpy(multipliers_gpu).astype(np.float32)
    
    # Run Numba-compiled simulation (parallel on CPU cores)
    returns, final_equities, trade_counts = simulate_portfolios_numba(
        prices_cpu, signals_cpu, multipliers_cpu, n_params, n_bars,
        initial_cash, monthly_contribution, commission, bars_per_month
    )
    
    # Build results
    results = []
    for i in range(n_params):
        result = param_batch[i].copy()
        result.update({
            'total_return_pct': float(returns[i]),
            'final_equity': float(final_equities[i]),
            'total_trades': int(trade_counts[i]),
        })
        results.append(result)
    
    return results


def main():
    print("=" * 70)
    print("Pure GPU-Accelerated DCA Optimization (CuPy)")
    print("=" * 70)
    print()
    
    # Load data
    db_path = 'data/crypto.db'
    symbol = 'BTCUSDT'
    start_date = '2017-01-01'
    end_date = '2025-10-31'
    
    print(f"Loading {symbol} data...")
    prices_cpu = load_data(db_path, symbol, start_date, end_date)
    print(f"✓ Loaded {len(prices_cpu):,} candles")
    
    # Transfer to GPU
    print("\nTransferring data to GPU...")
    prices_gpu = cp.asarray(prices_cpu, dtype=cp.float32)
    print(f"✓ Data on GPU: {prices_gpu.nbytes / 1024**2:.1f} MB")
    
    # Parameter grid
    param_grid = {
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
    
    # Generate valid combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combinations = [dict(zip(keys, combo)) for combo in product(*values)]
    
    def is_valid(p):
        return (
            p['min_signal_score'] < p['extreme_signal_score'] and
            p['min_signal_score'] < p['strong_signal_threshold'] and
            p['strong_signal_threshold'] < p['extreme_signal_score'] and
            p['rsi_oversold_extreme'] < p['rsi_oversold_moderate'] and
            p['ema_distance_extreme'] < p['ema_distance_strong'] and
            p['ema_distance_strong'] < -8 and
            p['strong_signal_multiplier'] < p['extreme_signal_multiplier']
        )
    
    param_combinations = [p for p in all_combinations if is_valid(p)]
    print(f"\nTesting {len(param_combinations):,} valid parameter combinations")
    
    # Phase 1: Calculate indicators on GPU
    print(f"\n{'='*70}")
    print("PHASE 1: Indicator Calculation (GPU)")
    print(f"{'='*70}")
    
    start_time = datetime.now()
    rsi_periods = sorted(set(param_grid['rsi_period']))
    ema_periods = sorted(set(param_grid['ema_period']))
    indicators = calculate_indicators_gpu(prices_gpu, rsi_periods, ema_periods)
    indicator_time = (datetime.now() - start_time).total_seconds()
    print(f"  Time: {indicator_time:.2f}s")
    
    # Phase 2: Process in batches
    print(f"\n{'='*70}")
    print("PHASE 2: Signal Generation + Portfolio Simulation (GPU)")
    print("="*70)
    
    batch_size = 25000  # Process 25K params at a time (balanced for 24GB VRAM)
    n_batches = (len(param_combinations) + batch_size - 1) // batch_size
    
    print(f"\nProcessing {len(param_combinations):,} params in {n_batches} batches of {batch_size:,}")
    
    all_results = []
    start_time = datetime.now()
    
    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(param_combinations))
        param_batch = param_combinations[batch_start:batch_end]
        
        print(f"\n  Batch {batch_idx + 1}/{n_batches}: params {batch_start:,} to {batch_end:,}")
        
        # Generate signals on GPU
        print(f"    Generating signals on GPU...")
        signals_gpu, multipliers_gpu = generate_signals_gpu_batch(
            prices_gpu, indicators, param_batch, batch_start
        )
        
        # Simulate portfolios
        print(f"    Simulating {len(param_batch):,} portfolios...")
        batch_results = simulate_portfolios_gpu(
            prices_gpu, signals_gpu, multipliers_gpu, param_batch
        )
        
        all_results.extend(batch_results)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        rate = len(all_results) / elapsed
        print(f"    ✓ Complete: {len(all_results):,}/{len(param_combinations):,} ({rate:.1f} tests/sec)")
    
    total_time = (datetime.now() - start_time).total_seconds()
    
    # Results
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('total_return_pct', ascending=False)
    
    print(f"\n{'='*70}")
    print("Performance Summary")
    print(f"{'='*70}")
    print(f"  Indicator calculation: {indicator_time:.2f}s")
    print(f"  Optimization: {total_time:.2f}s")
    print(f"  Total: {indicator_time + total_time:.2f}s")
    print(f"  Throughput: {len(param_combinations)/(indicator_time + total_time):.1f} tests/sec")
    
    # Top 10
    print(f"\n{'='*70}")
    print("Top 10 Results")
    print(f"{'='*70}")
    
    for idx, (i, row) in enumerate(results_df.head(10).iterrows(), 1):
        print(f"\n#{idx}: Return: {row['total_return_pct']:.2f}%, Trades: {int(row['total_trades'])}")
        print(f"  RSI: {int(row['rsi_period'])}, EMA: {int(row['ema_period'])}")
        print(f"  Multipliers: {row['strong_signal_multiplier']:.1f}x/{row['extreme_signal_multiplier']:.1f}x")
    
    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'optimization_cupy_{timestamp}.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")
    
    # Cleanup GPU
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()
    print("\n✓ GPU memory cleaned up")


if __name__ == '__main__':
    main()
