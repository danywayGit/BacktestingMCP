"""
VectorBT-Based DCA Strategy Optimization
Uses VectorBT's vectorized backtesting for MASSIVE speedup.

VectorBT can run thousands of parameter combinations simultaneously
using pure NumPy/Numba vectorization - this is MUCH faster than
our custom GPU approach.
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
from tqdm import tqdm
import vectorbt as vbt
warnings.filterwarnings('ignore')


def load_data(db_path, symbol, start_date, end_date, timeframe='1h'):
    """Load OHLCV data from database."""
    conn = sqlite3.connect(db_path)
    
    start_ts = int(pd.Timestamp(start_date).timestamp())
    end_ts = int(pd.Timestamp(end_date).timestamp())
    
    query = """
        SELECT timestamp, open, high, low, close, volume
        FROM market_data
        WHERE symbol = ?
        AND timeframe = ?
        AND timestamp >= ?
        AND timestamp <= ?
        ORDER BY timestamp
    """
    df = pd.read_sql_query(query, conn, params=(symbol, timeframe, start_ts, end_ts))
    conn.close()
    
    df['Timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
    df = df[['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df.set_index('Timestamp', inplace=True)
    
    return df


def calculate_indicators_vectorbt(close_prices, rsi_periods, ema_periods):
    """
    Calculate indicators using VectorBT (super fast).
    Returns dict of indicators for all parameter combinations.
    """
    print(f"\nCalculating indicators with VectorBT...")
    print(f"  RSI periods: {rsi_periods}")
    print(f"  EMA periods: {ema_periods}")
    
    indicators = {}
    
    # Calculate all RSI values at once (vectorized)
    for period in rsi_periods:
        rsi = vbt.RSI.run(close_prices, window=period).rsi
        indicators[f'rsi_{period}'] = rsi
    
    # Calculate all EMA values at once (vectorized)
    for period in ema_periods:
        ema = close_prices.ewm(span=period, adjust=False).mean()
        indicators[f'ema_{period}'] = ema
    
    print(f"✓ Indicators calculated")
    
    return indicators


def create_signal_matrix_batch(df, indicators, param_combinations, batch_size=5000):
    """
    Create signal matrix in batches to avoid memory issues.
    Processes batch_size parameters at a time.
    
    Returns:
        - signals: DataFrame with buy signals (True/False) for each param combo
        - multipliers: DataFrame with signal multipliers for each param combo
    """
    n_bars = len(df)
    n_params = len(param_combinations)
    
    # Calculate memory requirements
    bytes_per_param = n_bars * 8  # float64
    mb_per_param = bytes_per_param / (1024**2)
    total_mb = mb_per_param * n_params
    
    print(f"\nCreating signal matrix: {n_params:,} params × {n_bars:,} bars")
    print(f"  Memory required: {total_mb/1024:.1f} GB total")
    print(f"  Processing in batches of {batch_size:,} params (~{batch_size * mb_per_param / 1024:.1f} GB per batch)")
    
    all_results = []
    
    # Process in batches
    for batch_start in range(0, n_params, batch_size):
        batch_end = min(batch_start + batch_size, n_params)
        batch_params = param_combinations[batch_start:batch_end]
        batch_n = len(batch_params)
        
        print(f"\n  Batch {batch_start//batch_size + 1}: Processing params {batch_start:,} to {batch_end:,}")
        
        # Initialize signal arrays for this batch
        batch_signals = np.zeros((n_bars, batch_n), dtype=bool)
        batch_multipliers = np.ones((n_bars, batch_n), dtype=float)
    
        # Process each parameter combination in this batch
        for idx, params in enumerate(tqdm(batch_params, desc=f"    Batch signals", leave=False)):
            rsi_period = params['rsi_period']
            ema_period = params['ema_period']
            
            # Get indicators for this combination
            rsi = indicators[f'rsi_{rsi_period}'].values
            ema = indicators[f'ema_{ema_period}'].values
            close = df['Close'].values
            
            # Calculate signal scores (vectorized)
            scores = np.zeros(n_bars)
            
            # EMA distance scoring
            ema_distance_pct = ((close - ema) / ema) * 100
            scores += np.where(ema_distance_pct <= params['ema_distance_extreme'], 3, 0)
            scores += np.where(
                (ema_distance_pct > params['ema_distance_extreme']) & 
                (ema_distance_pct <= params['ema_distance_strong']), 
                2, 0
            )
            scores += np.where(
                (ema_distance_pct > params['ema_distance_strong']) & 
                (ema_distance_pct <= -8), 
                1, 0
            )
            
            # RSI scoring
            scores += np.where(rsi < params['rsi_oversold_extreme'], 2, 0)
            scores += np.where(
                (rsi >= params['rsi_oversold_extreme']) & 
                (rsi < params['rsi_oversold_moderate']), 
                1, 0
            )
            
            # Generate signals (score >= min AND price < EMA)
            signals = (scores >= params['min_signal_score']) & (close < ema)
            
            # Calculate multipliers based on signal strength
            multipliers = np.ones(n_bars)
            multipliers = np.where(scores >= params['extreme_signal_score'], params['extreme_signal_multiplier'], multipliers)
            multipliers = np.where(
                (scores >= params['strong_signal_threshold']) & (scores < params['extreme_signal_score']),
                params['strong_signal_multiplier'],
                multipliers
            )
            
            # Store in batch arrays
            batch_signals[:, idx] = signals
            batch_multipliers[:, idx] = multipliers
        
        # Run VectorBT portfolio simulation for this batch
        batch_results = run_vectorbt_batch(df, batch_signals, batch_multipliers, batch_params, batch_start)
        all_results.extend(batch_results)
        
        print(f"    ✓ Batch complete: {len(batch_results):,} results")
    
    print(f"\n✓ All batches processed: {len(all_results):,} total results")
    return all_results


def run_vectorbt_batch(df, batch_signals, batch_multipliers, batch_params, batch_offset):
    """
    Run VectorBT portfolio simulation for a batch of parameters.
    """
    n_params = len(batch_params)
    base_amount = 1000.0
    
    # Convert to DataFrames with proper column names
    param_names = [f"param_{batch_offset + i}" for i in range(n_params)]
    signals_df = pd.DataFrame(batch_signals, index=df.index, columns=param_names)
    multipliers_df = pd.DataFrame(batch_multipliers, index=df.index, columns=param_names)
    
    # Calculate order sizes
    order_sizes = signals_df.astype(float) * multipliers_df * base_amount
    
    # Run VectorBT portfolio simulation
    pf = vbt.Portfolio.from_orders(
        close=df['Close'],
        size=order_sizes,
        size_type='value',
        fees=0.001,
        freq='1h',
        init_cash=5000.0,
        cash_sharing=False,
        call_seq='auto'
    )
    
    # Extract results
    total_returns = pf.total_return() * 100
    final_values = pf.final_value()
    total_trades = pf.trades.count()
    
    # Build results list
    results = []
    for idx, params in enumerate(batch_params):
        param_name = f"param_{batch_offset + idx}"
        result = params.copy()
        result.update({
            'total_return_pct': total_returns[param_name] if param_name in total_returns.index else 0,
            'final_equity': final_values[param_name] if param_name in final_values.index else 0,
            'total_trades': total_trades[param_name] if param_name in total_trades.index else 0,
        })
        results.append(result)
    
    return results





def main():
    print("=" * 70)
    print("VectorBT-Based DCA Strategy Optimization")
    print("=" * 70)
    print()
    
    # Load data
    db_path = 'data/crypto.db'
    symbol = 'BTCUSDT'
    start_date = '2017-01-01'
    end_date = '2025-10-31'
    
    print(f"Loading {symbol} data from {start_date} to {end_date}...")
    df = load_data(db_path, symbol, start_date, end_date)
    print(f"✓ Loaded {len(df):,} candles")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    print(f"  Price range: ${df['Close'].min():,.2f} - ${df['Close'].max():,.2f}")
    
    # Define parameter grid - FULL GRID with batch processing
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
    
    print(f"\n🚀 Using FULL parameter grid with batch processing")
    print(f"   Processing in ~20GB chunks to avoid memory issues")
    
    # Generate valid parameter combinations
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
    print(f"Total possible: {len(all_combinations):,}")
    
    # Extract unique indicator periods
    rsi_periods = sorted(set(param_grid['rsi_period']))
    ema_periods = sorted(set(param_grid['ema_period']))
    
    # Phase 1: Calculate indicators (super fast with VectorBT)
    start_time = datetime.now()
    indicators = calculate_indicators_vectorbt(df['Close'], rsi_periods, ema_periods)
    indicator_time = (datetime.now() - start_time).total_seconds()
    print(f"  Time: {indicator_time:.2f}s")
    
    # Phase 2: Create signal matrix and run optimization in batches
    print(f"\n{'='*70}")
    print("PHASE 2: Signal Generation + Portfolio Simulation (Batched)")
    print(f"{'='*70}")
    
    start_time = datetime.now()
    
    # Process in batches of 25000 parameters (~13.5GB per batch)
    results = create_signal_matrix_batch(df, indicators, param_combinations, batch_size=25000)
    
    total_time = (datetime.now() - start_time).total_seconds()
    print(f"\n  Total Phase 2 time: {total_time:.2f}s")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('total_return_pct', ascending=False)
    
    # Summary
    print(f"\n{'='*70}")
    print("Performance Summary")
    print(f"{'='*70}")
    print(f"  Indicator calculation: {indicator_time:.2f}s")
    print(f"  Signal generation + Portfolio simulation: {total_time:.2f}s")
    print(f"  Total time: {indicator_time + total_time:.2f}s")
    print(f"  Throughput: {len(param_combinations)/(indicator_time + total_time):.1f} tests/sec")
    
    # Display top 10
    print(f"\n{'='*70}")
    print("Top 10 Parameter Combinations")
    print(f"{'='*70}")
    
    for idx, (i, row) in enumerate(results_df.head(10).iterrows(), 1):
        print(f"\n#{idx}: Return: {row['total_return_pct']:.2f}%, Trades: {int(row['total_trades'])}")
        print(f"  RSI: {int(row['rsi_period'])}, EMA: {int(row['ema_period'])}")
        print(f"  Scores: min={int(row['min_signal_score'])}, strong={int(row['strong_signal_threshold'])}, extreme={int(row['extreme_signal_score'])}")
        print(f"  RSI Thresh: {row['rsi_oversold_extreme']:.0f}/{row['rsi_oversold_moderate']:.0f}")
        print(f"  EMA Dist: {row['ema_distance_extreme']:.0f}/{row['ema_distance_strong']:.0f}")
        print(f"  Multipliers: {row['strong_signal_multiplier']:.1f}x/{row['extreme_signal_multiplier']:.1f}x")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'optimization_vectorbt_{timestamp}.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")
    
    # Best parameters
    best = results_df.iloc[0]
    print(f"\n{'='*70}")
    print("Best Parameters")
    print(f"{'='*70}")
    print(f"""
rsi_period = {int(best['rsi_period'])}
ema_period = {int(best['ema_period'])}
min_signal_score = {int(best['min_signal_score'])}
strong_signal_threshold = {int(best['strong_signal_threshold'])}
extreme_signal_score = {int(best['extreme_signal_score'])}
rsi_oversold_extreme = {best['rsi_oversold_extreme']:.0f}
rsi_oversold_moderate = {best['rsi_oversold_moderate']:.0f}
ema_distance_extreme = {best['ema_distance_extreme']:.0f}
ema_distance_strong = {best['ema_distance_strong']:.0f}
strong_signal_multiplier = {best['strong_signal_multiplier']:.1f}
extreme_signal_multiplier = {best['extreme_signal_multiplier']:.1f}

Performance:
- Return: {best['total_return_pct']:.2f}%
- Trades: {int(best['total_trades'])}
- Final Value: ${best['final_equity']:,.2f}
""")


if __name__ == '__main__':
    main()
