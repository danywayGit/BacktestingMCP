"""
Improved Parallel Optimization using multiprocessing
Uses actual strategy code with CPU parallelization for accuracy + speed
"""

import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import pandas as pd
import sqlite3
from datetime import datetime
import warnings
from multiprocessing import Pool, cpu_count
import signal
warnings.filterwarnings('ignore')

# Import strategy
import importlib.util
spec = importlib.util.spec_from_file_location(
    "fractional_dca_comparison", 
    os.path.join(parent_dir, "scripts", "fractional_dca_comparison.py")
)
frac_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frac_module)

FractionalSignalDCAStrategy = frac_module.FractionalSignalDCAStrategy
calculate_analytics = frac_module.calculate_analytics

# Global data (loaded once, shared by workers)
global_df = None

def load_data():
    """Load data once"""
    conn = sqlite3.connect('data/crypto.db')
    start_ts = int(pd.Timestamp('2023-01-01').timestamp())
    end_ts = int(pd.Timestamp('2024-12-31').timestamp())
    
    query = """
        SELECT timestamp, open, high, low, close, volume
        FROM market_data
        WHERE symbol = 'BTCUSDT'
        AND timeframe = '1h'
        AND timestamp >= ?
        AND timestamp <= ?
        ORDER BY timestamp
    """
    df = pd.read_sql_query(query, conn, params=(start_ts, end_ts))
    conn.close()
    
    df['Timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
    df = df[['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df.set_index('Timestamp', inplace=True)
    
    return df

def run_single_backtest(params):
    """Worker function - runs one backtest"""
    try:
        # Create custom strategy
        class CustomSignalDCA(FractionalSignalDCAStrategy):
            def __init__(self, symbol, initial_cash, monthly_contribution, commission=0.001):
                super().__init__(symbol, initial_cash, monthly_contribution, commission)
                self.rsi_period = params['rsi_period']
                self.ema_period = params['ema_period']
                self.min_signal_score = params['min_signal_score']
                self.strong_signal_threshold = params['strong_signal_threshold']
                self.extreme_signal_score = params['extreme_signal_score']
                self.rsi_oversold_extreme = params['rsi_oversold_extreme']
                self.rsi_oversold_moderate = params['rsi_oversold_moderate']
                self.ema_distance_extreme = params['ema_distance_extreme']
                self.ema_distance_strong = params['ema_distance_strong']
                self.strong_signal_multiplier = params['strong_signal_multiplier']
                self.extreme_signal_multiplier = params['extreme_signal_multiplier']
        
        strategy = CustomSignalDCA(
            symbol='BTCUSDT',
            initial_cash=5000.0,
            monthly_contribution=1000.0,
            commission=0.001
        )
        
        result = strategy.run_backtest(global_df)
        analytics = calculate_analytics(result, global_df)
        
        result_dict = params.copy()
        result_dict.update({
            'total_return_pct': result['total_return'],
            'final_equity': result['final_equity'],
            'total_trades': result['total_trades'],
            'buy_trades': result['buy_trades'],
            'sell_trades': result['sell_trades'],
            'win_rate': analytics['win_rate'],
            'sharpe_ratio': analytics['sharpe_ratio'],
            'max_drawdown': analytics['max_drawdown'],
            'profit_factor': analytics['profit_factor']
        })
        
        return result_dict
    except Exception as e:
        print(f"Error with params {params}: {e}")
        return None

def main():
    global global_df
    
    print("=" * 70)
    print("Improved Parallel DCA Strategy Optimization")
    print("=" * 70)
    print()
    
    # Load data
    print("Loading data...")
    global_df = load_data()
    print(f"Loaded {len(global_df)} candles")
    print()
    
    # Define FOCUSED parameter grid (most impactful parameters)
    param_combinations = []
    
    # Core strategy parameters (highest impact)
    for min_score in [3, 4, 5, 6]:  # Entry aggressiveness
        for extreme_score in [9, 10, 11]:  # Reserve cash unlock
            for strong_threshold in [7, 8]:  # Position size trigger
                # RSI parameters
                for rsi_period in [14, 21]:
                    for rsi_extreme in [28, 30, 33, 35]:  # Oversold detection
                        for rsi_moderate in [38, 40, 43, 45]:
                            # EMA parameters
                            for ema_period in [150, 200, 250]:
                                for ema_extreme in [-22, -20, -18, -16]:  # Distance thresholds
                                    for ema_strong in [-16, -14, -12, -10]:
                                        # Position sizing
                                        for strong_mult in [1.1, 1.2, 1.3, 1.4]:
                                            for extreme_mult in [1.3, 1.4, 1.5, 1.6]:
                                                
                                                # Validation rules
                                                if min_score >= extreme_score:
                                                    continue
                                                if min_score >= strong_threshold:
                                                    continue
                                                if strong_threshold >= extreme_score:
                                                    continue
                                                if rsi_extreme >= rsi_moderate:
                                                    continue
                                                if ema_extreme >= ema_strong:
                                                    continue
                                                if ema_strong >= -8:
                                                    continue
                                                if strong_mult >= extreme_mult:
                                                    continue
                                                
                                                param_combinations.append({
                                                    'rsi_period': rsi_period,
                                                    'ema_period': ema_period,
                                                    'min_signal_score': min_score,
                                                    'strong_signal_threshold': strong_threshold,
                                                    'extreme_signal_score': extreme_score,
                                                    'rsi_oversold_extreme': float(rsi_extreme),
                                                    'rsi_oversold_moderate': float(rsi_moderate),
                                                    'ema_distance_extreme': float(ema_extreme),
                                                    'ema_distance_strong': float(ema_strong),
                                                    'strong_signal_multiplier': strong_mult,
                                                    'extreme_signal_multiplier': extreme_mult
                                                })
    
    total = len(param_combinations)
    print(f"Testing {total} parameter combinations")
    print(f"Using {cpu_count()} CPU cores for parallel processing")
    print()
    
    # Run optimization in parallel
    print("Running optimization...")
    with Pool(processes=cpu_count()) as pool:
        results = []
        for i, result in enumerate(pool.imap_unordered(run_single_backtest, param_combinations), 1):
            if result:
                results.append(result)
            if i % 100 == 0:
                print(f"Progress: {i}/{total} ({i*100//total}%)")
    
    print(f"Completed: {len(results)} successful backtests")
    print()
    
    # Convert to DataFrame and sort
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('total_return_pct', ascending=False)
    
    # Display top 10
    print("=" * 70)
    print("Top 10 Parameter Combinations")
    print("=" * 70)
    
    for idx, (i, row) in enumerate(results_df.head(10).iterrows(), 1):
        print(f"\n#{idx}: Return: {row['total_return_pct']:.2f}%, Trades: {int(row['total_trades'])}")
        print(f"  RSI: {int(row['rsi_period'])}, EMA: {int(row['ema_period'])}")
        print(f"  Scores: min={int(row['min_signal_score'])}, strong={int(row['strong_signal_threshold'])}, extreme={int(row['extreme_signal_score'])}")
        print(f"  RSI Thresh: {row['rsi_oversold_extreme']:.0f}/{row['rsi_oversold_moderate']:.0f}")
        print(f"  EMA Dist: {row['ema_distance_extreme']:.0f}/{row['ema_distance_strong']:.0f}")
        print(f"  Multipliers: {row['strong_signal_multiplier']:.1f}x/{row['extreme_signal_multiplier']:.1f}x")
        print(f"  Metrics: Sharpe={row['sharpe_ratio']:.2f}, MaxDD={row['max_drawdown']:.2f}%, PF={row['profit_factor']:.2f}")
    
    # Analyze parameter impact
    print("\n" + "=" * 70)
    print("📊 Parameter Impact Analysis")
    print("=" * 70)
    
    key_params = ['min_signal_score', 'rsi_period', 'ema_period', 'extreme_signal_score']
    for param in key_params:
        print(f"\n{param}:")
        grouped = results_df.groupby(param).agg({
            'total_return_pct': ['mean', 'max', 'count'],
            'total_trades': 'mean'
        }).round(2)
        print(grouped)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'optimization_parallel_{timestamp}.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Best parameters
    print("\n" + "=" * 70)
    print("Recommended Parameters")
    print("=" * 70)
    best = results_df.iloc[0]
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
- Win Rate: {best['win_rate']:.1f}%
- Sharpe Ratio: {best['sharpe_ratio']:.2f}
- Max Drawdown: {best['max_drawdown']:.2f}%
- Profit Factor: {best['profit_factor']:.2f}
""")

if __name__ == '__main__':
    # Needed for Windows multiprocessing
    from multiprocessing import freeze_support
    freeze_support()
    main()
