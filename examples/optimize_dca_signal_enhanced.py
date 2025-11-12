"""
Enhanced Signal-Based DCA Strategy Parameter Optimization
Optimizes all configurable parameters including indicators and thresholds.
"""

import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
import warnings
import signal
from itertools import product
warnings.filterwarnings('ignore')

# Import directly
scripts_dir = os.path.join(parent_dir, 'scripts')
sys.path.insert(0, scripts_dir)

import importlib.util
spec = importlib.util.spec_from_file_location(
    "fractional_dca_comparison", 
    os.path.join(scripts_dir, "fractional_dca_comparison.py")
)
frac_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frac_module)

FractionalSignalDCAStrategy = frac_module.FractionalSignalDCAStrategy
calculate_analytics = frac_module.calculate_analytics

# Global flag for Ctrl+C handling
interrupted = False

def signal_handler(sig, frame):
    global interrupted
    interrupted = True
    print("\n\n⚠️  Optimization interrupted by user. Finishing current test...")

signal.signal(signal.SIGINT, signal_handler)

def load_data(db_path, symbol, start_date, end_date):
    """Load OHLCV data from database."""
    conn = sqlite3.connect(db_path)
    
    # Convert dates to timestamps
    start_ts = int(pd.Timestamp(start_date).timestamp())
    end_ts = int(pd.Timestamp(end_date).timestamp())
    
    query = """
        SELECT timestamp, open, high, low, close, volume
        FROM market_data
        WHERE symbol = ?
        AND timeframe = '1h'
        AND timestamp >= ?
        AND timestamp <= ?
        ORDER BY timestamp
    """
    df = pd.read_sql_query(query, conn, params=(symbol, start_ts, end_ts))
    conn.close()
    
    # Convert timestamp to datetime
    df['Timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
    df = df[['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df.set_index('Timestamp', inplace=True)
    
    return df

def main():
    global interrupted
    
    print("=" * 70)
    print("Enhanced Signal-Based DCA Strategy Optimization")
    print("=" * 70)
    print()
    
    # Load data
    db_path = 'data/crypto.db'
    symbol = 'BTCUSDT'
    start_date = '2023-01-01'
    end_date = '2024-12-31'
    
    print(f"Loading {symbol} data from {start_date} to {end_date}...")
    df = load_data(db_path, symbol, start_date, end_date)
    print(f"✓ Loaded {len(df)} candles")
    print()
    
    # Define parameter grid - ALL OPTIMIZABLE PARAMETERS
    param_grid = {
        # Indicator periods
        'rsi_period': [14, 21],
        'ema_period': [150, 200],
        
        # Signal score thresholds
        'min_signal_score': [4, 5, 6],
        'extreme_signal_score': [9, 10],
        'strong_signal_threshold': [7, 8],
        
        # RSI thresholds (oversold detection)
        'rsi_oversold_extreme': [30, 35],  # 2 points
        'rsi_oversold_moderate': [40, 45],  # 1 point
        
        # EMA distance thresholds (as negative percentages)
        'ema_distance_extreme': [-20, -18],  # 3 points
        'ema_distance_strong': [-14, -12],   # 2 points
        
        # Position sizing multipliers
        'strong_signal_multiplier': [1.2, 1.3],
        'extreme_signal_multiplier': [1.4, 1.5],
    }
    
    # Generate all valid combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combinations = list(product(*values))
    
    # Filter valid combinations
    valid_combinations = []
    for combo in all_combinations:
        params = dict(zip(keys, combo))
        
        # Validation rules
        if params['min_signal_score'] >= params['extreme_signal_score']:
            continue
        if params['min_signal_score'] >= params['strong_signal_threshold']:
            continue
        if params['strong_signal_threshold'] >= params['extreme_signal_score']:
            continue
        if params['rsi_oversold_extreme'] >= params['rsi_oversold_moderate']:
            continue
        if params['ema_distance_extreme'] >= params['ema_distance_strong']:
            continue
        if params['ema_distance_strong'] >= -8:  # Must be below moderate threshold
            continue
        if params['strong_signal_multiplier'] >= params['extreme_signal_multiplier']:
            continue
        
        valid_combinations.append(params)
    
    total = len(valid_combinations)
    print(f"Testing {total} valid parameter combinations")
    print(f"Parameters: {', '.join(keys)}")
    print()
    print("⚠️  Note: Cash allocation (70% active / 30% reserve) is FIXED by design")
    print("Press Ctrl+C to stop optimization early")
    print("-" * 70)
    print()
    
    # Run optimization
    results = []
    
    for i, params in enumerate(valid_combinations, 1):
        if interrupted:
            print("\n⚠️  Optimization stopped by user")
            break
        
        print(f"[{i}/{total}] Testing: rsi={params['rsi_period']}, ema={params['ema_period']}, "
              f"min={params['min_signal_score']}, extreme={params['extreme_signal_score']}")
        
        try:
            # Create custom strategy with these parameters
            class CustomSignalDCA(FractionalSignalDCAStrategy):
                def __init__(self, symbol, initial_cash, monthly_contribution, commission=0.001):
                    super().__init__(symbol, initial_cash, monthly_contribution, commission)
                    # Override all optimizable parameters
                    self.rsi_period = params['rsi_period']
                    self.ema_period = params['ema_period']
                    self.min_signal_score = params['min_signal_score']
                    self.extreme_signal_score = params['extreme_signal_score']
                    self.strong_signal_threshold = params['strong_signal_threshold']
                    self.rsi_oversold_extreme = params['rsi_oversold_extreme']
                    self.rsi_oversold_moderate = params['rsi_oversold_moderate']
                    self.ema_distance_extreme = params['ema_distance_extreme']
                    self.ema_distance_strong = params['ema_distance_strong']
                    self.strong_signal_multiplier = params['strong_signal_multiplier']
                    self.extreme_signal_multiplier = params['extreme_signal_multiplier']
            
            # Run backtest
            strategy = CustomSignalDCA(
                symbol='BTCUSDT',
                initial_cash=5000.0,
                monthly_contribution=1000.0,
                commission=0.001
            )
            
            result = strategy.run_backtest(df)
            analytics = calculate_analytics(result, df)
            
            # Store results
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
            results.append(result_dict)
            
            print(f"  → Return: {result['total_return']:.2f}%, Trades: {result['total_trades']}, "
                  f"Win Rate: {analytics['win_rate']:.1f}%")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
    
    print()
    print("=" * 70)
    print(f"Optimization Complete - Tested {len(results)}/{total} combinations")
    print("=" * 70)
    print()
    
    if not results:
        print("❌ No valid results to display")
        return
    
    # Convert to DataFrame and sort by return
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('total_return_pct', ascending=False)
    
    # Display top 10 results
    print("🏆 Top 10 Parameter Combinations:")
    print("-" * 70)
    
    for idx, (i, row) in enumerate(results_df.head(10).iterrows(), 1):
        print(f"\n#{idx}:")
        print(f"  Indicator Params:")
        print(f"    RSI Period: {int(row['rsi_period'])}")
        print(f"    EMA Period: {int(row['ema_period'])}")
        print(f"  Score Thresholds:")
        print(f"    Min Signal: {int(row['min_signal_score'])}")
        print(f"    Strong Signal: {int(row['strong_signal_threshold'])}")
        print(f"    Extreme Signal: {int(row['extreme_signal_score'])}")
        print(f"  RSI Thresholds:")
        print(f"    Extreme: {row['rsi_oversold_extreme']:.0f}, Moderate: {row['rsi_oversold_moderate']:.0f}")
        print(f"  EMA Distance Thresholds:")
        print(f"    Extreme: {row['ema_distance_extreme']:.0f}%, Strong: {row['ema_distance_strong']:.0f}%")
        print(f"  Multipliers:")
        print(f"    Strong: {row['strong_signal_multiplier']:.1f}x, Extreme: {row['extreme_signal_multiplier']:.1f}x")
        print(f"  Performance:")
        print(f"    Return: {row['total_return_pct']:.2f}%")
        print(f"    Final Equity: ${row['final_equity']:,.2f}")
        print(f"    Trades: {int(row['total_trades'])} (Buy: {int(row['buy_trades'])}, Sell: {int(row['sell_trades'])})")
        print(f"    Win Rate: {row['win_rate']:.1f}%")
        print(f"    Sharpe: {row['sharpe_ratio']:.2f}, Max DD: {row['max_drawdown']:.2f}%")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'optimization_enhanced_{timestamp}.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Full results saved to: {output_file}")
    
    # Best parameters summary
    print("\n" + "=" * 70)
    print("🎯 Recommended Parameters (Best Return)")
    print("=" * 70)
    best = results_df.iloc[0]
    print(f"""
# Indicator Parameters
rsi_period = {int(best['rsi_period'])}
ema_period = {int(best['ema_period'])}

# Signal Score Thresholds
min_signal_score = {int(best['min_signal_score'])}
strong_signal_threshold = {int(best['strong_signal_threshold'])}
extreme_signal_score = {int(best['extreme_signal_score'])}

# RSI Thresholds
rsi_oversold_extreme = {best['rsi_oversold_extreme']:.0f}
rsi_oversold_moderate = {best['rsi_oversold_moderate']:.0f}

# EMA Distance Thresholds
ema_distance_extreme = {best['ema_distance_extreme']:.0f}
ema_distance_strong = {best['ema_distance_strong']:.0f}

# Position Sizing Multipliers
strong_signal_multiplier = {best['strong_signal_multiplier']:.1f}
extreme_signal_multiplier = {best['extreme_signal_multiplier']:.1f}

Expected Performance:
- Return: {best['total_return_pct']:.2f}%
- Win Rate: {best['win_rate']:.1f}%
- Sharpe Ratio: {best['sharpe_ratio']:.2f}
- Max Drawdown: {best['max_drawdown']:.2f}%
- Total Trades: {int(best['total_trades'])}
""")

if __name__ == '__main__':
    main()
