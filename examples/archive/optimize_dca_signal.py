"""
Signal-Based DCA Strategy Parameter Optimization
Optimizes the key parameters of the Signal-Based DCA strategy.
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
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Rename columns
    df.columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    df.set_index('Timestamp', inplace=True)
    
    return df

def main():
    global interrupted
    
    print("=" * 70)
    print("Signal-Based DCA Strategy Parameter Optimization")
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
    
    # Define parameter grid - focusing on the most impactful parameters
    param_grid = {
        'min_signal_score': [4, 5, 6],  # Lower = more aggressive
        'extreme_signal_score': [9, 10, 11],  # When to use reserve cash
        'strong_signal_multiplier': [1.2, 1.3, 1.5],  # Size bonus for strong signals
        'extreme_signal_multiplier': [1.4, 1.5, 1.6],  # Size bonus for extreme signals
    }
    
    # Generate all valid combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combinations = list(product(*values))
    
    # Filter: min_signal_score < extreme_signal_score
    valid_combinations = [
        combo for combo in all_combinations
        if combo[0] < combo[1]  # min < extreme
    ]
    
    total = len(valid_combinations)
    print(f"Testing {total} parameter combinations")
    print(f"Parameters: {', '.join(keys)}")
    print()
    print("Press Ctrl+C to stop optimization early")
    print("-" * 70)
    print()
    
    # Run optimization
    results = []
    
    for i, combo in enumerate(valid_combinations, 1):
        if interrupted:
            print("\n⚠️  Optimization stopped by user")
            break
        
        min_sig, extreme_sig, strong_mult, extreme_mult = combo
        
        print(f"[{i}/{total}] Testing: min={min_sig}, extreme={extreme_sig}, "
              f"strong_mult={strong_mult:.1f}, extreme_mult={extreme_mult:.1f}")
        
        try:
            # Create custom strategy
            class CustomSignalDCA(FractionalSignalDCAStrategy):
                def __init__(self, symbol, initial_cash, monthly_contribution, commission=0.001):
                    super().__init__(symbol, initial_cash, monthly_contribution, commission)
                    self.min_signal_score = min_sig
                    self.extreme_signal_score = extreme_sig
                    self.strong_signal_multiplier = strong_mult
                    self.extreme_signal_multiplier = extreme_mult
            
            # Run backtest
            strategy = CustomSignalDCA(
                symbol='BTCUSDT',
                initial_cash=5000.0,
                monthly_contribution=1000.0,
                commission=0.001
            )
            
            result = strategy.run_backtest(df)
            
            # Calculate analytics
            analytics = calculate_analytics(result, df)
            
            # Store results
            results.append({
                'min_signal_score': min_sig,
                'extreme_signal_score': extreme_sig,
                'strong_signal_multiplier': strong_mult,
                'extreme_signal_multiplier': extreme_mult,
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
    
    for i, row in results_df.head(10).iterrows():
        print(f"\n#{results_df.index.get_loc(i)+1}:")
        print(f"  Parameters:")
        print(f"    min_signal_score: {row['min_signal_score']}")
        print(f"    extreme_signal_score: {row['extreme_signal_score']}")
        print(f"    strong_signal_multiplier: {row['strong_signal_multiplier']:.1f}")
        print(f"    extreme_signal_multiplier: {row['extreme_signal_multiplier']:.1f}")
        print(f"  Performance:")
        print(f"    Return: {row['total_return_pct']:.2f}%")
        print(f"    Final Equity: ${row['final_equity']:,.2f}")
        print(f"    Total Trades: {row['total_trades']}")
        print(f"    Buy/Sell: {row['buy_trades']}/{row['sell_trades']}")
        print(f"    Win Rate: {row['win_rate']:.1f}%")
        print(f"    Sharpe Ratio: {row['sharpe_ratio']:.2f}")
        print(f"    Max Drawdown: {row['max_drawdown']:.2f}%")
        print(f"    Profit Factor: {row['profit_factor']:.2f}")
    
    # Parameter impact analysis
    print("\n" + "=" * 70)
    print("📊 Parameter Impact Analysis")
    print("=" * 70)
    
    for param in ['min_signal_score', 'extreme_signal_score', 'strong_signal_multiplier', 'extreme_signal_multiplier']:
        print(f"\n{param}:")
        grouped = results_df.groupby(param).agg({
            'total_return_pct': ['mean', 'std', 'max'],
            'win_rate': 'mean',
            'total_trades': 'mean'
        }).round(2)
        print(grouped)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'optimization_results_{timestamp}.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Full results saved to: {output_file}")
    
    # Best parameters summary
    print("\n" + "=" * 70)
    print("🎯 Recommended Parameters (Best Return)")
    print("=" * 70)
    best = results_df.iloc[0]
    print(f"""
min_signal_score = {int(best['min_signal_score'])}
extreme_signal_score = {int(best['extreme_signal_score'])}
strong_signal_multiplier = {best['strong_signal_multiplier']:.1f}
extreme_signal_multiplier = {best['extreme_signal_multiplier']:.1f}

Expected Performance:
- Return: {best['total_return_pct']:.2f}%
- Win Rate: {best['win_rate']:.1f}%
- Sharpe Ratio: {best['sharpe_ratio']:.2f}
- Max Drawdown: {best['max_drawdown']:.2f}%
""")

if __name__ == '__main__':
    main()
