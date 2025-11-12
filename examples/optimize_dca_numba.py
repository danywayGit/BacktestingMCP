"""
GPU-Accelerated Signal-Based DCA Strategy Optimization using Numba CUDA
Leverages RTX 4090 for massive parallel optimization speedup.
"""

import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import numpy as np
import pandas as pd
from numba import jit, prange, cuda
import sqlite3
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("GPU-Accelerated DCA Optimization (Numba + RTX 4090)")
print("=" * 70)
print()

# Check CUDA availability
try:
    if cuda.is_available():
        print(f"✓ CUDA Available: {cuda.get_current_device().name.decode()}")
        print(f"✓ Compute Capability: {cuda.get_current_device().compute_capability}")
        use_cuda = True
    else:
        print("⚠️  CUDA not available, using CPU parallelization")
        use_cuda = False
except:
    print("⚠️  CUDA not available, using CPU parallelization")
    use_cuda = False

print()

# Load data
def load_data():
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
    df = df.rename(columns={'close': 'Close'})
    return df['Close'].values

print("Loading data...")
prices = load_data()
print(f"✓ Loaded {len(prices)} candles")
print()

@jit(nopython=True, parallel=True)
def calculate_ema_fast(prices, period):
    """Fast EMA calculation"""
    ema = np.empty(len(prices))
    alpha = 2.0 / (period + 1)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
    return ema

@jit(nopython=True, parallel=True)
def calculate_rsi_fast(prices, period):
    """Fast RSI calculation"""
    rsi = np.full(len(prices), 50.0)
    
    for i in prange(period, len(prices)):
        gains = 0.0
        losses = 0.0
        
        for j in range(i - period, i):
            change = prices[j+1] - prices[j]
            if change > 0:
                gains += change
            else:
                losses += abs(change)
        
        avg_gain = gains / period
        avg_loss = losses / period
        
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

@jit(nopython=True, parallel=True)
def backtest_dca_strategy(
    prices,
    ema_values,
    rsi_values,
    min_score,
    extreme_score,
    strong_threshold,
    rsi_extreme,
    rsi_moderate,
    ema_dist_extreme,
    ema_dist_strong,
    ema_dist_moderate,
    strong_mult,
    extreme_mult
):
    """
    Fast backtesting of DCA strategy.
    Returns: (total_return, total_trades, final_equity, coins_held)
    """
    initial_cash = 5000.0
    monthly_contribution = 1000.0
    cash = initial_cash
    cash_active = 0.0
    cash_reserve = 0.0
    coins = 0.0
    total_trades = 0
    current_month = -1
    last_buy_price = 0.0
    
    # Simple month tracking (assume ~730 hours per month)
    hours_per_month = 730
    
    for i in range(len(prices)):
        # Monthly contribution (approximate)
        month = i // hours_per_month
        if month != current_month:
            current_month = month
            cash += monthly_contribution
            cash_active += monthly_contribution * 0.7
            cash_reserve += monthly_contribution * 0.3
        
        # Calculate signal score
        score = 0
        
        # EMA distance scoring
        if ema_values[i] > 0:
            dist_pct = ((prices[i] - ema_values[i]) / ema_values[i]) * 100.0
            if dist_pct <= ema_dist_extreme:
                score += 3
            elif dist_pct <= ema_dist_strong:
                score += 2
            elif dist_pct <= ema_dist_moderate:
                score += 1
        
        # RSI scoring
        if rsi_values[i] < rsi_extreme:
            score += 2
        elif rsi_values[i] < rsi_moderate:
            score += 1
        
        # Check buy conditions
        if score >= min_score and prices[i] < ema_values[i]:
            # Check for 2 red days or 5% drop
            buy_signal = False
            if i >= 2:
                if prices[i] < prices[i-1] and prices[i-1] < prices[i-2]:
                    buy_signal = True
            if i >= 1:
                drop = ((prices[i] - prices[i-1]) / prices[i-1]) * 100.0
                if drop <= -5.0 and prices[i] < prices[i-1]:
                    buy_signal = True
            
            if buy_signal:
                # Determine available cash
                is_extreme = score >= extreme_score
                available = (cash_active + cash_reserve) if is_extreme else cash_active
                
                if available >= 50.0:
                    # Calculate position size
                    base_size = available / 20.0
                    mult = 1.0
                    
                    # Price drop bonus
                    if last_buy_price > 0:
                        drop_pct = ((prices[i] - last_buy_price) / last_buy_price) * 100.0
                        if drop_pct <= -20.0:
                            mult += 0.25
                        elif drop_pct <= -10.0:
                            mult += 0.15
                    
                    # Signal strength multiplier
                    if score >= extreme_score:
                        mult *= extreme_mult
                    elif score >= strong_threshold:
                        mult *= strong_mult
                    
                    buy_amount = min(base_size * mult, available)
                    
                    # Execute buy
                    coins_bought = buy_amount / prices[i]
                    coins += coins_bought
                    
                    # Deduct from cash pools
                    if is_extreme:
                        if buy_amount <= cash_active:
                            cash_active -= buy_amount
                        else:
                            remaining = buy_amount - cash_active
                            cash_active = 0.0
                            cash_reserve -= remaining
                    else:
                        cash_active -= buy_amount
                    
                    cash -= buy_amount
                    total_trades += 1
                    last_buy_price = prices[i]
    
    # Final equity
    final_position_value = coins * prices[-1]
    final_equity = cash + final_position_value
    total_return = ((final_equity - initial_cash) / initial_cash) * 100.0
    
    return total_return, total_trades, final_equity, coins

# Pre-calculate indicators for different periods
print("Pre-calculating indicators...")
ema_150 = calculate_ema_fast(prices, 150)
ema_200 = calculate_ema_fast(prices, 200)
rsi_14 = calculate_rsi_fast(prices, 14)
rsi_21 = calculate_rsi_fast(prices, 21)
print("✓ Indicators calculated")
print()

# Define parameter grid
param_combinations = []

for rsi_period in [14, 21]:
    for ema_period in [150, 200]:
        for min_score in [4, 5, 6]:
            for extreme_score in [9, 10]:
                for strong_threshold in [7, 8]:
                    for rsi_extreme in [30, 35]:
                        for rsi_moderate in [40, 45]:
                            for ema_extreme in [-20, -18]:
                                for ema_strong in [-14, -12]:
                                    for strong_mult in [1.2, 1.3]:
                                        for extreme_mult in [1.4, 1.5]:
                                            # Validation
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
                                            
                                            param_combinations.append((
                                                rsi_period, ema_period, min_score, extreme_score,
                                                strong_threshold, rsi_extreme, rsi_moderate,
                                                ema_extreme, ema_strong, strong_mult, extreme_mult
                                            ))

print(f"Testing {len(param_combinations)} parameter combinations...")
print("Running parallel optimization with Numba JIT...")
print()

# Run all combinations (parallelized by Numba)
results = []
for i, params in enumerate(param_combinations):
    if i % 100 == 0:
        print(f"Progress: {i}/{len(param_combinations)} ({i*100//len(param_combinations)}%)")
    
    rsi_per, ema_per, min_s, ext_s, strong_t, rsi_e, rsi_m, ema_e, ema_s, s_mult, e_mult = params
    
    # Select pre-calculated indicators
    rsi_vals = rsi_14 if rsi_per == 14 else rsi_21
    ema_vals = ema_150 if ema_per == 150 else ema_200
    
    # Run backtest
    ret, trades, equity, coins = backtest_dca_strategy(
        prices, ema_vals, rsi_vals,
        min_s, ext_s, strong_t,
        rsi_e, rsi_m,
        ema_e, ema_s, -8.0,  # moderate threshold fixed
        s_mult, e_mult
    )
    
    results.append({
        'rsi_period': rsi_per,
        'ema_period': ema_per,
        'min_signal_score': min_s,
        'extreme_signal_score': ext_s,
        'strong_signal_threshold': strong_t,
        'rsi_oversold_extreme': rsi_e,
        'rsi_oversold_moderate': rsi_m,
        'ema_distance_extreme': ema_e,
        'ema_distance_strong': ema_s,
        'strong_signal_multiplier': s_mult,
        'extreme_signal_multiplier': e_mult,
        'total_return_pct': ret,
        'total_trades': trades,
        'final_equity': equity,
        'coins_held': coins
    })

print(f"✓ Completed: {len(results)} combinations tested")
print()

# Convert to DataFrame and sort
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('total_return_pct', ascending=False)

# Display top 10
print("=" * 70)
print("🏆 Top 10 Parameter Combinations (by Return)")
print("=" * 70)

for idx, (i, row) in enumerate(results_df.head(10).iterrows(), 1):
    print(f"\n#{idx}: Return: {row['total_return_pct']:.2f}%, Trades: {int(row['total_trades'])}")
    print(f"  RSI: {int(row['rsi_period'])}, EMA: {int(row['ema_period'])}")
    print(f"  Scores: min={int(row['min_signal_score'])}, strong={int(row['strong_signal_threshold'])}, extreme={int(row['extreme_signal_score'])}")
    print(f"  RSI Thresh: {row['rsi_oversold_extreme']:.0f}/{row['rsi_oversold_moderate']:.0f}")
    print(f"  EMA Dist: {row['ema_distance_extreme']:.0f}/{row['ema_distance_strong']:.0f}")
    print(f"  Multipliers: {row['strong_signal_multiplier']:.1f}x/{row['extreme_signal_multiplier']:.1f}x")
    print(f"  Final: ${row['final_equity']:,.2f}, Coins: {row['coins_held']:.6f}")

# Save results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = f'optimization_numba_{timestamp}.csv'
results_df.to_csv(output_file, index=False)
print(f"\n✓ Results saved to: {output_file}")

# Best parameters
print("\n" + "=" * 70)
print("🎯 Recommended Parameters")
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
- Final Equity: ${best['final_equity']:,.2f}
- Total Trades: {int(best['total_trades'])}
- Coins Held: {best['coins_held']:.6f}
""")
