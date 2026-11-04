"""
GPU-Accelerated Optimization for EMA Crossover RSI Strategy.

Optimizes parameters for BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT on 4H timeframe.

Parameters to optimize:
- ema_fast: Fast EMA period
- ema_slow: Slow EMA period
- rsi_period: RSI calculation period
- rsi_threshold: Entry threshold
- atr_multiplier: Stop loss multiplier
- reward_risk_ratio: R:R ratio (minimum 2:1)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime
from itertools import product
from typing import Dict, List, Tuple, Any

# GPU/Numba imports
try:
    import cupy as cp
    GPU_AVAILABLE = True
    device = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(device.id)
    gpu_name = props['name'].decode('utf-8')
    print(f"✓ GPU Acceleration: {gpu_name}")
except ImportError:
    GPU_AVAILABLE = False
    cp = np
    print("⚠️  CuPy not available - using CPU")

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
    print("✓ Numba JIT compilation available")
except ImportError:
    NUMBA_AVAILABLE = False
    print("⚠️  Numba not available")


# =============================================================================
# GPU-ACCELERATED INDICATOR CALCULATIONS
# =============================================================================

def calculate_ema_gpu(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculate EMA on GPU."""
    if GPU_AVAILABLE:
        prices_gpu = cp.asarray(prices, dtype=cp.float32)
        ema_gpu = cp.empty_like(prices_gpu)
        alpha = 2.0 / (period + 1)
        
        ema_gpu[0] = prices_gpu[0]
        for i in range(1, len(prices_gpu)):
            ema_gpu[i] = alpha * prices_gpu[i] + (1 - alpha) * ema_gpu[i-1]
        
        return cp.asnumpy(ema_gpu)
    else:
        ema = np.empty(len(prices), dtype=np.float32)
        alpha = 2.0 / (period + 1)
        ema[0] = prices[0]
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        return ema


def calculate_rsi_cpu(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculate RSI (pandas is fastest for this)."""
    prices_series = pd.Series(prices)
    delta = prices_series.diff()
    
    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)
    
    avg_gain = gains.ewm(span=period, adjust=False).mean()
    avg_loss = losses.ewm(span=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.fillna(50.0).values.astype(np.float32)


def calculate_atr_cpu(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Calculate ATR."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    prev_close = close_s.shift(1)
    
    tr1 = high_s - low_s
    tr2 = (high_s - prev_close).abs()
    tr3 = (low_s - prev_close).abs()
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(span=period, adjust=False).mean()
    
    return atr.fillna(0).values.astype(np.float32)


def batch_calculate_indicators(
    prices: np.ndarray, 
    high: np.ndarray, 
    low: np.ndarray,
    ema_fast_periods: List[int],
    ema_slow_periods: List[int],
    ema_12h_periods: List[int],
    rsi_periods: List[int],
    atr_periods: List[int]
) -> Dict[str, Dict[int, np.ndarray]]:
    """Pre-calculate all required indicators."""
    
    indicators = {
        'ema_fast': {},
        'ema_slow': {},
        'ema_12h': {},
        'rsi': {},
        'atr': {}
    }
    
    # Fast EMAs
    for period in ema_fast_periods:
        indicators['ema_fast'][period] = calculate_ema_gpu(prices, period)
    
    # Slow EMAs
    for period in ema_slow_periods:
        indicators['ema_slow'][period] = calculate_ema_gpu(prices, period)
    
    # 12H EMAs (3x period on 4H data)
    for period in ema_12h_periods:
        indicators['ema_12h'][period] = calculate_ema_gpu(prices, period * 3)
    
    # RSI
    for period in rsi_periods:
        indicators['rsi'][period] = calculate_rsi_cpu(prices, period)
    
    # ATR
    for period in atr_periods:
        indicators['atr'][period] = calculate_atr_cpu(high, low, prices, period)
    
    return indicators


# =============================================================================
# NUMBA-ACCELERATED PORTFOLIO SIMULATION WITH DAY-OF-WEEK FILTER
# =============================================================================

@njit(parallel=True)
def simulate_strategy_batch_with_days(
    prices: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    day_of_week: np.ndarray,  # 0=Monday, 6=Sunday
    ema_fast_all: np.ndarray,
    ema_slow_all: np.ndarray,
    ema_12h_all: np.ndarray,
    rsi_all: np.ndarray,
    atr_all: np.ndarray,
    param_indices: np.ndarray,
    params_values: np.ndarray,
    day_masks: np.ndarray,  # Shape (n_day_combos, 7) - boolean for each day
    initial_cash: float,
    risk_per_trade: float,
    n_bars: int,
    n_base_params: int,
    n_day_combos: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
           np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray]:
    """
    Simulate EMA crossover strategy with day-of-week filtering.
    
    Returns:
        total_returns, final_equities, trade_counts, sqn_values,
        max_drawdowns, win_rates, avg_win, avg_loss,
        avg_bars_win, avg_bars_tp, avg_bars_sl, win_counts, day_combo_indices
    """
    
    n_total = n_base_params * n_day_combos
    
    total_returns = np.zeros(n_total, dtype=np.float32)
    final_equities = np.zeros(n_total, dtype=np.float32)
    trade_counts = np.zeros(n_total, dtype=np.int32)
    sqn_values = np.zeros(n_total, dtype=np.float32)
    max_drawdowns = np.zeros(n_total, dtype=np.float32)
    win_rates = np.zeros(n_total, dtype=np.float32)
    avg_wins = np.zeros(n_total, dtype=np.float32)
    avg_losses = np.zeros(n_total, dtype=np.float32)
    avg_bars_wins = np.zeros(n_total, dtype=np.float32)
    avg_bars_tps = np.zeros(n_total, dtype=np.float32)
    avg_bars_sls = np.zeros(n_total, dtype=np.float32)
    win_counts = np.zeros(n_total, dtype=np.int32)
    day_combo_indices = np.zeros(n_total, dtype=np.int32)
    
    for idx in prange(n_total):
        p = idx // n_day_combos  # Base param index
        d = idx % n_day_combos   # Day combo index
        
        day_combo_indices[idx] = d
        
        ema_fast_idx = param_indices[p, 0]
        ema_slow_idx = param_indices[p, 1]
        ema_12h_idx = param_indices[p, 2]
        rsi_idx = param_indices[p, 3]
        atr_idx = param_indices[p, 4]
        
        rsi_threshold = params_values[p, 0]
        atr_multiplier = params_values[p, 1]
        reward_risk_ratio = params_values[p, 2]
        
        equity = initial_cash
        peak_equity = initial_cash
        max_dd = 0.0
        position = 0.0
        entry_price = 0.0
        entry_bar = 0
        stop_loss = 0.0
        take_profit = 0.0
        
        trades = 0
        wins = 0
        losses_count = 0
        
        trade_returns = np.zeros(500, dtype=np.float32)
        trade_idx = 0
        
        win_returns = np.zeros(250, dtype=np.float32)
        loss_returns = np.zeros(250, dtype=np.float32)
        win_idx = 0
        loss_idx = 0
        
        bars_in_wins = np.zeros(250, dtype=np.int32)
        bars_in_tp = np.zeros(250, dtype=np.int32)
        bars_in_sl = np.zeros(250, dtype=np.int32)
        win_bars_idx = 0
        tp_bars_idx = 0
        sl_bars_idx = 0
        
        for bar in range(200, n_bars):
            current_price = prices[bar]
            current_day = day_of_week[bar]
            
            # Update max drawdown
            if equity > peak_equity:
                peak_equity = equity
            dd = (peak_equity - equity) / peak_equity * 100.0
            if dd > max_dd:
                max_dd = dd
            
            ema_fast = ema_fast_all[bar, ema_fast_idx]
            ema_slow = ema_slow_all[bar, ema_slow_idx]
            ema_fast_prev = ema_fast_all[bar-1, ema_fast_idx]
            ema_slow_prev = ema_slow_all[bar-1, ema_slow_idx]
            ema_12h = ema_12h_all[bar, ema_12h_idx]
            rsi = rsi_all[bar, rsi_idx]
            atr = atr_all[bar, atr_idx]
            
            if position > 0:
                bars_held = bar - entry_bar
                
                # 1. Stop loss hit
                if low[bar] <= stop_loss:
                    exit_price = stop_loss
                    pnl = position * (exit_price - entry_price)
                    equity += pnl
                    trade_return = (exit_price - entry_price) / entry_price
                    position = 0.0
                    trades += 1
                    
                    if trade_idx < 500:
                        trade_returns[trade_idx] = trade_return
                        trade_idx += 1
                    
                    losses_count += 1
                    if loss_idx < 250:
                        loss_returns[loss_idx] = trade_return
                        loss_idx += 1
                    if sl_bars_idx < 250:
                        bars_in_sl[sl_bars_idx] = bars_held
                        sl_bars_idx += 1
                    continue
                
                # 2. Take profit hit
                if high[bar] >= take_profit:
                    exit_price = take_profit
                    pnl = position * (exit_price - entry_price)
                    equity += pnl
                    trade_return = (exit_price - entry_price) / entry_price
                    position = 0.0
                    trades += 1
                    wins += 1
                    
                    if trade_idx < 500:
                        trade_returns[trade_idx] = trade_return
                        trade_idx += 1
                    if win_idx < 250:
                        win_returns[win_idx] = trade_return
                        win_idx += 1
                    if tp_bars_idx < 250:
                        bars_in_tp[tp_bars_idx] = bars_held
                        tp_bars_idx += 1
                    if win_bars_idx < 250:
                        bars_in_wins[win_bars_idx] = bars_held
                        win_bars_idx += 1
                    continue
                
                # 3. Signal exit
                ema_cross_down = ema_fast_prev > ema_slow_prev and ema_fast <= ema_slow
                rsi_overbought = rsi > 70
                
                if ema_cross_down or rsi_overbought:
                    risk = entry_price - stop_loss
                    current_profit = current_price - entry_price
                    current_rr = current_profit / risk if risk > 0 else 0
                    
                    if current_rr >= 1.0:
                        exit_price = current_price
                        pnl = position * (exit_price - entry_price)
                        equity += pnl
                        trade_return = (exit_price - entry_price) / entry_price
                        position = 0.0
                        trades += 1
                        
                        if trade_idx < 500:
                            trade_returns[trade_idx] = trade_return
                            trade_idx += 1
                        
                        wins += 1
                        if win_idx < 250:
                            win_returns[win_idx] = trade_return
                            win_idx += 1
                        if win_bars_idx < 250:
                            bars_in_wins[win_bars_idx] = bars_held
                            win_bars_idx += 1
            
            else:
                # CHECK DAY OF WEEK FILTER FOR ENTRIES ONLY
                can_trade_today = day_masks[d, current_day]
                
                if not can_trade_today:
                    continue
                
                # Entry conditions
                ema_crossover = ema_fast_prev < ema_slow_prev and ema_fast > ema_slow
                rsi_ok = rsi > rsi_threshold
                trend_bullish = current_price > ema_12h
                
                if ema_crossover and rsi_ok and trend_bullish and atr > 0:
                    entry_price = current_price
                    entry_bar = bar
                    stop_loss = entry_price - (atr_multiplier * atr)
                    risk_per_unit = entry_price - stop_loss
                    take_profit = entry_price + (reward_risk_ratio * risk_per_unit)
                    
                    risk_amount = equity * risk_per_trade
                    position = risk_amount / risk_per_unit
                    
                    max_position = (equity * 0.99) / entry_price
                    if position > max_position:
                        position = max_position
        
        # Close final position
        if position > 0:
            final_price = prices[n_bars - 1]
            pnl = position * (final_price - entry_price)
            equity += pnl
            trade_return = (final_price - entry_price) / entry_price
            bars_held = n_bars - 1 - entry_bar
            
            if trade_idx < 500:
                trade_returns[trade_idx] = trade_return
                trade_idx += 1
            trades += 1
            
            if trade_return > 0:
                wins += 1
                if win_idx < 250:
                    win_returns[win_idx] = trade_return
                    win_idx += 1
                if win_bars_idx < 250:
                    bars_in_wins[win_bars_idx] = bars_held
                    win_bars_idx += 1
            else:
                losses_count += 1
                if loss_idx < 250:
                    loss_returns[loss_idx] = trade_return
                    loss_idx += 1
        
        # Calculate final stats
        final_equity = equity
        total_return = ((final_equity - initial_cash) / initial_cash) * 100.0
        
        if trade_idx > 1:
            returns_used = trade_returns[:trade_idx]
            mean_return = np.mean(returns_used)
            std_return = np.std(returns_used)
            if std_return > 0:
                sqn = (mean_return / std_return) * np.sqrt(trade_idx)
            else:
                sqn = 0.0
        else:
            sqn = 0.0
        
        win_rate = (wins / trades * 100.0) if trades > 0 else 0.0
        avg_win = np.mean(win_returns[:win_idx]) * 100.0 if win_idx > 0 else 0.0
        avg_loss = np.mean(loss_returns[:loss_idx]) * 100.0 if loss_idx > 0 else 0.0
        avg_bars_win = np.mean(bars_in_wins[:win_bars_idx].astype(np.float32)) if win_bars_idx > 0 else 0.0
        avg_bars_tp = np.mean(bars_in_tp[:tp_bars_idx].astype(np.float32)) if tp_bars_idx > 0 else 0.0
        avg_bars_sl = np.mean(bars_in_sl[:sl_bars_idx].astype(np.float32)) if sl_bars_idx > 0 else 0.0
        
        total_returns[idx] = total_return
        final_equities[idx] = final_equity
        trade_counts[idx] = trades
        sqn_values[idx] = sqn
        max_drawdowns[idx] = max_dd
        win_rates[idx] = win_rate
        avg_wins[idx] = avg_win
        avg_losses[idx] = avg_loss
        avg_bars_wins[idx] = avg_bars_win
        avg_bars_tps[idx] = avg_bars_tp
        avg_bars_sls[idx] = avg_bars_sl
        win_counts[idx] = wins
    
    return (total_returns, final_equities, trade_counts, sqn_values,
            max_drawdowns, win_rates, avg_wins, avg_losses,
            avg_bars_wins, avg_bars_tps, avg_bars_sls, win_counts, day_combo_indices)


# =============================================================================
# NUMBA-ACCELERATED PORTFOLIO SIMULATION WITH DETAILED STATS
# =============================================================================

@njit(parallel=True)
def simulate_strategy_batch_detailed(
    prices: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    ema_fast_all: np.ndarray,
    ema_slow_all: np.ndarray,
    ema_12h_all: np.ndarray,
    rsi_all: np.ndarray,
    atr_all: np.ndarray,
    param_indices: np.ndarray,
    params_values: np.ndarray,
    initial_cash: float,
    risk_per_trade: float,
    n_bars: int,
    n_params: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
           np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate EMA crossover strategy with detailed statistics.
    
    Returns:
        total_returns, final_equities, trade_counts, sqn_values,
        max_drawdowns, win_rates, avg_win, avg_loss,
        avg_bars_win, avg_bars_tp, avg_bars_sl, win_counts
    """
    
    total_returns = np.zeros(n_params, dtype=np.float32)
    final_equities = np.zeros(n_params, dtype=np.float32)
    trade_counts = np.zeros(n_params, dtype=np.int32)
    sqn_values = np.zeros(n_params, dtype=np.float32)
    max_drawdowns = np.zeros(n_params, dtype=np.float32)
    win_rates = np.zeros(n_params, dtype=np.float32)
    avg_wins = np.zeros(n_params, dtype=np.float32)
    avg_losses = np.zeros(n_params, dtype=np.float32)
    avg_bars_wins = np.zeros(n_params, dtype=np.float32)
    avg_bars_tps = np.zeros(n_params, dtype=np.float32)
    avg_bars_sls = np.zeros(n_params, dtype=np.float32)
    win_counts = np.zeros(n_params, dtype=np.int32)
    
    for p in prange(n_params):
        ema_fast_idx = param_indices[p, 0]
        ema_slow_idx = param_indices[p, 1]
        ema_12h_idx = param_indices[p, 2]
        rsi_idx = param_indices[p, 3]
        atr_idx = param_indices[p, 4]
        
        rsi_threshold = params_values[p, 0]
        atr_multiplier = params_values[p, 1]
        reward_risk_ratio = params_values[p, 2]
        
        equity = initial_cash
        peak_equity = initial_cash
        max_dd = 0.0
        position = 0.0
        entry_price = 0.0
        entry_bar = 0
        stop_loss = 0.0
        take_profit = 0.0
        
        trades = 0
        wins = 0
        losses_count = 0
        tp_exits = 0
        sl_exits = 0
        
        trade_returns = np.zeros(500, dtype=np.float32)
        trade_idx = 0
        
        # Track wins/losses separately
        win_returns = np.zeros(250, dtype=np.float32)
        loss_returns = np.zeros(250, dtype=np.float32)
        win_idx = 0
        loss_idx = 0
        
        # Track bars in trade
        bars_in_wins = np.zeros(250, dtype=np.int32)
        bars_in_tp = np.zeros(250, dtype=np.int32)
        bars_in_sl = np.zeros(250, dtype=np.int32)
        win_bars_idx = 0
        tp_bars_idx = 0
        sl_bars_idx = 0
        
        for bar in range(200, n_bars):
            current_price = prices[bar]
            
            # Update max drawdown
            if equity > peak_equity:
                peak_equity = equity
            dd = (peak_equity - equity) / peak_equity * 100.0
            if dd > max_dd:
                max_dd = dd
            
            ema_fast = ema_fast_all[bar, ema_fast_idx]
            ema_slow = ema_slow_all[bar, ema_slow_idx]
            ema_fast_prev = ema_fast_all[bar-1, ema_fast_idx]
            ema_slow_prev = ema_slow_all[bar-1, ema_slow_idx]
            ema_12h = ema_12h_all[bar, ema_12h_idx]
            rsi = rsi_all[bar, rsi_idx]
            atr = atr_all[bar, atr_idx]
            
            if position > 0:
                bars_held = bar - entry_bar
                
                # 1. Stop loss hit
                if low[bar] <= stop_loss:
                    exit_price = stop_loss
                    pnl = position * (exit_price - entry_price)
                    equity += pnl
                    trade_return = (exit_price - entry_price) / entry_price
                    position = 0.0
                    trades += 1
                    sl_exits += 1
                    
                    if trade_idx < 500:
                        trade_returns[trade_idx] = trade_return
                        trade_idx += 1
                    
                    # It's a loss (SL hit)
                    losses_count += 1
                    if loss_idx < 250:
                        loss_returns[loss_idx] = trade_return
                        loss_idx += 1
                    if sl_bars_idx < 250:
                        bars_in_sl[sl_bars_idx] = bars_held
                        sl_bars_idx += 1
                    continue
                
                # 2. Take profit hit
                if high[bar] >= take_profit:
                    exit_price = take_profit
                    pnl = position * (exit_price - entry_price)
                    equity += pnl
                    trade_return = (exit_price - entry_price) / entry_price
                    position = 0.0
                    trades += 1
                    tp_exits += 1
                    wins += 1
                    
                    if trade_idx < 500:
                        trade_returns[trade_idx] = trade_return
                        trade_idx += 1
                    if win_idx < 250:
                        win_returns[win_idx] = trade_return
                        win_idx += 1
                    if tp_bars_idx < 250:
                        bars_in_tp[tp_bars_idx] = bars_held
                        tp_bars_idx += 1
                    if win_bars_idx < 250:
                        bars_in_wins[win_bars_idx] = bars_held
                        win_bars_idx += 1
                    continue
                
                # 3. Signal exit (EMA cross down or RSI overbought)
                ema_cross_down = ema_fast_prev > ema_slow_prev and ema_fast <= ema_slow
                rsi_overbought = rsi > 70
                
                if ema_cross_down or rsi_overbought:
                    risk = entry_price - stop_loss
                    current_profit = current_price - entry_price
                    current_rr = current_profit / risk if risk > 0 else 0
                    
                    if current_rr >= 1.0:
                        exit_price = current_price
                        pnl = position * (exit_price - entry_price)
                        equity += pnl
                        trade_return = (exit_price - entry_price) / entry_price
                        position = 0.0
                        trades += 1
                        
                        if trade_idx < 500:
                            trade_returns[trade_idx] = trade_return
                            trade_idx += 1
                        
                        # Signal exit is a win (since RR >= 1)
                        wins += 1
                        if win_idx < 250:
                            win_returns[win_idx] = trade_return
                            win_idx += 1
                        if win_bars_idx < 250:
                            bars_in_wins[win_bars_idx] = bars_held
                            win_bars_idx += 1
            
            else:
                # Entry conditions
                ema_crossover = ema_fast_prev < ema_slow_prev and ema_fast > ema_slow
                rsi_ok = rsi > rsi_threshold
                trend_bullish = current_price > ema_12h
                
                if ema_crossover and rsi_ok and trend_bullish and atr > 0:
                    entry_price = current_price
                    entry_bar = bar
                    stop_loss = entry_price - (atr_multiplier * atr)
                    risk_per_unit = entry_price - stop_loss
                    take_profit = entry_price + (reward_risk_ratio * risk_per_unit)
                    
                    risk_amount = equity * risk_per_trade
                    position = risk_amount / risk_per_unit
                    
                    max_position = (equity * 0.99) / entry_price
                    if position > max_position:
                        position = max_position
        
        # Close final position
        if position > 0:
            final_price = prices[n_bars - 1]
            pnl = position * (final_price - entry_price)
            equity += pnl
            trade_return = (final_price - entry_price) / entry_price
            bars_held = n_bars - 1 - entry_bar
            
            if trade_idx < 500:
                trade_returns[trade_idx] = trade_return
                trade_idx += 1
            trades += 1
            
            if trade_return > 0:
                wins += 1
                if win_idx < 250:
                    win_returns[win_idx] = trade_return
                    win_idx += 1
                if win_bars_idx < 250:
                    bars_in_wins[win_bars_idx] = bars_held
                    win_bars_idx += 1
            else:
                losses_count += 1
                if loss_idx < 250:
                    loss_returns[loss_idx] = trade_return
                    loss_idx += 1
        
        # Calculate final stats
        final_equity = equity
        total_return = ((final_equity - initial_cash) / initial_cash) * 100.0
        
        # SQN
        if trade_idx > 1:
            returns_used = trade_returns[:trade_idx]
            mean_return = np.mean(returns_used)
            std_return = np.std(returns_used)
            if std_return > 0:
                sqn = (mean_return / std_return) * np.sqrt(trade_idx)
            else:
                sqn = 0.0
        else:
            sqn = 0.0
        
        # Win rate
        win_rate = (wins / trades * 100.0) if trades > 0 else 0.0
        
        # Average win/loss
        avg_win = np.mean(win_returns[:win_idx]) * 100.0 if win_idx > 0 else 0.0
        avg_loss = np.mean(loss_returns[:loss_idx]) * 100.0 if loss_idx > 0 else 0.0
        
        # Average bars
        avg_bars_win = np.mean(bars_in_wins[:win_bars_idx].astype(np.float32)) if win_bars_idx > 0 else 0.0
        avg_bars_tp = np.mean(bars_in_tp[:tp_bars_idx].astype(np.float32)) if tp_bars_idx > 0 else 0.0
        avg_bars_sl = np.mean(bars_in_sl[:sl_bars_idx].astype(np.float32)) if sl_bars_idx > 0 else 0.0
        
        total_returns[p] = total_return
        final_equities[p] = final_equity
        trade_counts[p] = trades
        sqn_values[p] = sqn
        max_drawdowns[p] = max_dd
        win_rates[p] = win_rate
        avg_wins[p] = avg_win
        avg_losses[p] = avg_loss
        avg_bars_wins[p] = avg_bars_win
        avg_bars_tps[p] = avg_bars_tp
        avg_bars_sls[p] = avg_bars_sl
        win_counts[p] = wins
    
    return (total_returns, final_equities, trade_counts, sqn_values,
            max_drawdowns, win_rates, avg_wins, avg_losses,
            avg_bars_wins, avg_bars_tps, avg_bars_sls, win_counts)


# =============================================================================
# DAY-OF-WEEK COMBINATIONS GENERATOR
# =============================================================================

def generate_day_combinations():
    """
    Generate meaningful day-of-week combinations to test.
    0=Monday, 1=Tuesday, 2=Wednesday, 3=Thursday, 4=Friday, 5=Saturday, 6=Sunday
    
    Returns list of (name, mask) tuples where mask is a 7-element boolean array.
    """
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    combinations = []
    
    # All days (baseline)
    combinations.append(('All Days', [True]*7))
    
    # Weekdays only (Mon-Fri)
    combinations.append(('Weekdays', [True, True, True, True, True, False, False]))
    
    # Weekend only
    combinations.append(('Weekend', [False, False, False, False, False, True, True]))
    
    # Exclude individual days
    for i in range(7):
        mask = [True]*7
        mask[i] = False
        combinations.append((f'No {day_names[i]}', mask))
    
    # Pairs of excluded days
    exclude_pairs = [
        (5, 6, 'No Weekend'),  # No Sat+Sun
        (0, 6, 'No Mon+Sun'),
        (0, 1, 'No Mon+Tue'),
        (4, 5, 'No Fri+Sat'),
        (3, 4, 'No Thu+Fri'),
    ]
    for d1, d2, name in exclude_pairs:
        mask = [True]*7
        mask[d1] = False
        mask[d2] = False
        combinations.append((name, mask))
    
    # Specific trading windows
    combinations.append(('Mon-Wed', [True, True, True, False, False, False, False]))
    combinations.append(('Wed-Fri', [False, False, True, True, True, False, False]))
    combinations.append(('Tue-Thu', [False, True, True, True, False, False, False]))
    combinations.append(('Mon-Thu', [True, True, True, True, False, False, False]))
    combinations.append(('Tue-Fri', [False, True, True, True, True, False, False]))
    
    # Best 3 days (common patterns)
    combinations.append(('Mon+Wed+Fri', [True, False, True, False, True, False, False]))
    combinations.append(('Tue+Thu+Sat', [False, True, False, True, False, True, False]))
    combinations.append(('Mon+Tue+Wed', [True, True, True, False, False, False, False]))
    combinations.append(('Wed+Thu+Fri', [False, False, True, True, True, False, False]))
    
    return combinations


# =============================================================================
# MAIN OPTIMIZATION FUNCTION WITH DAY FILTER
# =============================================================================

def optimize_emacross_strategy(
    symbol: str,
    timeframe: str = '4h',
    start_date: str = '2021-01-01',
    end_date: str = '2025-12-01',
    db_path: str = 'data/crypto.db',
    initial_cash: float = 10000.0
) -> Dict[str, Any]:
    """
    Optimize EMA crossover strategy for a single symbol with day-of-week filtering.
    """
    
    print(f"\n{'='*60}")
    print(f"Optimizing EMA Crossover RSI for {symbol}")
    print(f"{'='*60}")
    
    # Load data
    symbol_normalized = symbol.replace('/', '')
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
    df = pd.read_sql_query(query, conn, params=(symbol_normalized, timeframe, start_ts, end_ts))
    conn.close()
    
    if len(df) == 0:
        return {'error': f"No data for {symbol}"}
    
    # Convert timestamp to datetime and extract day of week
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df['day_of_week'] = df['datetime'].dt.dayofweek  # 0=Monday, 6=Sunday
    
    print(f"Data loaded: {len(df)} bars")
    
    prices = df['close'].values.astype(np.float32)
    high = df['high'].values.astype(np.float32)
    low = df['low'].values.astype(np.float32)
    day_of_week = df['day_of_week'].values.astype(np.int32)
    
    # Use best parameters from previous optimization for each symbol
    # We'll test day combinations with the top 5 parameter sets
    best_params_by_symbol = {
        'BTCUSDT': [
            {'ema_fast': 8, 'ema_slow': 21, 'ema_12h': 13, 'rsi_period': 14, 'rsi_threshold': 40, 'atr_period': 21, 'atr_multiplier': 1.0, 'reward_risk_ratio': 4.0},
            {'ema_fast': 8, 'ema_slow': 34, 'ema_12h': 21, 'rsi_period': 14, 'rsi_threshold': 55, 'atr_period': 14, 'atr_multiplier': 1.0, 'reward_risk_ratio': 4.0},
            {'ema_fast': 8, 'ema_slow': 21, 'ema_12h': 13, 'rsi_period': 10, 'rsi_threshold': 55, 'atr_period': 21, 'atr_multiplier': 1.0, 'reward_risk_ratio': 4.0},
        ],
        'ETHUSDT': [
            {'ema_fast': 8, 'ema_slow': 21, 'ema_12h': 21, 'rsi_period': 14, 'rsi_threshold': 55, 'atr_period': 21, 'atr_multiplier': 1.0, 'reward_risk_ratio': 3.0},
            {'ema_fast': 8, 'ema_slow': 55, 'ema_12h': 34, 'rsi_period': 21, 'rsi_threshold': 50, 'atr_period': 10, 'atr_multiplier': 1.0, 'reward_risk_ratio': 4.0},
            {'ema_fast': 8, 'ema_slow': 21, 'ema_12h': 21, 'rsi_period': 10, 'rsi_threshold': 55, 'atr_period': 21, 'atr_multiplier': 1.0, 'reward_risk_ratio': 3.5},
        ],
        'SOLUSDT': [
            {'ema_fast': 13, 'ema_slow': 55, 'ema_12h': 21, 'rsi_period': 21, 'rsi_threshold': 50, 'atr_period': 21, 'atr_multiplier': 1.0, 'reward_risk_ratio': 4.0},
            {'ema_fast': 13, 'ema_slow': 55, 'ema_12h': 13, 'rsi_period': 21, 'rsi_threshold': 50, 'atr_period': 21, 'atr_multiplier': 1.0, 'reward_risk_ratio': 4.0},
            {'ema_fast': 13, 'ema_slow': 55, 'ema_12h': 21, 'rsi_period': 21, 'rsi_threshold': 45, 'atr_period': 14, 'atr_multiplier': 1.0, 'reward_risk_ratio': 4.0},
        ],
        'BNBUSDT': [
            {'ema_fast': 8, 'ema_slow': 89, 'ema_12h': 34, 'rsi_period': 7, 'rsi_threshold': 55, 'atr_period': 14, 'atr_multiplier': 1.5, 'reward_risk_ratio': 3.5},
            {'ema_fast': 8, 'ema_slow': 89, 'ema_12h': 34, 'rsi_period': 7, 'rsi_threshold': 55, 'atr_period': 21, 'atr_multiplier': 1.5, 'reward_risk_ratio': 3.5},
            {'ema_fast': 8, 'ema_slow': 89, 'ema_12h': 34, 'rsi_period': 10, 'rsi_threshold': 55, 'atr_period': 14, 'atr_multiplier': 1.5, 'reward_risk_ratio': 3.5},
        ],
    }
    
    # Use best params for this symbol, or default
    valid_combinations = best_params_by_symbol.get(symbol, best_params_by_symbol['BTCUSDT'])
    
    # Generate day combinations
    day_combinations = generate_day_combinations()
    day_combo_names = [d[0] for d in day_combinations]
    day_masks = np.array([d[1] for d in day_combinations], dtype=np.bool_)
    n_day_combos = len(day_combinations)
    
    print(f"Testing {len(valid_combinations)} param sets × {n_day_combos} day combos = {len(valid_combinations) * n_day_combos} total")
    
    # Pre-calculate indicators for needed periods
    all_ema_fast = sorted(set(p['ema_fast'] for p in valid_combinations))
    all_ema_slow = sorted(set(p['ema_slow'] for p in valid_combinations))
    all_ema_12h = sorted(set(p['ema_12h'] for p in valid_combinations))
    all_rsi = sorted(set(p['rsi_period'] for p in valid_combinations))
    all_atr = sorted(set(p['atr_period'] for p in valid_combinations))
    
    print("\nPre-calculating indicators...")
    indicator_start = datetime.now()
    
    indicators = batch_calculate_indicators(
        prices, high, low,
        ema_fast_periods=all_ema_fast,
        ema_slow_periods=all_ema_slow,
        ema_12h_periods=all_ema_12h,
        rsi_periods=all_rsi,
        atr_periods=all_atr
    )
    
    indicator_time = (datetime.now() - indicator_start).total_seconds()
    print(f"Indicators calculated in {indicator_time:.2f}s")
    
    # Build index maps
    ema_fast_idx_map = {p: i for i, p in enumerate(all_ema_fast)}
    ema_slow_idx_map = {p: i for i, p in enumerate(all_ema_slow)}
    ema_12h_idx_map = {p: i for i, p in enumerate(all_ema_12h)}
    rsi_idx_map = {p: i for i, p in enumerate(all_rsi)}
    atr_idx_map = {p: i for i, p in enumerate(all_atr)}
    
    # Stack indicator arrays
    n_bars = len(prices)
    
    ema_fast_all = np.stack([indicators['ema_fast'][p] for p in all_ema_fast], axis=1).astype(np.float32)
    ema_slow_all = np.stack([indicators['ema_slow'][p] for p in all_ema_slow], axis=1).astype(np.float32)
    ema_12h_all = np.stack([indicators['ema_12h'][p] for p in all_ema_12h], axis=1).astype(np.float32)
    rsi_all = np.stack([indicators['rsi'][p] for p in all_rsi], axis=1).astype(np.float32)
    atr_all = np.stack([indicators['atr'][p] for p in all_atr], axis=1).astype(np.float32)
    
    # Build parameter index arrays
    n_params = len(valid_combinations)
    param_indices = np.zeros((n_params, 5), dtype=np.int32)
    params_values = np.zeros((n_params, 3), dtype=np.float32)
    
    for i, p in enumerate(valid_combinations):
        param_indices[i, 0] = ema_fast_idx_map[p['ema_fast']]
        param_indices[i, 1] = ema_slow_idx_map[p['ema_slow']]
        param_indices[i, 2] = ema_12h_idx_map[p['ema_12h']]
        param_indices[i, 3] = rsi_idx_map[p['rsi_period']]
        param_indices[i, 4] = atr_idx_map[p['atr_period']]
        
        params_values[i, 0] = p['rsi_threshold']
        params_values[i, 1] = p['atr_multiplier']
        params_values[i, 2] = p['reward_risk_ratio']
    
    # Run simulation with day combinations
    print(f"\nRunning {n_params * n_day_combos} simulations...")
    sim_start = datetime.now()
    
    risk_per_trade = 0.01
    
    (total_returns, final_equities, trade_counts, sqn_values,
     max_drawdowns, win_rates, avg_wins, avg_losses,
     avg_bars_wins, avg_bars_tps, avg_bars_sls, win_counts, 
     day_combo_indices) = simulate_strategy_batch_with_days(
        prices, high, low, day_of_week,
        ema_fast_all, ema_slow_all, ema_12h_all, rsi_all, atr_all,
        param_indices, params_values, day_masks,
        initial_cash, risk_per_trade, n_bars, n_params, n_day_combos
    )
    
    sim_time = (datetime.now() - sim_start).total_seconds()
    n_total = n_params * n_day_combos
    throughput = n_total / max(sim_time, 0.001)  # Avoid division by zero
    print(f"Simulation complete in {sim_time:.2f}s ({throughput:.0f} tests/sec)")
    
    # Build results DataFrame
    results_data = []
    for idx in range(n_total):
        p_idx = idx // n_day_combos
        d_idx = idx % n_day_combos
        p = valid_combinations[p_idx]
        
        results_data.append({
            **p,
            'day_combo': day_combo_names[d_idx],
            'day_combo_idx': d_idx,
            'total_return_pct': float(total_returns[idx]),
            'final_equity': float(final_equities[idx]),
            'trades': int(trade_counts[idx]),
            'sqn': float(sqn_values[idx]),
            'max_drawdown': float(max_drawdowns[idx]),
            'win_rate': float(win_rates[idx]),
            'avg_win_pct': float(avg_wins[idx]),
            'avg_loss_pct': float(avg_losses[idx]),
            'avg_bars_win': float(avg_bars_wins[idx]),
            'avg_bars_tp': float(avg_bars_tps[idx]),
            'avg_bars_sl': float(avg_bars_sls[idx]),
            'wins': int(win_counts[idx]),
            'losses': int(trade_counts[idx]) - int(win_counts[idx])
        })
    
    results_df = pd.DataFrame(results_data)
    
    # Filter by max drawdown < 10% and minimum trades
    results_df_filtered = results_df[(results_df['max_drawdown'] <= 10.0) & (results_df['trades'] >= 20)].copy()
    
    # Sort by RETURN
    results_df_filtered = results_df_filtered.sort_values(['total_return_pct', 'sqn'], ascending=[False, False])
    
    # Print top 15 results
    print(f"\n{'='*70}")
    print(f"TOP 15 RESULTS FOR {symbol} (Max DD <= 10%, sorted by Return)")
    print(f"{'='*70}")
    print(f"Total combinations with DD <= 10%: {len(results_df_filtered)}/{len(results_df)}")
    
    if len(results_df_filtered) == 0:
        print("No combinations found with max drawdown <= 10%!")
        results_df_filtered = results_df.nsmallest(15, 'max_drawdown')
    
    top_15 = results_df_filtered.head(15)
    for rank, (idx, row) in enumerate(top_15.iterrows(), 1):
        print(f"\n#{rank}: [{row['day_combo']}]")
        print(f"  EMA: {int(row['ema_fast'])}/{int(row['ema_slow'])}/{int(row['ema_12h'])}, RSI: {int(row['rsi_period'])}/{int(row['rsi_threshold'])}, ATR: {row['atr_multiplier']:.1f}x, R:R: {row['reward_risk_ratio']:.1f}")
        print(f"  → Return: {row['total_return_pct']:.2f}%, SQN: {row['sqn']:.2f}, Max DD: {row['max_drawdown']:.1f}%")
        print(f"  → Trades: {int(row['trades'])} (W:{int(row['wins'])}/L:{int(row['losses'])}), Win Rate: {row['win_rate']:.1f}%")
    
    # Day analysis - which days are best?
    print(f"\n{'='*70}")
    print(f"DAY COMBINATION ANALYSIS FOR {symbol}")
    print(f"{'='*70}")
    
    day_summary = results_df_filtered.groupby('day_combo').agg({
        'total_return_pct': 'max',
        'max_drawdown': 'min',
        'win_rate': 'max',
        'trades': 'mean'
    }).sort_values('total_return_pct', ascending=False)
    
    print(f"\n{'Day Combo':<20} {'Best Return':>12} {'Min DD':>10} {'Best WR':>10} {'Avg Trades':>12}")
    print("-" * 70)
    for day_combo, row in day_summary.head(10).iterrows():
        print(f"{day_combo:<20} {row['total_return_pct']:>11.2f}% {row['max_drawdown']:>9.1f}% {row['win_rate']:>9.1f}% {row['trades']:>11.0f}")
    
    best = results_df_filtered.iloc[0] if len(results_df_filtered) > 0 else results_df.iloc[0]
    
    return {
        'symbol': symbol,
        'best_params': {
            'ema_fast': int(best['ema_fast']),
            'ema_slow': int(best['ema_slow']),
            'ema_12h_period': int(best['ema_12h']),
            'rsi_period': int(best['rsi_period']),
            'rsi_threshold': int(best['rsi_threshold']),
            'atr_period': int(best['atr_period']),
            'atr_multiplier': float(best['atr_multiplier']),
            'reward_risk_ratio': float(best['reward_risk_ratio']),
            'trading_days': best['day_combo']
        },
        'best_sqn': float(best['sqn']),
        'best_return': float(best['total_return_pct']),
        'best_trades': int(best['trades']),
        'best_max_dd': float(best['max_drawdown']),
        'best_win_rate': float(best['win_rate']),
        'best_avg_win': float(best['avg_win_pct']),
        'best_avg_loss': float(best['avg_loss_pct']),
        'day_analysis': day_summary.head(10).to_dict(),
        'total_combinations': n_total,
        'optimization_time': sim_time + indicator_time,
        'throughput': throughput
    }


def main():
    """Optimize for all 4 target pairs with day-of-week analysis."""
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']
    all_results = {}
    
    print("\n" + "="*80)
    print("GPU-ACCELERATED EMA CROSSOVER RSI STRATEGY OPTIMIZATION")
    print("WITH DAY-OF-WEEK ANALYSIS")
    print("="*80)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Timeframe: 4H")
    print(f"Period: 2021-01-01 to 2025-12-01")
    print("="*80)
    
    total_start = datetime.now()
    
    for symbol in symbols:
        result = optimize_emacross_strategy(
            symbol=symbol,
            timeframe='4h',
            start_date='2021-01-01',
            end_date='2025-12-01',
            initial_cash=10000000.0
        )
        all_results[symbol] = result
    
    total_time = (datetime.now() - total_start).total_seconds()
    
    # Summary
    print("\n" + "="*80)
    print("OPTIMIZATION SUMMARY WITH BEST TRADING DAYS")
    print("="*80)
    
    for symbol, result in all_results.items():
        if 'error' in result:
            print(f"\n{symbol}: ERROR - {result['error']}")
            continue
            
        print(f"\n{symbol}:")
        print(f"  Best Return: {result['best_return']:.2f}%")
        print(f"  Max Drawdown: {result['best_max_dd']:.1f}%")
        print(f"  Trades: {result['best_trades']}, Win Rate: {result['best_win_rate']:.1f}%")
        print(f"  Trading Days: {result['best_params']['trading_days']}")
        print(f"  Parameters: EMA {result['best_params']['ema_fast']}/{result['best_params']['ema_slow']}/{result['best_params']['ema_12h_period']}")
    
    print(f"\nTotal optimization time: {total_time:.1f}s")
    print(f"Total combinations tested: {sum(r.get('total_combinations', 0) for r in all_results.values())}")
    
    return all_results


if __name__ == '__main__':
    results = main()
