"""
DCA (Dollar Cost Averaging) Trading Strategies

This module implements two variations of DCA strategies for cryptocurrency trading:
1. Monthly DCA Strategy - Regular scheduled buys with rebalancing
2. Signal-Based DCA Strategy - Opportunistic buying based on technical signals
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from backtesting import Strategy
from dataclasses import dataclass, field
import ta

from ..core.backtesting_engine import BaseStrategy


# =====================================================================
# HELPER FUNCTIONS FOR TECHNICAL ANALYSIS
# =====================================================================

def calculate_ema(close_prices: pd.Series, period: int = 200) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return ta.trend.ema_indicator(close_prices, window=period).ffill().bfill()


def calculate_rsi(close_prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    return ta.momentum.rsi(close_prices, window=period).ffill().bfill().fillna(50)


def calculate_volume_avg(volume: pd.Series, period: int = 20) -> pd.Series:
    """Calculate average volume."""
    return volume.rolling(window=period).mean().ffill().bfill()


def count_red_days(close_prices: pd.Series, lookback: int = 10) -> int:
    """Count consecutive red (down) days."""
    if len(close_prices) < 2:
        return 0
    
    red_count = 0
    for i in range(len(close_prices) - 1, max(0, len(close_prices) - lookback - 1), -1):
        if i > 0 and close_prices.iloc[i] < close_prices.iloc[i-1]:
            red_count += 1
        else:
            break
    return red_count


def is_v_shape_bottom(close_prices: pd.Series, lookback: int = 20, threshold: float = 0.25) -> bool:
    """
    Detect V-shape pattern (price in lowest 25% of recent range and starting to recover).
    
    Args:
        close_prices: Price series
        lookback: Number of periods to analyze
        threshold: Percentile threshold (0.25 = bottom 25%)
    
    Returns:
        True if V-shape bottom detected
    """
    if len(close_prices) < lookback + 2:
        return False
    
    recent = close_prices.iloc[-lookback:]
    current = close_prices.iloc[-1]
    prev = close_prices.iloc[-2]
    
    # Check if current price is in lowest 25% of recent range
    percentile_25 = recent.quantile(threshold)
    
    # Check if price is recovering (current > previous)
    is_recovering = current > prev
    
    return current <= percentile_25 and is_recovering


def is_inverse_v_shape_top(close_prices: pd.Series, lookback: int = 20, threshold: float = 0.75) -> bool:
    """
    Detect inverse V-shape pattern (price in highest 25% and starting to decline).
    
    Args:
        close_prices: Price series
        lookback: Number of periods to analyze
        threshold: Percentile threshold (0.75 = top 25%)
    
    Returns:
        True if inverse V-shape top detected
    """
    if len(close_prices) < lookback + 2:
        return False
    
    recent = close_prices.iloc[-lookback:]
    current = close_prices.iloc[-1]
    prev = close_prices.iloc[-2]
    
    # Check if current price is in highest 25% of recent range
    percentile_75 = recent.quantile(threshold)
    
    # Check if price peaked and declining
    is_declining = current < prev
    
    return current >= percentile_75 and is_declining


def calculate_price_acceleration(close_prices: pd.Series, days: int = 2) -> float:
    """
    Calculate price acceleration (% change over recent days).
    
    Returns:
        Percentage change (negative means decline)
    """
    if len(close_prices) < days + 1:
        return 0.0
    
    old_price = close_prices.iloc[-(days + 1)]
    current_price = close_prices.iloc[-1]
    
    if old_price == 0:
        return 0.0
    
    return ((current_price - old_price) / old_price) * 100


# =====================================================================
# DATA CLASSES FOR TRACKING
# =====================================================================

@dataclass
class CoinPosition:
    """Track individual coin position and statistics."""
    symbol: str
    holdings: float = 0.0  # Number of coins held
    total_invested: float = 0.0  # Total USD invested
    buy_count: int = 0  # Number of buys executed
    monthly_buy_count: int = 0  # Buys this month
    last_buy_price: float = 0.0  # Last buy price
    last_buy_date: Optional[datetime] = None
    red_day_count: int = 0  # Consecutive red days
    stop_buying: bool = False  # Stop buying flag (after sell until 200 EMA)
    monthly_budget_used: float = 0.0  # Budget used this month
    monthly_budget: float = 0.0  # Total monthly budget
    
    @property
    def average_cost(self) -> float:
        """Calculate average cost basis."""
        if self.holdings == 0:
            return 0.0
        return self.total_invested / self.holdings
    
    @property
    def current_profit_pct(self, current_price: float) -> float:
        """Calculate current profit percentage."""
        if self.average_cost == 0:
            return 0.0
        return ((current_price - self.average_cost) / self.average_cost) * 100


# =====================================================================
# STRATEGY 1: MONTHLY DCA WITH REBALANCING
# =====================================================================

class DCAMonthlyStrategy(BaseStrategy):
    """
    Monthly DCA Strategy
    
    Budget Allocation:
    - $600/month split: BTC 50% | ETH 25% | BNB 15% | TRX 10%
    - Each coin's budget divided into 5 equal parts
    
    Buy Rules:
    - Standard: After 2 red days → buy 1/5
    - Oversold: Price down ≥10% → buy 2/5
    - Minimum: 2 buys per month per coin
    
    Sell Rules:
    - Price up ≥15% from 200 EMA AND profit ≥10% from avg cost
    - Sell 15% of holdings (max 1x/week)
    - Keep 20% minimum
    - After sell: stop buying until price hits 200 EMA
    
    Monthly Rebalancing:
    - Unused capital + new funds split 50/50
    - 50% follows original ratio, 50% goes to underused coins
    """
    
    # Strategy parameters
    monthly_budget = 600.0
    allocation_ratios = {
        'BTC': 0.50,
        'ETH': 0.25,
        'BNB': 0.15,
        'TRX': 0.10
    }
    
    # Buy rules
    red_days_trigger = 2
    oversold_threshold = -10.0  # % decline
    oversold_multiplier = 2  # Buy 2x on oversold
    min_monthly_buys = 2
    
    # Sell rules
    ema_profit_threshold = 15.0  # % above 200 EMA
    avg_cost_profit_threshold = 10.0  # % above avg cost
    sell_percentage = 15.0  # % of holdings to sell
    min_holdings_pct = 20.0  # Minimum % to keep
    max_sell_frequency_days = 7  # Max 1 sell per week
    
    # Technical indicators
    ema_period = 200
    
    def init(self):
        """Initialize indicators and tracking variables."""
        super().init()
        
        # Calculate 200 EMA
        self.ema_200 = self.I(calculate_ema, self.data.Close, self.ema_period)
        
        # Initialize tracking
        self.current_month = None
        self.cash_available = self.monthly_budget  # Start with first month's budget
        self.unused_capital = 0.0  # Accumulates from unused monthly budget
        
        # Track positions for each coin (simulating multi-asset)
        # Note: In actual multi-asset, we'd track separately
        self.position_data = CoinPosition(
            symbol=self.data.symbol if hasattr(self.data, 'symbol') else 'CRYPTO',
            monthly_budget=self.monthly_budget
        )
        
        self.last_sell_date = None
        self.monthly_rebalance_done = False
    
    def next(self):
        """Execute strategy logic each time step."""
        if not self.should_trade():
            return
        
        current_date = self.data.index[-1]
        current_price = self.data.Close[-1]
        current_month = (current_date.year, current_date.month)
        
        # Check for new month - do rebalancing
        if self.current_month != current_month:
            self._new_month_rebalancing(current_date)
            self.current_month = current_month
        
        # Update red day counter
        if len(self.data.Close) >= 2:
            if self.data.Close[-1] < self.data.Close[-2]:
                self.position_data.red_day_count += 1
            else:
                self.position_data.red_day_count = 0
        
        # Check sell conditions first
        if self.position.size > 0:
            self._check_sell_conditions(current_date, current_price)
        
        # Check buy conditions
        if not self.position_data.stop_buying:
            self._check_buy_conditions(current_date, current_price)
    
    def _new_month_rebalancing(self, current_date: datetime):
        """
        Handle monthly rebalancing logic.
        
        Total available = Unused + $600 + extra funds
        Split 50/50:
        - 50% follows original ratio
        - 50% goes proportionally to underused coins
        """
        # Calculate total available capital
        total_available = self.unused_capital + self.monthly_budget
        
        # Simple rebalancing: just add to available cash
        # In multi-asset version, this would distribute across coins
        self.cash_available = total_available
        self.unused_capital = 0.0
        
        # Reset monthly tracking
        self.position_data.monthly_buy_count = 0
        self.position_data.monthly_budget_used = 0.0
        self.monthly_rebalance_done = True
        
        # Calculate new 1/5 amounts based on allocation
        # For single asset testing, we use full budget / 5
        self.position_data.monthly_budget = self.monthly_budget / 5
    
    def _check_buy_conditions(self, current_date: datetime, current_price: float):
        """Check if buy conditions are met."""
        # Calculate buy size (1/5 of monthly budget)
        buy_size_usd = self.position_data.monthly_budget
        
        # Check if we have enough cash
        if self.cash_available < buy_size_usd:
            return
        
        # Get 200 EMA value
        ema_200_value = self.ema_200[-1]
        
        # Check for oversold condition (≥10% below last buy or EMA)
        reference_price = max(self.position_data.last_buy_price, ema_200_value) if self.position_data.last_buy_price > 0 else ema_200_value
        price_decline_pct = ((current_price - reference_price) / reference_price) * 100 if reference_price > 0 else 0
        
        is_oversold = price_decline_pct <= self.oversold_threshold
        
        # Determine buy multiplier
        buy_multiplier = self.oversold_multiplier if is_oversold else 1
        
        # Check standard buy condition: 2 red days
        should_buy_standard = self.position_data.red_day_count >= self.red_days_trigger
        
        # Check if we should buy
        should_buy = should_buy_standard or is_oversold
        
        if should_buy and self.position_data.monthly_buy_count < 5:
            # Calculate actual buy amount
            buy_amount_usd = buy_size_usd * buy_multiplier
            buy_amount_usd = min(buy_amount_usd, self.cash_available)
            
            # Calculate number of coins to buy
            coins_to_buy = buy_amount_usd / current_price
            
            # Execute buy
            if coins_to_buy > 0:
                self.buy(size=coins_to_buy)
                
                # Update tracking
                self.position_data.holdings += coins_to_buy
                self.position_data.total_invested += buy_amount_usd
                self.position_data.buy_count += 1
                self.position_data.monthly_buy_count += 1
                self.position_data.last_buy_price = current_price
                self.position_data.last_buy_date = current_date
                self.position_data.red_day_count = 0  # Reset counter
                self.position_data.monthly_budget_used += buy_amount_usd
                self.cash_available -= buy_amount_usd
    
    def _check_sell_conditions(self, current_date: datetime, current_price: float):
        """Check if sell conditions are met."""
        if self.position.size == 0:
            return
        
        # Check sell frequency limit (max 1x per week)
        if self.last_sell_date:
            days_since_sell = (current_date - self.last_sell_date).days
            if days_since_sell < self.max_sell_frequency_days:
                return
        
        # Get 200 EMA value
        ema_200_value = self.ema_200[-1]
        
        # Check condition 1: Price up ≥15% from 200 EMA
        if ema_200_value > 0:
            ema_profit_pct = ((current_price - ema_200_value) / ema_200_value) * 100
        else:
            ema_profit_pct = 0
        
        # Check condition 2: Profit ≥10% from average cost
        avg_cost = self.position_data.average_cost
        if avg_cost > 0:
            cost_profit_pct = ((current_price - avg_cost) / avg_cost) * 100
        else:
            cost_profit_pct = 0
        
        # Both conditions must be met
        if ema_profit_pct >= self.ema_profit_threshold and cost_profit_pct >= self.avg_cost_profit_threshold:
            # Calculate sell amount (15% of holdings, keep minimum 20%)
            current_holdings = self.position.size
            sell_amount = current_holdings * (self.sell_percentage / 100)
            
            # Ensure we keep minimum 20%
            min_holdings = self.position_data.holdings * (self.min_holdings_pct / 100)
            max_sell = current_holdings - min_holdings
            
            if max_sell > 0:
                sell_amount = min(sell_amount, max_sell)
                
                # Execute sell
                self.position.close(sell_amount / current_holdings)  # Close percentage
                
                # Update tracking
                sell_proceeds = sell_amount * current_price
                self.cash_available += sell_proceeds
                self.position_data.holdings -= sell_amount
                self.last_sell_date = current_date
                
                # Set stop buying flag until price hits 200 EMA
                self.position_data.stop_buying = True
        
        # Check if we should resume buying (price back to 200 EMA)
        if self.position_data.stop_buying and current_price <= ema_200_value:
            self.position_data.stop_buying = False


# =====================================================================
# STRATEGY 2: SIGNAL-BASED DCA
# =====================================================================

class DCASignalStrategy(BaseStrategy):
    """
    Signal-Based DCA Strategy
    
    Core Structure:
    - $600/month into cash pool
    - No scheduled buys - only on strong signals
    - Cash accumulates between signals
    
    Position Sizing (Dynamic):
    - Base: 1/10 to 1/25 depending on accumulated cash
    - Bonuses: +15-25% for price drops, ×1.3-1.5 for high signals
    
    Buy Signal (6+ points required):
    - Distance from 200 EMA: 1-3 pts
    - RSI(14): 1-2 pts
    - V-shape pattern: 2 pts
    - Acceleration: 1 pt
    - Volume spike: 1 pt
    - Price vs last buy: 1 pt
    
    Must also have:
    - Price below 200 EMA
    - 2 red days OR 1 red day with ≥5% drop
    
    Sell Signal (all required):
    - Distance from 200 EMA: >+15% (moderate), >+25% (strong), >+40% (extreme)
    - Inverse V-shape pattern
    - RSI(14) >65
    - Profit ≥15% from average cost
    - Held ≥30 days
    
    Cash Management:
    - 70% Active pool (signals 6-9)
    - 30% Reserve pool (signals 10+)
    """
    
    # Budget parameters
    monthly_contribution = 600.0
    active_cash_pct = 70.0  # 70% for regular signals
    reserve_cash_pct = 30.0  # 30% for extreme signals
    
    # Buy signal thresholds
    min_signal_score = 6
    strong_signal_score = 8
    extreme_signal_score = 10
    
    # Position sizing
    base_size_thresholds = {
        (600, 1200): 10,    # 1/10
        (1200, 2400): 15,   # 1/15
        (2400, 3600): 20,   # 1/20
        (3600, float('inf')): 25  # 1/25
    }
    
    # Size bonuses
    price_drop_bonus_10 = 15.0  # +15% for 10-19% drop
    price_drop_bonus_20 = 25.0  # +25% for 20%+ drop
    strong_signal_multiplier = 1.3
    extreme_signal_multiplier = 1.5
    
    # Buy requirements
    red_days_required = 2
    single_red_day_drop = -5.0  # % drop for 1 red day to count
    
    # Sell signal thresholds
    sell_moderate_ema_pct = 15.0
    sell_strong_ema_pct = 25.0
    sell_extreme_ema_pct = 40.0
    sell_rsi_threshold = 65
    sell_min_profit_pct = 15.0
    sell_min_hold_days = 30
    
    # Sell amounts
    sell_moderate_pct = 10.0
    sell_strong_pct = 20.0
    sell_extreme_pct = 30.0
    max_sell_pct = 30.0
    min_holdings_pct = 20.0
    
    # Sell frequency limits (days)
    moderate_sell_frequency = 14  # Every 2 weeks
    strong_sell_frequency = 7     # Weekly
    extreme_sell_frequency = 7    # Weekly
    
    # Technical indicators
    ema_period = 200
    rsi_period = 14
    volume_period = 20
    pattern_lookback = 20
    
    def init(self):
        """Initialize indicators and tracking."""
        super().init()
        
        # Technical indicators
        self.ema_200 = self.I(calculate_ema, self.data.Close, self.ema_period)
        self.rsi = self.I(calculate_rsi, self.data.Close, self.rsi_period)
        self.volume_avg = self.I(calculate_volume_avg, self.data.Volume, self.volume_period)
        
        # Cash management
        self.cash_active = self.monthly_contribution * (self.active_cash_pct / 100)
        self.cash_reserve = self.monthly_contribution * (self.reserve_cash_pct / 100)
        self.total_cash_accumulated = self.monthly_contribution
        
        # Position tracking
        self.position_data = CoinPosition(
            symbol=self.data.symbol if hasattr(self.data, 'symbol') else 'CRYPTO'
        )
        
        # Tracking
        self.current_month = None
        self.last_sell_date = None
        self.buy_count_since_reset = 0
        self.max_buys_before_reset = 5
        
        # Stop buying flag
        self.stop_buying_until_ema = False
    
    def next(self):
        """Execute strategy logic."""
        if not self.should_trade():
            return
        
        current_date = self.data.index[-1]
        current_price = self.data.Close[-1]
        current_month = (current_date.year, current_date.month)
        
        # Add monthly contribution
        if self.current_month != current_month:
            self._add_monthly_contribution()
            self.current_month = current_month
        
        # Check sell conditions
        if self.position.size > 0:
            self._check_sell_signal(current_date, current_price)
        
        # Check buy conditions
        if not self.stop_buying_until_ema:
            self._check_buy_signal(current_date, current_price)
        else:
            # Check if we should resume buying
            ema_200_value = self.ema_200[-1]
            distance_from_ema = ((current_price - ema_200_value) / ema_200_value) * 100 if ema_200_value > 0 else 0
            if abs(distance_from_ema) <= 5.0:  # Within 5% of 200 EMA
                self.stop_buying_until_ema = False
    
    def _add_monthly_contribution(self):
        """Add monthly contribution to cash pools."""
        active_portion = self.monthly_contribution * (self.active_cash_pct / 100)
        reserve_portion = self.monthly_contribution * (self.reserve_cash_pct / 100)
        
        self.cash_active += active_portion
        self.cash_reserve += reserve_portion
        self.total_cash_accumulated += self.monthly_contribution
    
    def _calculate_buy_signal_score(self, current_price: float) -> Tuple[int, Dict[str, int]]:
        """
        Calculate buy signal score (0-10+ points).
        
        Returns:
            Tuple of (total_score, score_breakdown)
        """
        score = 0
        breakdown = {}
        
        ema_200_value = self.ema_200[-1]
        rsi_value = self.rsi[-1]
        current_volume = self.data.Volume[-1]
        avg_volume = self.volume_avg[-1]
        
        # 1. Distance from 200 EMA (1-3 points)
        if ema_200_value > 0:
            distance_pct = ((current_price - ema_200_value) / ema_200_value) * 100
            
            if distance_pct <= -18:
                ema_points = 3
            elif distance_pct <= -12:
                ema_points = 2
            elif distance_pct <= -8:
                ema_points = 1
            else:
                ema_points = 0
            
            score += ema_points
            breakdown['ema_distance'] = ema_points
        
        # 2. RSI (1-2 points)
        if rsi_value < 35:
            rsi_points = 2
        elif rsi_value < 45:
            rsi_points = 1
        else:
            rsi_points = 0
        
        score += rsi_points
        breakdown['rsi'] = rsi_points
        
        # 3. V-shape pattern (2 points)
        if is_v_shape_bottom(self.data.Close, self.pattern_lookback):
            score += 2
            breakdown['v_shape'] = 2
        else:
            breakdown['v_shape'] = 0
        
        # 4. Acceleration (1 point)
        acceleration = calculate_price_acceleration(self.data.Close, days=2)
        if acceleration <= -3.0:  # Worsened ≥3% in 2 days
            score += 1
            breakdown['acceleration'] = 1
        else:
            breakdown['acceleration'] = 0
        
        # 5. Volume spike (1 point)
        if avg_volume > 0 and current_volume > avg_volume * 1.5:
            score += 1
            breakdown['volume_spike'] = 1
        else:
            breakdown['volume_spike'] = 0
        
        # 6. Price vs last buy (1 point)
        if self.position_data.last_buy_price > 0:
            drop_from_last_buy = ((current_price - self.position_data.last_buy_price) / self.position_data.last_buy_price) * 100
            if drop_from_last_buy <= -10:
                score += 1
                breakdown['price_vs_last_buy'] = 1
            else:
                breakdown['price_vs_last_buy'] = 0
        else:
            breakdown['price_vs_last_buy'] = 0
        
        return score, breakdown
    
    def _check_buy_signal(self, current_date: datetime, current_price: float):
        """Check if buy signal is present and execute if conditions met."""
        # Calculate signal score
        signal_score, breakdown = self._calculate_buy_signal_score(current_price)
        
        # Check minimum score requirement
        if signal_score < self.min_signal_score:
            return
        
        # Check required conditions
        ema_200_value = self.ema_200[-1]
        
        # Must be below 200 EMA
        if current_price > ema_200_value:
            return
        
        # Check red day requirements
        red_days = count_red_days(self.data.Close)
        
        # Check single day drop
        single_day_drop = 0.0
        if len(self.data.Close) >= 2:
            single_day_drop = ((self.data.Close[-1] - self.data.Close[-2]) / self.data.Close[-2]) * 100
        
        red_day_condition_met = (red_days >= self.red_days_required) or (red_days >= 1 and single_day_drop <= self.single_red_day_drop)
        
        if not red_day_condition_met:
            return
        
        # Determine if this is an extreme signal
        is_extreme_signal = signal_score >= self.extreme_signal_score
        
        # Check if we have sufficient cash
        if is_extreme_signal:
            available_cash = self.cash_active + self.cash_reserve
        else:
            available_cash = self.cash_active
        
        if available_cash < 50:  # Minimum buy amount
            return
        
        # Calculate position size
        buy_amount_usd = self._calculate_position_size(signal_score, current_price, available_cash)
        
        if buy_amount_usd > 0:
            # Calculate number of coins
            coins_to_buy = buy_amount_usd / current_price
            
            # Execute buy
            self.buy(size=coins_to_buy)
            
            # Update cash pools
            if is_extreme_signal:
                # Use from both pools
                if buy_amount_usd <= self.cash_active:
                    self.cash_active -= buy_amount_usd
                else:
                    remaining = buy_amount_usd - self.cash_active
                    self.cash_active = 0
                    self.cash_reserve -= remaining
            else:
                # Use from active pool only
                self.cash_active -= buy_amount_usd
            
            # Update tracking
            self.position_data.holdings += coins_to_buy
            self.position_data.total_invested += buy_amount_usd
            self.position_data.buy_count += 1
            self.position_data.last_buy_price = current_price
            self.position_data.last_buy_date = current_date
            self.buy_count_since_reset += 1
            
            # Check if we need to reset position sizing
            if self.buy_count_since_reset >= self.max_buys_before_reset:
                self.buy_count_since_reset = 0
    
    def _calculate_position_size(self, signal_score: int, current_price: float, available_cash: float) -> float:
        """
        Calculate position size based on signal strength and cash available.
        
        Returns:
            Dollar amount to invest
        """
        # Determine base divisor based on accumulated cash
        base_divisor = 20  # Default
        
        for (min_cash, max_cash), divisor in self.base_size_thresholds.items():
            if min_cash <= self.total_cash_accumulated < max_cash:
                base_divisor = divisor
                break
        
        # Get base allocation for this coin (would come from allocation ratios in multi-asset)
        # For single asset testing, use full available cash
        coin_allocation = available_cash
        
        # Calculate base size
        base_size = coin_allocation / base_divisor
        
        # Apply size bonuses for price drops
        size_multiplier = 1.0
        
        if self.position_data.last_buy_price > 0:
            price_drop_pct = ((current_price - self.position_data.last_buy_price) / self.position_data.last_buy_price) * 100
            
            if price_drop_pct <= -20:
                size_multiplier += (self.price_drop_bonus_20 / 100)
            elif price_drop_pct <= -10:
                size_multiplier += (self.price_drop_bonus_10 / 100)
        
        # Apply signal score multipliers
        if signal_score >= self.extreme_signal_score:
            size_multiplier *= self.extreme_signal_multiplier
        elif signal_score >= self.strong_signal_score:
            size_multiplier *= self.strong_signal_multiplier
        
        # Calculate final size
        final_size = base_size * size_multiplier
        
        # Cap at available cash
        final_size = min(final_size, available_cash)
        
        return final_size
    
    def _check_sell_signal(self, current_date: datetime, current_price: float):
        """Check if sell conditions are met."""
        if self.position.size == 0:
            return
        
        # Check holding period (≥30 days)
        if self.position_data.last_buy_date:
            days_held = (current_date - self.position_data.last_buy_date).days
            if days_held < self.sell_min_hold_days:
                return
        
        # Get technical values
        ema_200_value = self.ema_200[-1]
        rsi_value = self.rsi[-1]
        
        # Check distance from 200 EMA
        if ema_200_value > 0:
            ema_distance_pct = ((current_price - ema_200_value) / ema_200_value) * 100
        else:
            return
        
        # Determine overbought level
        if ema_distance_pct > self.sell_extreme_ema_pct:
            overbought_level = 'extreme'
            sell_pct = self.sell_extreme_pct
            min_frequency_days = self.extreme_sell_frequency
        elif ema_distance_pct > self.sell_strong_ema_pct:
            overbought_level = 'strong'
            sell_pct = self.sell_strong_pct
            min_frequency_days = self.strong_sell_frequency
        elif ema_distance_pct > self.sell_moderate_ema_pct:
            overbought_level = 'moderate'
            sell_pct = self.sell_moderate_pct
            min_frequency_days = self.moderate_sell_frequency
        else:
            return  # Not overbought enough
        
        # Check sell frequency
        if self.last_sell_date:
            days_since_sell = (current_date - self.last_sell_date).days
            if days_since_sell < min_frequency_days:
                return
        
        # Check inverse V-shape
        if not is_inverse_v_shape_top(self.data.Close, self.pattern_lookback):
            return
        
        # Check RSI
        if rsi_value <= self.sell_rsi_threshold:
            return
        
        # Check profit from average cost
        avg_cost = self.position_data.average_cost
        if avg_cost > 0:
            profit_pct = ((current_price - avg_cost) / avg_cost) * 100
            if profit_pct < self.sell_min_profit_pct:
                return
        else:
            return
        
        # All conditions met - execute sell
        current_holdings = self.position.size
        sell_amount = current_holdings * (sell_pct / 100)
        
        # Ensure we keep minimum holdings
        min_holdings = self.position_data.holdings * (self.min_holdings_pct / 100)
        max_sell_amount = current_holdings - min_holdings
        
        if max_sell_amount > 0:
            sell_amount = min(sell_amount, max_sell_amount)
            
            # Never sell more than 30% at once
            absolute_max = current_holdings * (self.max_sell_pct / 100)
            sell_amount = min(sell_amount, absolute_max)
            
            # Execute sell
            sell_fraction = sell_amount / current_holdings
            self.position.close(sell_fraction)
            
            # Update tracking
            sell_proceeds = sell_amount * current_price
            
            # Return proceeds to cash pool (70/30 split)
            self.cash_active += sell_proceeds * (self.active_cash_pct / 100)
            self.cash_reserve += sell_proceeds * (self.reserve_cash_pct / 100)
            
            self.position_data.holdings -= sell_amount
            self.last_sell_date = current_date
            
            # Set stop buying flag until within 5% of 200 EMA
            self.stop_buying_until_ema = True


# =====================================================================
# STRATEGY REGISTRY
# =====================================================================

DCA_STRATEGY_REGISTRY = {
    'dca_monthly': DCAMonthlyStrategy,
    'dca_signal': DCASignalStrategy,
}


def get_dca_strategy(name: str) -> type:
    """Get DCA strategy class by name."""
    if name not in DCA_STRATEGY_REGISTRY:
        raise ValueError(f"Unknown DCA strategy: {name}. Available: {list(DCA_STRATEGY_REGISTRY.keys())}")
    return DCA_STRATEGY_REGISTRY[name]


def list_dca_strategies() -> List[str]:
    """List all available DCA strategies."""
    return list(DCA_STRATEGY_REGISTRY.keys())
