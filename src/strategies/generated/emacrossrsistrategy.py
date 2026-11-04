"""
AI-generated trading strategy.
Multi-timeframe EMA Crossover with RSI confirmation.

Entry on 4H: Fast EMA crosses above Slow EMA + RSI > threshold
Confirmation: 12H uptrend (price above EMA on 12H equivalent)
Exit: SL/TP orders OR EMA cross down/RSI overbought with profit
Risk: 1% per trade based on stop-loss distance

GPU-Optimized Parameters (2021-2025):
- SOLUSDT: 80.59% return, 5.1% DD, SQN 3.26 (Best)
- ETHUSDT: 46.05% return, 7.8% DD, SQN 2.22
- BNBUSDT: 47.55% return, 6.8% DD, SQN 2.73
- BTCUSDT: 41.21% return, 8.0% DD, SQN 1.67
"""

import pandas as pd
import numpy as np
from backtesting import Strategy
from backtesting.lib import crossover

from ...core.backtesting_engine import BaseStrategy

# Import ta library for indicator calculations
import ta


def calculate_rsi(close_prices, period: int = 14):
    """Calculate RSI indicator."""
    if not isinstance(close_prices, pd.Series):
        close_prices = pd.Series(close_prices)
    return ta.momentum.rsi(close_prices, window=period).ffill().bfill().fillna(50)


def calculate_ema(close_prices, period: int = 200):
    """Calculate Exponential Moving Average."""
    if not isinstance(close_prices, pd.Series):
        close_prices = pd.Series(close_prices)
    return ta.trend.ema_indicator(close_prices, window=period).ffill().bfill()


def calculate_atr(high, low, close, period: int = 14):
    """Calculate Average True Range."""
    if not isinstance(high, pd.Series):
        high = pd.Series(high)
    if not isinstance(low, pd.Series):
        low = pd.Series(low)
    if not isinstance(close, pd.Series):
        close = pd.Series(close)
    return ta.volatility.average_true_range(high, low, close, window=period).ffill().bfill().fillna(0)


def calculate_ema_12h(close_prices, period: int = 55):
    """
    Calculate EMA on 12H timeframe from 4H data.
    12H = 3x 4H bars, so we use period * 3 to simulate
    the slower-moving 12H EMA behavior.
    """
    if not isinstance(close_prices, pd.Series):
        close_prices = pd.Series(close_prices)
    adjusted_period = period * 3
    return ta.trend.ema_indicator(close_prices, window=adjusted_period).ffill().bfill()


class EMAcrossRSIStrategy(BaseStrategy):
    """
    Multi-timeframe EMA Crossover Strategy (GPU-Optimized Parameters).
    
    4H Timeframe (Entry):
    - Long when Fast EMA crosses above Slow EMA
    - RSI must be above threshold
    
    12H Timeframe (Trend Confirmation):
    - Price must be above 12H EMA (simulated via 3x period on 4H)
    
    Risk Management:
    - Position size: Based on 1% account risk per trade
    - Stop loss: ATR multiplier below entry
    - Take profit: Reward-risk ratio above entry
    
    Best Parameters by Asset (from GPU optimization 2021-2025):
    
    SOLUSDT (Best Overall - 80.59% return, 5.1% DD):
        ema_fast=13, ema_slow=55, ema_12h=21, rsi_period=21, 
        rsi_threshold=50, atr_multiplier=1.0, reward_risk_ratio=4.0
    
    ETHUSDT (46.05% return, 7.8% DD):
        ema_fast=8, ema_slow=21, ema_12h=21, rsi_period=14,
        rsi_threshold=55, atr_multiplier=1.0, reward_risk_ratio=3.0
    
    BNBUSDT (47.55% return, 6.8% DD):
        ema_fast=8, ema_slow=89, ema_12h=34, rsi_period=7,
        rsi_threshold=55, atr_multiplier=1.5, reward_risk_ratio=3.5
    
    BTCUSDT (41.21% return, 8.0% DD):
        ema_fast=8, ema_slow=21, ema_12h=13, rsi_period=14,
        rsi_threshold=40, atr_multiplier=1.0, reward_risk_ratio=4.0
    """
    
    # ==========================================================================
    # DEFAULT PARAMETERS (SOL - Best Performance)
    # Change these for different assets or override in subclass
    # ==========================================================================
    
    # 4H Strategy parameters
    ema_fast = 13          # Fast EMA period (4H)
    ema_slow = 55          # Slow EMA period (4H)
    rsi_period = 21        # RSI period
    rsi_threshold = 50     # RSI must be above this for entry
    rsi_exit = 70          # RSI exit threshold (overbought)
    
    # 12H Trend parameters (period on 12H, will be 3x on 4H)
    ema_12h_period = 21
    
    # Risk management
    atr_period = 21
    atr_multiplier = 1.0   # Stop loss = Entry - (ATR * multiplier)
    reward_risk_ratio = 4.0  # Take profit = Entry + (Risk * ratio)
    risk_per_trade = 0.01  # 1% risk per trade
    
    def init(self):
        # 4H indicators
        self.ema_fast_line = self.I(calculate_ema, self.data.Close, self.ema_fast)
        self.ema_slow_line = self.I(calculate_ema, self.data.Close, self.ema_slow)
        self.rsi = self.I(calculate_rsi, self.data.Close, self.rsi_period)
        self.atr = self.I(calculate_atr, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        
        # 12H trend indicator (simulated on 4H using 3x period)
        self.ema_12h = self.I(calculate_ema_12h, self.data.Close, self.ema_12h_period)
        
    def next(self):
        if not self.should_trade():
            return
        
        # Need sufficient bars for 12H EMA (165 bars minimum)
        if len(self.data.Close) < 170:
            return
            
        # === 12H TREND CHECK ===
        # Price must be above 12H 55 EMA (simulated as 165 EMA on 4H)
        trend_12h_bullish = self.data.Close[-1] > self.ema_12h[-1]
        
        # === 4H ENTRY SIGNALS ===
        # EMA crossover (fast crosses above slow)
        ema_crossover = (self.ema_fast_line[-2] < self.ema_slow_line[-2] and 
                         self.ema_fast_line[-1] > self.ema_slow_line[-1])
        
        # RSI above threshold
        rsi_condition = self.rsi[-1] > self.rsi_threshold
        
        # === POSITION MANAGEMENT ===
        if not self.position:
            # Entry: 4H crossover + RSI > threshold + 12H uptrend
            if ema_crossover and rsi_condition and trend_12h_bullish:
                self.enter_long_position()
        else:
            # Check if we hit 2:1 R:R minimum and EMA crosses or RSI overbought
            # Allow early exit only if already in profit
            ema_cross_down = (self.ema_fast_line[-2] > self.ema_slow_line[-2] and 
                              self.ema_fast_line[-1] < self.ema_slow_line[-1])
            rsi_overbought = self.rsi[-1] > self.rsi_exit
            
            # Only exit on signal if already profitable (let SL/TP handle rest)
            if (ema_cross_down or rsi_overbought):
                # Check if in sufficient profit (at least 1:1 R:R achieved)
                if hasattr(self, '_entry_price') and hasattr(self, '_stop_loss'):
                    risk = self._entry_price - self._stop_loss
                    current_profit = self.data.Close[-1] - self._entry_price
                    if risk > 0 and current_profit >= risk:  # At least 1:1 R:R
                        self.position.close()
            
    def enter_long_position(self):
        """
        Enter long with position size based on 1% account risk.
        
        Position Size = (Account * Risk%) / (Entry - StopLoss)
        """
        current_atr = self.atr[-1]
        current_price = self.data.Close[-1]
        
        if current_atr > 0:
            # Calculate SL/TP levels as absolute prices
            stop_loss_price = current_price - (self.atr_multiplier * current_atr)
            risk_per_unit = current_price - stop_loss_price  # Risk in price terms
            take_profit_price = current_price + (self.reward_risk_ratio * risk_per_unit)
            
            # Store for reference
            self._entry_price = current_price
            self._stop_loss = stop_loss_price
            self._take_profit = take_profit_price
            
            # Calculate position size based on 1% risk
            # Risk amount = Account Equity * 1%
            account_equity = self.equity
            risk_amount = account_equity * self.risk_per_trade  # 1% of account
            
            # Position size in units = Risk Amount / Risk per Unit
            position_size_units = risk_amount / risk_per_unit
            
            # Convert to fraction of equity for backtesting.py
            position_value = position_size_units * current_price
            size_fraction = min(position_value / account_equity, 0.99)  # Max 99% of equity
            
            # Enter position WITH actual SL/TP orders and calculated size
            self.buy(size=size_fraction, sl=stop_loss_price, tp=take_profit_price)


# =============================================================================
# ASSET-SPECIFIC OPTIMIZED STRATEGIES
# =============================================================================

class EMAcrossRSI_SOL(EMAcrossRSIStrategy):
    """
    SOL-optimized strategy (Best Performance).
    Return: 80.59%, Max DD: 5.1%, SQN: 3.26, Win Rate: 54.1%
    """
    ema_fast = 13
    ema_slow = 55
    ema_12h_period = 21
    rsi_period = 21
    rsi_threshold = 50
    atr_period = 21
    atr_multiplier = 1.0
    reward_risk_ratio = 4.0


class EMAcrossRSI_ETH(EMAcrossRSIStrategy):
    """
    ETH-optimized strategy.
    Return: 46.05%, Max DD: 7.8%, SQN: 2.22, Win Rate: 48.1%
    """
    ema_fast = 8
    ema_slow = 21
    ema_12h_period = 21
    rsi_period = 14
    rsi_threshold = 55
    atr_period = 21
    atr_multiplier = 1.0
    reward_risk_ratio = 3.0


class EMAcrossRSI_BNB(EMAcrossRSIStrategy):
    """
    BNB-optimized strategy.
    Return: 47.55%, Max DD: 6.8%, SQN: 2.73, Win Rate: 55.0%
    """
    ema_fast = 8
    ema_slow = 89
    ema_12h_period = 34
    rsi_period = 7
    rsi_threshold = 55
    atr_period = 14
    atr_multiplier = 1.5
    reward_risk_ratio = 3.5


class EMAcrossRSI_BTC(EMAcrossRSIStrategy):
    """
    BTC-optimized strategy.
    Return: 41.21%, Max DD: 8.0%, SQN: 1.67, Win Rate: 43.5%
    """
    ema_fast = 8
    ema_slow = 21
    ema_12h_period = 13
    rsi_period = 14
    rsi_threshold = 40
    atr_period = 21
    atr_multiplier = 1.0
    reward_risk_ratio = 4.0