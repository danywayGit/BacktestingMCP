"""
AI-generated trading strategy.
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


def calculate_sma(close_prices, period: int = 20):
    """Calculate Simple Moving Average."""
    if not isinstance(close_prices, pd.Series):
        close_prices = pd.Series(close_prices)
    return ta.trend.sma_indicator(close_prices, window=period).ffill().bfill()


def calculate_atr(high, low, close, period: int = 14):
    """Calculate Average True Range."""
    if not isinstance(high, pd.Series):
        high = pd.Series(high)
    if not isinstance(low, pd.Series):
        low = pd.Series(low)
    if not isinstance(close, pd.Series):
        close = pd.Series(close)
    return ta.volatility.average_true_range(high, low, close, window=period).ffill().bfill().fillna(0)


def calculate_bbands(close_prices, period: int = 20, std: int = 2):
    """Calculate Bollinger Bands."""
    if not isinstance(close_prices, pd.Series):
        close_prices = pd.Series(close_prices)
    bb = ta.volatility.BollingerBands(close_prices, window=period, window_dev=std)
    return bb.bollinger_hband().ffill(), bb.bollinger_mavg().ffill(), bb.bollinger_lband().ffill()


def calculate_macd(close_prices, fast: int = 12, slow: int = 26, signal: int = 9):
    """Calculate MACD."""
    if not isinstance(close_prices, pd.Series):
        close_prices = pd.Series(close_prices)
    macd_obj = ta.trend.MACD(close_prices, window_fast=fast, window_slow=slow, window_sign=signal)
    return macd_obj.macd().fillna(0), macd_obj.macd_signal().fillna(0), macd_obj.macd_diff().fillna(0)


class BuyAfterPullbackTrendContinuation(BaseStrategy):
    # Strategy parameters (can be optimized)
    ema_period = 200
    confirmation_bars = 90
    wait_bars = 20
    risk_reward_ratio = 2
    
    def init(self):
        self.ema = self.I(calculate_ema, self.data.Close, self.ema_period)
        self.ema_trend = self.I(self._calculate_ema_trend)
        
    def _calculate_ema_trend(self):
        # Track if price has been above EMA for confirmation_bars
        above_ema = self.data.Close > self.ema
        return above_ema
        
    def next(self):
        if not self.should_trade():
            return
            
        # Check if we're in a long trend
        current_above_ema = self.data.Close[-1] > self.ema[-1]
        previous_above_ema = self.data.Close[-2] > self.ema[-2]
        
        # Check if we've been above EMA for 90 bars
        ema_trend_condition = all(self.data.Close[-self.confirmation_bars+i] > self.ema[-self.confirmation_bars+i] 
                                for i in range(self.confirmation_bars))
        
        # Check if we've been below EMA for 20 bars
        below_ema_condition = all(self.data.Close[-self.wait_bars+i] <= self.ema[-self.wait_bars+i] 
                                for i in range(self.wait_bars))
        
        # Check if we're back above EMA
        back_above_ema = current_above_ema and not previous_above_ema
        
        if ema_trend_condition and below_ema_condition and back_above_ema:
            # Find the last swing low in the previous 30 bars
            swing_low = min(self.data.Close[-30:])
            
            # Calculate stop loss below swing low
            stop_loss = swing_low * 0.99  # 1% below swing low
            
            # Calculate take profit based on risk reward ratio of 2
            risk = self.data.Close[-1] - stop_loss
            take_profit = self.data.Close[-1] + (risk * self.risk_reward_ratio)
            
            # Enter long position
            self.enter_long_position(
                size=1,
                stop=stop_loss,
                limit=take_profit
            )