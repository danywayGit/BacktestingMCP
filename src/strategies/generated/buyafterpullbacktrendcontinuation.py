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
    confirmation_bars = 90    # Min bars price must stay above EMA (uptrend phase)
    wait_bars = 20            # Min bars price must stay below EMA (pullback phase)
    risk_reward_ratio = 2
    max_gap_bars = 5          # Max transition gap between phases

    def init(self):
        self.ema = self.I(calculate_ema, self.data.Close, self.ema_period)

    def next(self):
        if not self.should_trade():
            return

        n = len(self.data.Close)
        min_bars = self.confirmation_bars + self.wait_bars + self.max_gap_bars + 2
        if n < min_bars:
            return

        # Step 1: Detect crossover — previous bar below EMA, current bar above
        if not (self.data.Close[-1] > self.ema[-1] and self.data.Close[-2] <= self.ema[-2]):
            return

        # Step 2: Scan backward to find the pullback phase (below EMA)
        # Starting from bar -3 (allowing a small gap after crossover)
        pullback_count = 0
        pullback_end = None    # Most recent bar of the pullback (closest to now)
        pullback_start = None  # Earliest bar of the pullback (furthest from now)

        i = 3  # Start scanning from 3 bars ago (gap tolerance after crossover)
        # Skip a small gap of bars that may be near EMA (transition zone)
        gap_used = 0
        while i < n and gap_used <= self.max_gap_bars:
            if self.data.Close[-i] <= self.ema[-i]:
                break
            gap_used += 1
            i += 1

        if i >= n or gap_used > self.max_gap_bars:
            return

        # Now count consecutive bars below EMA (the pullback)
        pullback_end = i
        while i < n and self.data.Close[-i] <= self.ema[-i]:
            pullback_count += 1
            i += 1
        pullback_start = i - 1

        if pullback_count < self.wait_bars:
            return

        # Step 3: Allow a small transition gap between uptrend and pullback
        gap_used = 0
        while i < n and gap_used <= self.max_gap_bars:
            if self.data.Close[-i] > self.ema[-i]:
                break
            gap_used += 1
            i += 1

        if i >= n or gap_used > self.max_gap_bars:
            return

        # Step 4: Count consecutive bars above EMA (the prior uptrend)
        trend_count = 0
        while i < n and self.data.Close[-i] > self.ema[-i]:
            trend_count += 1
            i += 1

        if trend_count < self.confirmation_bars:
            return

        # All 3 phases confirmed in sequence: uptrend → pullback → crossover back above
        # Find swing low during the pullback for stop loss
        pullback_lows = [self.data.Low[-j] for j in range(pullback_end, pullback_start + 1)]
        swing_low = min(pullback_lows)

        # Stop loss 1% below swing low
        stop_loss = swing_low * 0.99

        # Take profit based on risk-reward ratio
        risk = self.data.Close[-1] - stop_loss
        if risk <= 0:
            return
        take_profit = self.data.Close[-1] + (risk * self.risk_reward_ratio)

        self.enter_long_position(stop_loss=stop_loss, take_profit=take_profit)