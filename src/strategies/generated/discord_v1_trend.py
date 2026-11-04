"""
Discord v1.x Trend Strategy (Deduced from signal analysis)

Pattern: MACD crossover + strong ADX trend + SuperTrend confirmation.
- LONG when MACD > Signal and ADX strong and price above SuperTrend
- SHORT when MACD < Signal and ADX strong and price below SuperTrend
Works across 45M, 1H, 2H, 3H, 8H timeframes.
"""

import pandas as pd
import numpy as np
from backtesting import Strategy
from backtesting.lib import crossover

from ...core.backtesting_engine import BaseStrategy
import ta


def calculate_macd_line(close_prices, fast=12, slow=26):
    if not isinstance(close_prices, pd.Series):
        close_prices = pd.Series(close_prices)
    macd = ta.trend.MACD(close_prices, window_fast=fast, window_slow=slow, window_sign=9)
    return macd.macd().ffill().bfill().fillna(0)


def calculate_macd_signal(close_prices, fast=12, slow=26, signal=9):
    if not isinstance(close_prices, pd.Series):
        close_prices = pd.Series(close_prices)
    macd = ta.trend.MACD(close_prices, window_fast=fast, window_slow=slow, window_sign=signal)
    return macd.macd_signal().ffill().bfill().fillna(0)


def calculate_adx(high, low, close, period=14):
    if not isinstance(close, pd.Series):
        high, low, close = pd.Series(high), pd.Series(low), pd.Series(close)
    adx = ta.trend.ADXIndicator(high, low, close, window=period)
    return adx.adx().ffill().bfill().fillna(20)


def calculate_supertrend_lower(high, low, close, period=14, multiplier=3):
    if not isinstance(close, pd.Series):
        high, low, close = pd.Series(high), pd.Series(low), pd.Series(close)
    atr = ta.volatility.average_true_range(high, low, close, window=period).ffill().bfill().fillna(0)
    hl2 = (high + low) / 2
    return (hl2 - multiplier * atr).ffill().bfill()


class DiscordV1TrendStrategy(BaseStrategy):
    """Trend-following strategy based on MACD + ADX + SuperTrend."""

    # Optimizable parameters
    macd_fast = 12
    macd_slow = 26
    macd_signal_period = 9
    adx_period = 14
    adx_threshold = 25
    supertrend_period = 14
    supertrend_multiplier = 3

    def init(self):
        self.macd_line = self.I(
            calculate_macd_line, self.data.Close,
            self.macd_fast, self.macd_slow
        )
        self.macd_signal = self.I(
            calculate_macd_signal, self.data.Close,
            self.macd_fast, self.macd_slow, self.macd_signal_period
        )
        self.adx = self.I(
            calculate_adx, self.data.High, self.data.Low, self.data.Close,
            self.adx_period
        )
        self.supertrend_lower = self.I(
            calculate_supertrend_lower,
            self.data.High, self.data.Low, self.data.Close,
            self.supertrend_period, self.supertrend_multiplier
        )

    def next(self):
        if not self.should_trade():
            return

        price = self.data.Close[-1]
        macd_bullish = self.macd_line[-1] > self.macd_signal[-1]
        strong_trend = self.adx[-1] > self.adx_threshold
        above_supertrend = price > self.supertrend_lower[-1]

        if not self.position:
            if macd_bullish and strong_trend and above_supertrend:
                self.enter_long_position()
            elif not macd_bullish and strong_trend and not above_supertrend:
                self.enter_short_position()
        else:
            # Exit long when MACD crosses below signal
            if self.position.is_long and not macd_bullish:
                self.position.close()
            # Exit short when MACD crosses above signal
            elif self.position.is_short and macd_bullish:
                self.position.close()
