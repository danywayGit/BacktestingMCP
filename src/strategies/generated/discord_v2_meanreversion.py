"""
Discord v2.x Mean Reversion Strategy (Deduced from signal analysis)

Pattern: RSI oversold/overbought + Bollinger Band position + strong ADX + SuperTrend.
- LONG when RSI < 40, price near lower BB, ADX > 25, above SuperTrend
- SHORT when RSI > 60, price near upper BB, ADX > 25
Works across 45M, 2H, 8H timeframes.
"""

import pandas as pd
import numpy as np
from backtesting import Strategy

from ...core.backtesting_engine import BaseStrategy
import ta


def calculate_rsi(close_prices, period=14):
    if not isinstance(close_prices, pd.Series):
        close_prices = pd.Series(close_prices)
    return ta.momentum.rsi(close_prices, window=period).ffill().bfill().fillna(50)


def calculate_bb_pct(close_prices, period=20, std=2):
    if not isinstance(close_prices, pd.Series):
        close_prices = pd.Series(close_prices)
    bb = ta.volatility.BollingerBands(close_prices, window=period, window_dev=std)
    return bb.bollinger_pband().ffill().bfill().fillna(0.5)


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


class DiscordV2MeanReversionStrategy(BaseStrategy):
    """Mean reversion with RSI + Bollinger Bands + ADX filter."""

    # Optimizable parameters
    rsi_period = 14
    rsi_long_threshold = 40
    rsi_short_threshold = 60
    bb_period = 20
    bb_std = 2
    bb_long_threshold = 0.30   # Below 30% of BB width
    bb_short_threshold = 0.70  # Above 70% of BB width
    adx_period = 14
    adx_threshold = 25
    supertrend_period = 14
    supertrend_multiplier = 3

    def init(self):
        self.rsi = self.I(calculate_rsi, self.data.Close, self.rsi_period)
        self.bb_pct = self.I(calculate_bb_pct, self.data.Close, self.bb_period, self.bb_std)
        self.adx = self.I(calculate_adx, self.data.High, self.data.Low, self.data.Close, self.adx_period)
        self.supertrend_lower = self.I(
            calculate_supertrend_lower,
            self.data.High, self.data.Low, self.data.Close,
            self.supertrend_period, self.supertrend_multiplier
        )

    def next(self):
        if not self.should_trade():
            return

        price = self.data.Close[-1]
        rsi = self.rsi[-1]
        bb = self.bb_pct[-1]
        adx = self.adx[-1]
        above_st = price > self.supertrend_lower[-1]
        strong_trend = adx > self.adx_threshold

        if not self.position:
            # Long: RSI oversold + near lower BB + strong trend + above SuperTrend
            if (rsi < self.rsi_long_threshold
                    and bb < self.bb_long_threshold
                    and strong_trend
                    and above_st):
                self.enter_long_position()

            # Short: RSI overbought + near upper BB + strong trend
            elif (rsi > self.rsi_short_threshold
                  and bb > self.bb_short_threshold
                  and strong_trend):
                self.enter_short_position()

        else:
            # Exit long when RSI recovers above 50 and BB position normalizes
            if self.position.is_long and rsi > 55 and bb > 0.5:
                self.position.close()
            # Exit short when RSI drops below 50 and BB position normalizes
            elif self.position.is_short and rsi < 45 and bb < 0.5:
                self.position.close()
