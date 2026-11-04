"""
Discord v3.x Trend Long-Only Strategy (Deduced from signal analysis)

Pattern: EMA alignment + Bollinger Band dip + ADX trend. LONG ONLY.
- LONG when EMA(21) > EMA(50), price near lower BB, ADX > 25, above SuperTrend
- v3.x was 100% LONG across all signals (no shorts)
Works across 45M, 1H, 2H timeframes.
"""

import pandas as pd
import numpy as np
from backtesting import Strategy

from ...core.backtesting_engine import BaseStrategy
import ta


def calculate_ema(close_prices, period=21):
    if not isinstance(close_prices, pd.Series):
        close_prices = pd.Series(close_prices)
    return ta.trend.ema_indicator(close_prices, window=period).ffill().bfill()


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


class DiscordV3TrendLongStrategy(BaseStrategy):
    """Long-only trend pullback strategy with EMA alignment + BB dip."""

    # Optimizable parameters
    ema_fast = 21
    ema_slow = 50
    rsi_period = 14
    rsi_max = 42          # Median RSI ~39 at entry, cap at ~42
    bb_period = 20
    bb_std = 2
    bb_threshold = 0.30   # Price near lower BB (median was 0.20-0.27)
    adx_period = 14
    adx_threshold = 25
    supertrend_period = 14
    supertrend_multiplier = 3
    rsi_exit = 60         # Exit when RSI recovers

    def init(self):
        self.ema_fast_line = self.I(calculate_ema, self.data.Close, self.ema_fast)
        self.ema_slow_line = self.I(calculate_ema, self.data.Close, self.ema_slow)
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
        uptrend = self.ema_fast_line[-1] > self.ema_slow_line[-1]
        rsi_low = self.rsi[-1] < self.rsi_max
        bb_low = self.bb_pct[-1] < self.bb_threshold
        strong_trend = self.adx[-1] > self.adx_threshold
        above_st = price > self.supertrend_lower[-1]

        if not self.position:
            # Long only: uptrend + RSI dip + near lower BB + strong ADX + above SuperTrend
            if uptrend and rsi_low and bb_low and strong_trend and above_st:
                self.enter_long_position()

        else:
            # Exit when RSI recovers or trend breaks
            if self.rsi[-1] > self.rsi_exit or not uptrend:
                self.position.close()
