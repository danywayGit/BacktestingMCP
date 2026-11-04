"""
Telegram Trend Momentum Strategy (Deduced from signal analysis)

Pattern found consistently across all 8 Telegram strategies with HIGH confidence:
- LONG: RSI > 70, EMA(9) > EMA(21) > EMA(50), MACD > Signal, ADX > 25,
        price above SuperTrend, above McGinley Dynamic
- SHORT: RSI < 35, EMA(9) < EMA(21) < EMA(50), MACD < Signal, ADX > 25,
         price below McGinley Dynamic

This is a strong momentum strategy - enters AFTER trend is confirmed (not mean reversion).
All Telegram strategies (ADA SuperF, BTC Trendhoo, DOGE Liquidity Spec, McGinley Trend,
NEAR Trendhoo, Precision Trend Mastery, Stiff Zone) showed this same pattern.
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


def calculate_ema(close_prices, period=9):
    if not isinstance(close_prices, pd.Series):
        close_prices = pd.Series(close_prices)
    return ta.trend.ema_indicator(close_prices, window=period).ffill().bfill()


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


def calculate_mcginley(close_prices, period=10):
    """McGinley Dynamic indicator - smoothed adaptive moving average."""
    if not isinstance(close_prices, pd.Series):
        close_prices = pd.Series(close_prices)
    mcg = close_prices.copy().astype(float)
    for i in range(1, len(close_prices)):
        prev = mcg.iloc[i - 1]
        cur = close_prices.iloc[i]
        if pd.notna(prev) and prev > 0:
            mcg.iloc[i] = prev + (cur - prev) / (period * (cur / prev) ** 4)
    return mcg.ffill().bfill()


class TelegramTrendMomentumStrategy(BaseStrategy):
    """Momentum trend strategy: enter after full EMA+MACD+RSI confirmation."""

    # Optimizable parameters
    rsi_period = 14
    rsi_long_threshold = 65    # Median was ~70-74, enter when momentum building
    rsi_short_threshold = 35   # Median was ~28-33
    ema_fast = 9
    ema_mid = 21
    ema_slow = 50
    macd_fast = 12
    macd_slow = 26
    macd_signal_period = 9
    adx_period = 14
    adx_threshold = 25
    supertrend_period = 14
    supertrend_multiplier = 3
    mcginley_period = 10
    rsi_exit_long = 45    # Exit long when momentum fades
    rsi_exit_short = 55   # Exit short when momentum fades

    def init(self):
        self.rsi = self.I(calculate_rsi, self.data.Close, self.rsi_period)
        self.ema_fast_line = self.I(calculate_ema, self.data.Close, self.ema_fast)
        self.ema_mid_line = self.I(calculate_ema, self.data.Close, self.ema_mid)
        self.ema_slow_line = self.I(calculate_ema, self.data.Close, self.ema_slow)
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
        self.mcginley = self.I(calculate_mcginley, self.data.Close, self.mcginley_period)

    def next(self):
        if not self.should_trade():
            return

        price = self.data.Close[-1]
        rsi = self.rsi[-1]
        ema_cascade_bull = (self.ema_fast_line[-1] > self.ema_mid_line[-1]
                           > self.ema_slow_line[-1])
        ema_cascade_bear = (self.ema_fast_line[-1] < self.ema_mid_line[-1]
                           < self.ema_slow_line[-1])
        macd_bull = self.macd_line[-1] > self.macd_signal[-1]
        strong_trend = self.adx[-1] > self.adx_threshold
        above_st = price > self.supertrend_lower[-1]
        above_mcg = price > self.mcginley[-1]

        if not self.position:
            # LONG: Full bullish alignment
            if (rsi > self.rsi_long_threshold
                    and ema_cascade_bull
                    and macd_bull
                    and strong_trend
                    and above_st
                    and above_mcg):
                self.enter_long_position()

            # SHORT: Full bearish alignment
            elif (rsi < self.rsi_short_threshold
                  and ema_cascade_bear
                  and not macd_bull
                  and strong_trend
                  and not above_mcg):
                self.enter_short_position()

        else:
            # Exit long when momentum fades
            if self.position.is_long:
                if rsi < self.rsi_exit_long or not ema_cascade_bull:
                    self.position.close()
            # Exit short when momentum recovers
            elif self.position.is_short:
                if rsi > self.rsi_exit_short or not ema_cascade_bear:
                    self.position.close()
