"""
AI-generated trading strategy.
"""

import pandas as pd
import numpy as np
from backtesting import Strategy
from backtesting.lib import crossover

from ...core.backtesting_engine import BaseStrategy

# Import indicator functions
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


class TestRSIStrategy(BaseStrategy):
    # Strategy parameters
    rsi_period = 14
    ema_period = 200
    rsi_oversold = 30
    rsi_overbought = 70
    
    def init(self):
        self.rsi = self.I(calculate_rsi, self.data.Close, self.rsi_period)
        self.ema = self.I(calculate_ema, self.data.Close, self.ema_period)
        
    def next(self):
        if not self.should_trade():
            return
            
        # Buy condition: RSI below 30 and price above 200 EMA
        if not self.position:
            if self.rsi[-1] < self.rsi_oversold and self.data.Close[-1] > self.ema[-1]:
                self.enter_long_position()
        
        # Sell condition: RSI above 70
        else:
            if self.rsi[-1] > self.rsi_overbought:
                self.position.close()