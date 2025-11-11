"""
Example trading strategy templates for the backtesting system.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from backtesting import Strategy
from backtesting.lib import crossover

# Use 'ta' library instead of talib for easier installation
try:
    import talib
    HAS_TALIB = True
except ImportError:
    import ta
    HAS_TALIB = False

from ..core.backtesting_engine import BaseStrategy
from .dca_strategies import DCAMonthlyStrategy, DCASignalStrategy


def calculate_rsi(close_prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI using ta library."""
    if HAS_TALIB:
        return talib.RSI(close_prices, timeperiod=period)
    else:
        result = ta.momentum.rsi(close_prices, window=period)
        # Handle NaN values by forward filling, then backward filling
        result = result.ffill().bfill().fillna(50)
        return result


def calculate_sma(close_prices: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average using ta library."""
    if HAS_TALIB:
        return talib.SMA(close_prices, timeperiod=period)
    else:
        result = ta.trend.sma_indicator(close_prices, window=period)
        # Forward fill NaN values, then use the first available value
        result = result.ffill().bfill()
        if result.isna().all():
            result = result.fillna(close_prices.mean())
        return result


def calculate_bbands(close_prices: pd.Series, period: int = 20, std: int = 2):
    """Calculate Bollinger Bands using ta library."""
    if HAS_TALIB:
        upper, middle, lower = talib.BBANDS(close_prices, timeperiod=period, nbdevup=std, nbdevdn=std)
        return upper, middle, lower
    else:
        bb = ta.volatility.BollingerBands(close_prices, window=period, window_dev=std)
        upper = bb.bollinger_hband().ffill()
        middle = bb.bollinger_mavg().ffill() 
        lower = bb.bollinger_lband().ffill()
        return upper, middle, lower


def calculate_macd(close_prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Calculate MACD using ta library."""
    if HAS_TALIB:
        macd, macd_signal, macd_hist = talib.MACD(close_prices, fastperiod=fast, slowperiod=slow, signalperiod=signal)
        return macd, macd_signal, macd_hist
    else:
        macd_obj = ta.trend.MACD(close_prices, window_fast=fast, window_slow=slow, window_sign=signal)
        macd = macd_obj.macd().fillna(0)
        macd_signal = macd_obj.macd_signal().fillna(0)
        macd_hist = macd_obj.macd_diff().fillna(0)
        return macd, macd_signal, macd_hist


class RSIMeanReversionStrategy(BaseStrategy):
    """
    RSI Mean Reversion Strategy
    
    Buy when RSI is oversold and price is above moving average
    Sell when RSI is overbought or hits stop/take profit
    """
    
    # Strategy parameters
    rsi_period = 14
    rsi_oversold = 30
    rsi_overbought = 70
    ma_period = 20
    
    def init(self):
        """Initialize indicators."""
        # RSI indicator
        self.rsi = self.I(calculate_rsi, self.data.Close, self.rsi_period)
        
        # Moving Average
        self.ma = self.I(calculate_sma, self.data.Close, self.ma_period)
    
    def next(self):
        """Strategy logic."""
        if not self.should_trade():
            return
        
        current_rsi = self.rsi[-1]
        current_price = self.data.Close[-1]
        current_ma = self.ma[-1]
        
        # Entry conditions
        if not self.position:
            # Long entry: RSI oversold and price above MA
            if current_rsi < self.rsi_oversold and current_price > current_ma:
                self.enter_long_position()
        
        # Exit conditions
        else:
            # Exit long: RSI overbought
            if self.position.is_long and current_rsi > self.rsi_overbought:
                self.position.close()


class MovingAverageCrossoverStrategy(BaseStrategy):
    """
    Moving Average Crossover Strategy
    
    Buy when fast MA crosses above slow MA
    Sell when fast MA crosses below slow MA
    """
    
    # Strategy parameters
    fast_ma_period = 10
    slow_ma_period = 30
    
    def init(self):
        """Initialize indicators."""
        # Fast and slow moving averages
        self.fast_ma = self.I(calculate_sma, self.data.Close, self.fast_ma_period)
        self.slow_ma = self.I(calculate_sma, self.data.Close, self.slow_ma_period)
    
    def next(self):
        """Strategy logic."""
        if not self.should_trade():
            return
        
        # Entry conditions
        if not self.position:
            # Long entry: fast MA crosses above slow MA
            if crossover(self.fast_ma, self.slow_ma):
                self.enter_long_position()
            
            # Short entry: fast MA crosses below slow MA
            elif crossover(self.slow_ma, self.fast_ma):
                self.enter_short_position()


class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands Mean Reversion Strategy
    
    Buy when price touches lower band
    Sell when price touches upper band
    """
    
    # Strategy parameters
    bb_period = 20
    bb_std = 2.0
    
    def init(self):
        """Initialize indicators."""
        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            calculate_bbands, 
            self.data.Close, 
            self.bb_period, 
            self.bb_std
        )
    
    def next(self):
        """Strategy logic."""
        if not self.should_trade():
            return
        
        current_price = self.data.Close[-1]
        current_upper = self.bb_upper[-1]
        current_lower = self.bb_lower[-1]
        current_middle = self.bb_middle[-1]
        
        # Entry conditions
        if not self.position:
            # Long entry: price touches lower band
            if current_price <= current_lower:
                self.enter_long_position()
            
            # Short entry: price touches upper band
            elif current_price >= current_upper:
                self.enter_short_position()
        
        # Exit conditions
        else:
            # Exit long: price reaches middle band
            if self.position.is_long and current_price >= current_middle:
                self.position.close()
            
            # Exit short: price reaches middle band
            elif self.position.is_short and current_price <= current_middle:
                self.position.close()


class MACDStrategy(BaseStrategy):
    """
    MACD Momentum Strategy
    
    Buy when MACD line crosses above signal line
    Sell when MACD line crosses below signal line
    """
    
    # Strategy parameters
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    
    def init(self):
        """Initialize indicators."""
        # MACD indicator
        self.macd_line, self.macd_signal_line, self.macd_histogram = self.I(
            calculate_macd,
            self.data.Close,
            self.macd_fast,
            self.macd_slow,
            self.macd_signal
        )
    
    def next(self):
        """Strategy logic."""
        if not self.should_trade():
            return
        
        # Entry conditions
        if not self.position:
            # Long entry: MACD crosses above signal
            if crossover(self.macd_line, self.macd_signal_line):
                self.enter_long_position()
            
            # Short entry: MACD crosses below signal
            elif crossover(self.macd_signal_line, self.macd_line):
                self.enter_short_position()


class SupportResistanceStrategy(BaseStrategy):
    """
    Support and Resistance Strategy
    
    Identifies support and resistance levels and trades bounces
    """
    
    # Strategy parameters
    lookback_period = 20
    min_touches = 2
    level_tolerance = 0.01  # 1% tolerance for level identification
    
    def init(self):
        """Initialize strategy."""
        self.support_levels = []
        self.resistance_levels = []
        self.last_update_bar = 0
    
    def identify_levels(self):
        """Identify support and resistance levels."""
        if len(self.data) < self.lookback_period + 10:
            return
        
        # Get recent high and low data
        highs = self.data.High[-self.lookback_period:]
        lows = self.data.Low[-self.lookback_period:]
        
        # Find local maxima (resistance) and minima (support)
        resistance_candidates = []
        support_candidates = []
        
        for i in range(1, len(highs) - 1):
            # Local maximum
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                resistance_candidates.append(highs[i])
            
            # Local minimum
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                support_candidates.append(lows[i])
        
        # Group similar levels
        self.resistance_levels = self._group_levels(resistance_candidates)
        self.support_levels = self._group_levels(support_candidates)
    
    def _group_levels(self, levels):
        """Group similar price levels together."""
        if not levels:
            return []
        
        grouped_levels = []
        levels.sort()
        
        current_group = [levels[0]]
        
        for level in levels[1:]:
            # Check if level is within tolerance of current group
            group_avg = sum(current_group) / len(current_group)
            if abs(level - group_avg) / group_avg <= self.level_tolerance:
                current_group.append(level)
            else:
                # Start new group if current group has enough touches
                if len(current_group) >= self.min_touches:
                    grouped_levels.append(sum(current_group) / len(current_group))
                current_group = [level]
        
        # Add last group if it has enough touches
        if len(current_group) >= self.min_touches:
            grouped_levels.append(sum(current_group) / len(current_group))
        
        return grouped_levels
    
    def next(self):
        """Strategy logic."""
        if not self.should_trade():
            return
        
        # Update levels periodically
        if len(self.data) - self.last_update_bar >= 10:
            self.identify_levels()
            self.last_update_bar = len(self.data)
        
        current_price = self.data.Close[-1]
        
        # Entry conditions
        if not self.position and self.support_levels and self.resistance_levels:
            # Find nearest support and resistance
            nearest_support = min(self.support_levels, key=lambda x: abs(x - current_price))
            nearest_resistance = min(self.resistance_levels, key=lambda x: abs(x - current_price))
            
            # Long entry: price near support
            support_distance = abs(current_price - nearest_support) / current_price
            if support_distance <= self.level_tolerance and current_price > nearest_support:
                self.enter_long_position()
            
            # Short entry: price near resistance
            resistance_distance = abs(current_price - nearest_resistance) / current_price
            if resistance_distance <= self.level_tolerance and current_price < nearest_resistance:
                self.enter_short_position()


class MultiTimeframeStrategy(BaseStrategy):
    """
    Multi-timeframe strategy that uses higher timeframe trend
    and lower timeframe entries.
    
    Note: This is a template - actual implementation would require
    additional data feeds for different timeframes.
    """
    
    # Strategy parameters
    trend_ma_period = 50  # For higher timeframe trend
    entry_rsi_period = 14  # For lower timeframe entries
    entry_rsi_oversold = 30
    entry_rsi_overbought = 70
    
    def init(self):
        """Initialize indicators."""
        # Trend indicator (higher timeframe)
        self.trend_ma = self.I(calculate_sma, self.data.Close, self.trend_ma_period)
        
        # Entry indicator (current timeframe)
        self.entry_rsi = self.I(calculate_rsi, self.data.Close, self.entry_rsi_period)
    
    def next(self):
        """Strategy logic."""
        if not self.should_trade():
            return
        
        current_price = self.data.Close[-1]
        current_trend_ma = self.trend_ma[-1]
        current_rsi = self.entry_rsi[-1]
        
        # Determine trend direction
        trend_is_up = current_price > current_trend_ma
        trend_is_down = current_price < current_trend_ma
        
        # Entry conditions
        if not self.position:
            # Long entry: uptrend + oversold RSI
            if trend_is_up and current_rsi < self.entry_rsi_oversold:
                self.enter_long_position()
            
            # Short entry: downtrend + overbought RSI
            elif trend_is_down and current_rsi > self.entry_rsi_overbought:
                self.enter_short_position()


# Strategy registry for easy access
STRATEGY_REGISTRY = {
    'rsi_mean_reversion': RSIMeanReversionStrategy,
    'moving_average_crossover': MovingAverageCrossoverStrategy,
    'bollinger_bands': BollingerBandsStrategy,
    'macd': MACDStrategy,
    'support_resistance': SupportResistanceStrategy,
    'multi_timeframe': MultiTimeframeStrategy,
    'monthly_dca': DCAMonthlyStrategy,
    'signal_based_dca': DCASignalStrategy,
}


def get_strategy_class(strategy_name: str) -> type:
    """Get strategy class by name."""
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(STRATEGY_REGISTRY.keys())}")
    
    return STRATEGY_REGISTRY[strategy_name]


def list_available_strategies() -> List[str]:
    """List all available strategy names."""
    return list(STRATEGY_REGISTRY.keys())


def get_strategy_parameters(strategy_name: str) -> Dict[str, Any]:
    """Get default parameters for a strategy."""
    strategy_class = get_strategy_class(strategy_name)
    
    # Extract class attributes that are likely parameters
    parameters = {}
    for attr_name in dir(strategy_class):
        if not attr_name.startswith('_') and not callable(getattr(strategy_class, attr_name)):
            attr_value = getattr(strategy_class, attr_name)
            if isinstance(attr_value, (int, float, str, bool, list)):
                parameters[attr_name] = attr_value
    
    return parameters
