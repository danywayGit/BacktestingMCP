"""
Example trading strategy templates for the backtesting system.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
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
from .generated.testrsistrategy import TestRSIStrategy
from .generated.emacrossrsistrategy import EMAcrossRSIStrategy


def calculate_rsi(close_prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI using ta library."""
    close_series = pd.Series(close_prices)
    if HAS_TALIB:
        return talib.RSI(close_series, timeperiod=period)
    else:
        result = ta.momentum.rsi(close_series, window=period)
        # Handle NaN values by forward filling, then backward filling
        result = result.ffill().bfill().fillna(50)
        return result


def calculate_sma(close_prices: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average using ta library."""
    close_series = pd.Series(close_prices)
    if HAS_TALIB:
        return talib.SMA(close_series, timeperiod=period)
    else:
        result = ta.trend.sma_indicator(close_series, window=period)
        # Forward fill NaN values, then use the first available value
        result = result.ffill().bfill()
        if result.isna().all():
            result = result.fillna(close_series.mean())
        return result


def calculate_ema(close_prices: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average using ta library."""
    close_series = pd.Series(close_prices)
    if HAS_TALIB:
        return talib.EMA(close_series, timeperiod=period)
    else:
        result = ta.trend.ema_indicator(close_series, window=period)
        # Forward fill NaN values
        result = result.ffill().bfill()
        if result.isna().all():
            result = result.fillna(close_series.mean())
        return result


def calculate_bbands(close_prices: pd.Series, period: int = 20, std: int = 2):
    """Calculate Bollinger Bands using ta library."""
    close_series = pd.Series(close_prices)
    if HAS_TALIB:
        upper, middle, lower = talib.BBANDS(close_series, timeperiod=period, nbdevup=std, nbdevdn=std)
        return upper, middle, lower
    else:
        bb = ta.volatility.BollingerBands(close_series, window=period, window_dev=std)
        upper = bb.bollinger_hband().ffill()
        middle = bb.bollinger_mavg().ffill() 
        lower = bb.bollinger_lband().ffill()
        return upper, middle, lower


def calculate_macd(close_prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Calculate MACD using ta library."""
    close_series = pd.Series(close_prices)
    if HAS_TALIB:
        macd, macd_signal, macd_hist = talib.MACD(close_series, fastperiod=fast, slowperiod=slow, signalperiod=signal)
        return macd, macd_signal, macd_hist
    else:
        macd_obj = ta.trend.MACD(close_series, window_fast=fast, window_slow=slow, window_sign=signal)
        macd = macd_obj.macd().fillna(0)
        macd_signal = macd_obj.macd_signal().fillna(0)
        macd_hist = macd_obj.macd_diff().fillna(0)
        return macd, macd_signal, macd_hist


def calculate_atr(high_prices: pd.Series, low_prices: pd.Series, close_prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate ATR using ta library."""
    if HAS_TALIB:
        return talib.ATR(high_prices, low_prices, close_prices, timeperiod=period)
    else:
        atr = ta.volatility.average_true_range(
            high=pd.Series(high_prices),
            low=pd.Series(low_prices),
            close=pd.Series(close_prices),
            window=period,
        )
        return atr.ffill().bfill().fillna(0)


def calculate_relative_volume(volumes: pd.Series, lookback: int = 10) -> pd.Series:
    """Current volume divided by rolling average volume."""
    volume_series = pd.Series(volumes)
    avg_volume = volume_series.rolling(lookback, min_periods=1).mean()
    relative_volume = volume_series / avg_volume.replace(0, np.nan)
    return relative_volume.ffill().bfill().fillna(1.0)


def _estimate_horizontal_resistance(high_values: np.ndarray, tolerance: float, min_touches: int) -> Optional[float]:
    """Estimate a horizontal resistance from local highs in a lookback window."""
    if len(high_values) < 5:
        return None

    local_highs = []
    for i in range(1, len(high_values) - 1):
        if high_values[i] > high_values[i - 1] and high_values[i] > high_values[i + 1]:
            local_highs.append(high_values[i])

    if len(local_highs) < min_touches:
        return None

    local_highs = sorted(local_highs)
    groups = [[local_highs[0]]]

    for level in local_highs[1:]:
        group_avg = float(np.mean(groups[-1]))
        if abs(level - group_avg) / max(group_avg, 1e-9) <= tolerance:
            groups[-1].append(level)
        else:
            groups.append([level])

    valid_groups = [g for g in groups if len(g) >= min_touches]
    if not valid_groups:
        return None

    top_group = max(valid_groups, key=lambda g: float(np.mean(g)))
    return float(np.mean(top_group))


class MomentumBreakoutBaseStrategy(BaseStrategy):
    """Shared helpers for long-only breakout strategies with ATR-based exits."""

    atr_length = 14
    atr_multiplier = 2.0
    risk_reward_ratio = 1.5
    trend_ma_period = 50
    breakout_buffer_pct = 0.001

    def init(self):
        self.atr = self.I(calculate_atr, self.data.High, self.data.Low, self.data.Close, self.atr_length)
        self.trend_ma = self.I(calculate_sma, self.data.Close, self.trend_ma_period)

    def _is_bullish_candle(self) -> bool:
        return self.data.Close[-1] > self.data.Open[-1]

    def _is_uptrend(self) -> bool:
        current_price = self.data.Close[-1]
        current_ma = self.trend_ma[-1]
        return np.isfinite(current_ma) and current_price > current_ma

    def _enter_with_atr_risk(self):
        atr = float(self.atr[-1])
        if not np.isfinite(atr) or atr <= 0:
            return

        entry_price = float(self.data.Close[-1])
        stop_loss = entry_price - (self.atr_multiplier * atr)
        if stop_loss >= entry_price:
            return

        take_profit = entry_price + ((entry_price - stop_loss) * self.risk_reward_ratio)
        self.enter_long_position(stop_loss=stop_loss, take_profit=take_profit)


class UnusualVolumeBreakoutStrategy(MomentumBreakoutBaseStrategy):
    """
    Long breakout strategy requiring unusually high relative volume.
    """

    volume_lookback = 10
    volume_multiplier = 2.0
    breakout_lookback = 20

    def init(self):
        super().init()
        self.relative_volume = self.I(calculate_relative_volume, self.data.Volume, self.volume_lookback)

    def next(self):
        if not self.should_trade():
            return

        needed_bars = max(self.volume_lookback, self.breakout_lookback, self.trend_ma_period) + 2
        if len(self.data) < needed_bars:
            return

        if self.position:
            return

        if not self._is_uptrend() or not self._is_bullish_candle():
            return

        prior_high = float(np.max(np.asarray(self.data.High[-(self.breakout_lookback + 1):-1], dtype=float)))
        current_close = float(self.data.Close[-1])
        current_rel_volume = float(self.relative_volume[-1])
        breakout_level = prior_high * (1 + self.breakout_buffer_pct)

        if current_rel_volume >= self.volume_multiplier and current_close > breakout_level:
            self._enter_with_atr_risk()


class NewLocalHighBreakoutStrategy(MomentumBreakoutBaseStrategy):
    """
    Long breakout strategy for fresh local highs in an existing uptrend.
    """

    local_high_lookback = 30
    min_relative_volume = 1.0
    volume_lookback = 10

    def init(self):
        super().init()
        self.relative_volume = self.I(calculate_relative_volume, self.data.Volume, self.volume_lookback)

    def next(self):
        if not self.should_trade():
            return

        needed_bars = max(self.local_high_lookback, self.trend_ma_period, self.volume_lookback) + 2
        if len(self.data) < needed_bars:
            return

        if self.position:
            return

        if not self._is_uptrend() or not self._is_bullish_candle():
            return

        prior_local_high = float(np.max(np.asarray(self.data.High[-(self.local_high_lookback + 1):-1], dtype=float)))
        current_close = float(self.data.Close[-1])
        current_rel_volume = float(self.relative_volume[-1])
        breakout_level = prior_local_high * (1 + self.breakout_buffer_pct)

        if current_rel_volume >= self.min_relative_volume and current_close > breakout_level:
            self._enter_with_atr_risk()


class ResistanceBreakoutStrategy(MomentumBreakoutBaseStrategy):
    """
    Long breakout strategy when price closes above a clustered horizontal resistance.
    """

    resistance_lookback = 50
    level_tolerance = 0.01
    min_touches = 2
    volume_lookback = 10
    min_relative_volume = 1.2

    def init(self):
        super().init()
        self.relative_volume = self.I(calculate_relative_volume, self.data.Volume, self.volume_lookback)

    def next(self):
        if not self.should_trade():
            return

        needed_bars = max(self.resistance_lookback, self.trend_ma_period, self.volume_lookback) + 2
        if len(self.data) < needed_bars:
            return

        if self.position:
            return

        if not self._is_uptrend() or not self._is_bullish_candle():
            return

        highs = np.asarray(self.data.High[-(self.resistance_lookback + 1):-1], dtype=float)
        resistance = _estimate_horizontal_resistance(highs, self.level_tolerance, self.min_touches)
        if resistance is None:
            return

        current_close = float(self.data.Close[-1])
        previous_close = float(self.data.Close[-2])
        breakout_level = resistance * (1 + self.breakout_buffer_pct)
        current_rel_volume = float(self.relative_volume[-1])

        if previous_close <= breakout_level and current_close > breakout_level and current_rel_volume >= self.min_relative_volume:
            self._enter_with_atr_risk()


class AscendingTriangleBreakoutStrategy(MomentumBreakoutBaseStrategy):
    """
    Long breakout strategy using a simplified ascending-triangle approximation.
    """

    pattern_lookback = 40
    level_tolerance = 0.01
    min_ceiling_touches = 2
    min_support_slope = 0.0
    volume_lookback = 10
    min_relative_volume = 1.2

    def init(self):
        super().init()
        self.relative_volume = self.I(calculate_relative_volume, self.data.Volume, self.volume_lookback)

    def _rising_support_confirmed(self, lows: np.ndarray) -> bool:
        if len(lows) < 6:
            return False
        x = np.arange(len(lows), dtype=float)
        slope, _ = np.polyfit(x, lows, 1)
        return slope > self.min_support_slope

    def next(self):
        if not self.should_trade():
            return

        needed_bars = max(self.pattern_lookback, self.trend_ma_period, self.volume_lookback) + 2
        if len(self.data) < needed_bars:
            return

        if self.position:
            return

        if not self._is_uptrend() or not self._is_bullish_candle():
            return

        highs = np.asarray(self.data.High[-(self.pattern_lookback + 1):-1], dtype=float)
        lows = np.asarray(self.data.Low[-(self.pattern_lookback + 1):-1], dtype=float)
        resistance = _estimate_horizontal_resistance(highs, self.level_tolerance, self.min_ceiling_touches)
        if resistance is None or not self._rising_support_confirmed(lows):
            return

        current_close = float(self.data.Close[-1])
        previous_close = float(self.data.Close[-2])
        breakout_level = resistance * (1 + self.breakout_buffer_pct)
        current_rel_volume = float(self.relative_volume[-1])

        if previous_close <= breakout_level and current_close > breakout_level and current_rel_volume >= self.min_relative_volume:
            self._enter_with_atr_risk()


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
    'unusual_volume_breakout': UnusualVolumeBreakoutStrategy,
    'new_local_high_breakout': NewLocalHighBreakoutStrategy,
    'resistance_breakout': ResistanceBreakoutStrategy,
    'ascending_triangle_breakout': AscendingTriangleBreakoutStrategy,
    'monthly_dca': DCAMonthlyStrategy,
    'signal_based_dca': DCASignalStrategy,
    'test_rsi_strategy': TestRSIStrategy,
    'emacrossrsistrategy': EMAcrossRSIStrategy,
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
