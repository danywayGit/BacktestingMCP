"""
Scanner functions for momentum breakout opportunities.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


SCAN_TYPES = [
    "unusual_volume_breakout",
    "new_local_high_breakout",
    "resistance_breakout",
    "ascending_triangle_breakout",
]


def _require_ohlcv(data: pd.DataFrame) -> None:
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def _atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = data["High"] - data["Low"]
    high_close = (data["High"] - data["Close"].shift(1)).abs()
    low_close = (data["Low"] - data["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean().fillna(method="bfill")


def _relative_volume(data: pd.DataFrame, lookback: int = 10) -> pd.Series:
    avg = data["Volume"].rolling(lookback, min_periods=1).mean()
    rel = data["Volume"] / avg.replace(0, np.nan)
    return rel.fillna(1.0)


def _uptrend_filter(data: pd.DataFrame, trend_ma_period: int = 50) -> bool:
    ma = data["Close"].rolling(trend_ma_period, min_periods=trend_ma_period).mean()
    if len(ma) == 0 or not np.isfinite(ma.iloc[-1]):
        return False
    return bool(data["Close"].iloc[-1] > ma.iloc[-1])


def _bullish_candle(data: pd.DataFrame) -> bool:
    return bool(data["Close"].iloc[-1] > data["Open"].iloc[-1])


def _estimate_horizontal_resistance(high_values: np.ndarray, tolerance: float, min_touches: int) -> Optional[float]:
    if len(high_values) < 5:
        return None

    local_highs = []
    for i in range(1, len(high_values) - 1):
        if high_values[i] > high_values[i - 1] and high_values[i] > high_values[i + 1]:
            local_highs.append(float(high_values[i]))

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

    return float(np.mean(max(valid_groups, key=lambda g: float(np.mean(g)))))


def scan_unusual_volume_breakout(data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    params = params or {}
    volume_lookback = int(params.get("volume_lookback", 10))
    volume_multiplier = float(params.get("volume_multiplier", 2.0))
    breakout_lookback = int(params.get("breakout_lookback", 20))
    trend_ma_period = int(params.get("trend_ma_period", 50))
    breakout_buffer_pct = float(params.get("breakout_buffer_pct", 0.001))

    needed = max(volume_lookback, breakout_lookback, trend_ma_period) + 2
    if len(data) < needed:
        return {"triggered": False, "reason": "not_enough_data"}

    rel_volume = _relative_volume(data, volume_lookback).iloc[-1]
    prior_high = float(data["High"].iloc[-(breakout_lookback + 1):-1].max())
    breakout_level = prior_high * (1 + breakout_buffer_pct)
    close = float(data["Close"].iloc[-1])

    triggered = (
        _uptrend_filter(data, trend_ma_period)
        and _bullish_candle(data)
        and rel_volume >= volume_multiplier
        and close > breakout_level
    )

    return {
        "triggered": triggered,
        "close": close,
        "breakout_level": breakout_level,
        "relative_volume": float(rel_volume),
    }


def scan_new_local_high_breakout(data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    params = params or {}
    local_high_lookback = int(params.get("local_high_lookback", 30))
    trend_ma_period = int(params.get("trend_ma_period", 50))
    min_relative_volume = float(params.get("min_relative_volume", 1.0))
    volume_lookback = int(params.get("volume_lookback", 10))
    breakout_buffer_pct = float(params.get("breakout_buffer_pct", 0.001))

    needed = max(local_high_lookback, trend_ma_period, volume_lookback) + 2
    if len(data) < needed:
        return {"triggered": False, "reason": "not_enough_data"}

    rel_volume = _relative_volume(data, volume_lookback).iloc[-1]
    prior_local_high = float(data["High"].iloc[-(local_high_lookback + 1):-1].max())
    breakout_level = prior_local_high * (1 + breakout_buffer_pct)
    close = float(data["Close"].iloc[-1])

    triggered = (
        _uptrend_filter(data, trend_ma_period)
        and _bullish_candle(data)
        and rel_volume >= min_relative_volume
        and close > breakout_level
    )

    return {
        "triggered": triggered,
        "close": close,
        "breakout_level": breakout_level,
        "relative_volume": float(rel_volume),
    }


def scan_resistance_breakout(data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    params = params or {}
    resistance_lookback = int(params.get("resistance_lookback", 50))
    trend_ma_period = int(params.get("trend_ma_period", 50))
    level_tolerance = float(params.get("level_tolerance", 0.01))
    min_touches = int(params.get("min_touches", 2))
    min_relative_volume = float(params.get("min_relative_volume", 1.2))
    volume_lookback = int(params.get("volume_lookback", 10))
    breakout_buffer_pct = float(params.get("breakout_buffer_pct", 0.001))

    needed = max(resistance_lookback, trend_ma_period, volume_lookback) + 2
    if len(data) < needed:
        return {"triggered": False, "reason": "not_enough_data"}

    highs = data["High"].iloc[-(resistance_lookback + 1):-1].to_numpy(dtype=float)
    resistance = _estimate_horizontal_resistance(highs, level_tolerance, min_touches)
    if resistance is None:
        return {"triggered": False, "reason": "no_resistance_cluster"}

    rel_volume = _relative_volume(data, volume_lookback).iloc[-1]
    close = float(data["Close"].iloc[-1])
    prev_close = float(data["Close"].iloc[-2])
    breakout_level = resistance * (1 + breakout_buffer_pct)

    triggered = (
        _uptrend_filter(data, trend_ma_period)
        and _bullish_candle(data)
        and rel_volume >= min_relative_volume
        and prev_close <= breakout_level
        and close > breakout_level
    )

    return {
        "triggered": triggered,
        "close": close,
        "breakout_level": breakout_level,
        "relative_volume": float(rel_volume),
        "resistance": resistance,
    }


def scan_ascending_triangle_breakout(data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    params = params or {}
    pattern_lookback = int(params.get("pattern_lookback", 40))
    trend_ma_period = int(params.get("trend_ma_period", 50))
    level_tolerance = float(params.get("level_tolerance", 0.01))
    min_ceiling_touches = int(params.get("min_ceiling_touches", 2))
    min_support_slope = float(params.get("min_support_slope", 0.0))
    min_relative_volume = float(params.get("min_relative_volume", 1.2))
    volume_lookback = int(params.get("volume_lookback", 10))
    breakout_buffer_pct = float(params.get("breakout_buffer_pct", 0.001))

    needed = max(pattern_lookback, trend_ma_period, volume_lookback) + 2
    if len(data) < needed:
        return {"triggered": False, "reason": "not_enough_data"}

    highs = data["High"].iloc[-(pattern_lookback + 1):-1].to_numpy(dtype=float)
    lows = data["Low"].iloc[-(pattern_lookback + 1):-1].to_numpy(dtype=float)
    resistance = _estimate_horizontal_resistance(highs, level_tolerance, min_ceiling_touches)
    if resistance is None:
        return {"triggered": False, "reason": "no_triangle_ceiling"}

    x = np.arange(len(lows), dtype=float)
    support_slope, _ = np.polyfit(x, lows, 1)
    if support_slope <= min_support_slope:
        return {"triggered": False, "reason": "support_not_rising", "support_slope": float(support_slope)}

    rel_volume = _relative_volume(data, volume_lookback).iloc[-1]
    close = float(data["Close"].iloc[-1])
    prev_close = float(data["Close"].iloc[-2])
    breakout_level = resistance * (1 + breakout_buffer_pct)

    triggered = (
        _uptrend_filter(data, trend_ma_period)
        and _bullish_candle(data)
        and rel_volume >= min_relative_volume
        and prev_close <= breakout_level
        and close > breakout_level
    )

    return {
        "triggered": triggered,
        "close": close,
        "breakout_level": breakout_level,
        "relative_volume": float(rel_volume),
        "support_slope": float(support_slope),
        "resistance": resistance,
    }


def evaluate_scan(data: pd.DataFrame, scan_name: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    _require_ohlcv(data)

    scan_map = {
        "unusual_volume_breakout": scan_unusual_volume_breakout,
        "new_local_high_breakout": scan_new_local_high_breakout,
        "resistance_breakout": scan_resistance_breakout,
        "ascending_triangle_breakout": scan_ascending_triangle_breakout,
    }

    if scan_name == "all":
        return {name: fn(data, params) for name, fn in scan_map.items()}

    if scan_name not in scan_map:
        raise ValueError(f"Unknown scan: {scan_name}. Available: {list(scan_map.keys()) + ['all']}")

    return scan_map[scan_name](data, params)
