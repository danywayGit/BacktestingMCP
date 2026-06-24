"""
Order-flow signal sources — computes volume-profile and volume-momentum
metrics from raw OHLCV data for use in the edge scoring system.

Each metric is normalised to the range [-1.0, 1.0] where
  positive = bullish implication,
  negative = bearish implication,
    0.0   = neutral / insufficient data.

All functions are safe to call with empty or insufficient data — they
return a neutral (0.0, 0.0, …) result and log a debug message.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import TimeFrame
from src.core.backtesting_engine import engine

logger = logging.getLogger(__name__)

# ── helpers ──────────────────────────────────────────────────────────────────

_NO_DATA: Dict[str, float] = {
    "volume_profile_hvn": 0.0,
    "volume_profile_lvn": 0.0,
    "volume_delta": 0.0,
    "volume_momentum_score": 0.0,
}


def _log_return(msg: str, symbol: str, target: str = "all") -> Dict[str, float]:
    """Log a neutral return reason and yield the zero dict."""
    logger.debug("orderflow[%s] %s — %s", symbol, msg, target)
    return _NO_DATA


def _safe_volume(data: pd.DataFrame, window: int = 20) -> np.ndarray:
    """Compute a rolling-average-smoothed volume series (avoid outliers)."""
    vol = data["Volume"].values.astype(float)
    if len(vol) < window:
        return vol
    return vol / np.maximum(
        pd.Series(vol).rolling(window, min_periods=1).mean().values, 1e-9
    )


# ── public API ───────────────────────────────────────────────────────────────


def compute_orderflow_metrics(
    symbol: str,
    timeframe: TimeFrame = TimeFrame.H1,
    lookback_days: int = 30,
    bins: int = 20,
) -> Dict[str, float]:
    """Compute all volume-profile / order-flow metrics for *symbol*.

    Parameters
    ----------
    symbol : str
        altFINS-style symbol (e.g. ``"BTC"``, ``"ETH"``).  The pair is
        built by appending ``"USDT"``.
    timeframe : TimeFrame
        Bar interval (default: 1 h).
    lookback_days : int
        How many days of history to fetch (default: 30).
    bins : int
        Number of price bins for VPVR (default: 20).

    Returns
    -------
    dict
        Keys: ``volume_profile_hvn``, ``volume_profile_lvn``,
        ``volume_delta``, ``volume_momentum_score``.
        Each value is in [-1, 1].  On any error the dict contains only 0.0.
    """
    pair = f"{symbol.upper()}USDT"
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)

    try:
        data = engine.get_data(pair, timeframe, start, end)
    except Exception as exc:
        return _log_return("get_data failed", symbol, str(exc))

    if data is None or not isinstance(data, pd.DataFrame) or data.empty:
        return _log_return("no OHLCV data returned", symbol)
    if "Volume" not in data.columns or "Close" not in data.columns:
        return _log_return("missing required columns (Volume, Close)", symbol)

    # ──────────────────────────────────────────────────────────────────────
    # 1. VPVR — Volume-by-Price (High- / Low-Volume Nodes)
    # ──────────────────────────────────────────────────────────────────────
    hvn_score, lvn_score = _compute_vpvr(data, bins)

    # ──────────────────────────────────────────────────────────────────────
    # 2. Volume Delta — approximated buy/sell imbalance
    # ──────────────────────────────────────────────────────────────────────
    delta_score = _compute_volume_delta(data)

    # ──────────────────────────────────────────────────────────────────────
    # 3. Volume Momentum — rate of change of smoothed volume
    # ──────────────────────────────────────────────────────────────────────
    momentum_score = _compute_volume_momentum(data)

    return {
        "volume_profile_hvn": round(float(hvn_score), 4),
        "volume_profile_lvn": round(float(lvn_score), 4),
        "volume_delta": round(float(delta_score), 4),
        "volume_momentum_score": round(float(momentum_score), 4),
    }


# ── internal component functions ─────────────────────────────────────────────


def _compute_vpvr(
    data: pd.DataFrame, bins: int = 20
) -> Tuple[float, float]:
    """
    Volume-by-Price Visible Range analysis.

    Returns
    -------
    (hvn_score, lvn_score)  both in [-1, 1].

    * HVN score: positive when the current price sits *inside* or just
      above a high-volume node (support), negative when it sits just
      below (resistance).  0 when no HVN interaction.
    * LVN score: positive when price breaks *above* a low-volume gap
      (upside breakout potential), negative when it breaks *below*
      (downside vulnerability).
    """
    df = data.copy()
    if len(df) < bins:
        return 0.0, 0.0

    price_min = float(df["Low"].min())
    price_max = float(df["High"].max())
    if price_max <= price_min:
        return 0.0, 0.0

    bin_width = (price_max - price_min) / bins
    current_price = float(df["Close"].iloc[-1])
    current_bin = int((current_price - price_min) / bin_width)
    current_bin = min(current_bin, bins - 1)

    # Bin edges
    edges = np.linspace(price_min, price_max, bins + 1)

    # Sum volume per bin
    volume_by_bin = np.zeros(bins, dtype=float)
    for i in range(len(df)):
        low = float(df["Low"].iloc[i])
        high = float(df["High"].iloc[i])
        vol = float(df["Volume"].iloc[i])
        # Distribute volume proportionally across bins this candle spans
        bin_low = max(0, int((low - price_min) / bin_width))
        bin_high = min(bins - 1, int((high - price_min) / bin_width))
        cnt = max(bin_high - bin_low + 1, 1)
        volume_by_bin[bin_low : bin_high + 1] += vol / cnt

    avg_vol = float(np.mean(volume_by_bin))
    if avg_vol <= 0:
        return 0.0, 0.0

    # HVN: bins with volume > 1.5x average
    hvn_mask = volume_by_bin > 1.5 * avg_vol
    # LVN: bins with volume < 0.5x average
    lvn_mask = volume_by_bin < 0.5 * avg_vol

    # ── HVN score ────────────────────────────────────────────────────
    # If current bin is an HVN:
    #   check nearest HVN above (resistance) and below (support)
    hvn_score = 0.0
    if hvn_mask[current_bin]:
        # Inside an HVN → check if trend is leaning one way
        # Find nearest non-current HVN below (support) and above (resistance)
        below = np.where(hvn_mask[:current_bin])[0]
        above = np.where(hvn_mask[current_bin + 1 :])[0] + current_bin + 1
        has_support = len(below) > 0
        has_resistance = len(above) > 0
        if has_support and not has_resistance:
            hvn_score = 0.5   # support below, breakout potential up
        elif has_resistance and not has_support:
            hvn_score = -0.5  # resistance above, rejection potential down
        else:
            hvn_score = 0.2   # sandwiched between HVNs — neutral-ish
    else:
        # Price is not in an HVN — check proximity to nearest HVN
        hvns = np.where(hvn_mask)[0]
        if len(hvns) > 0:
            nearest = hvns[np.argmin(np.abs(hvns - current_bin))]
            dist = nearest - current_bin
            if abs(dist) <= 2:  # within 2 bins
                if dist > 0:
                    # HVN is above = resistance overhead
                    hvn_score = -0.3 * (1.0 - abs(dist) / 3.0)
                else:
                    # HVN is below = support underneath
                    hvn_score = 0.3 * (1.0 - abs(dist) / 3.0)

    # ── LVN score ────────────────────────────────────────────────────
    # LVN gaps represent areas of low trading activity —
    # if price is breaking through an LVN, it suggests a directional move.
    lvn_score = 0.0
    if lvn_mask[current_bin]:
        # Price is currently in a low-volume node
        # Check direction of most recent candles
        recent_close = float(df["Close"].iloc[-min(5, len(df)) :].mean())
        prev_close = float(
            df["Close"].iloc[-min(10, len(df)) : -min(5, len(df))].mean()
            if len(df) >= 10
            else df["Close"].iloc[0]
        )
        if recent_close > prev_close:
            lvn_score = 0.7  # breaking up through LVN → bullish
        elif recent_close < prev_close:
            lvn_score = -0.7  # breaking down through LVN → bearish
        else:
            lvn_score = 0.0  # sideways in LVN → neutral

    return hvn_score, lvn_score


def _compute_volume_delta(data: pd.DataFrame) -> float:
    """
    Approximate buy/sell volume imbalance.

    Uses price-change direction × normalised volume as a proxy for
    delta (since we don't have tick-level data).  Each candle:
        signed_vol = sign(close - open) * (volume / avg_volume)
    Return value is the mean signed volume over the lookback, clamped
    to [-1, 1].
    """
    df = data.copy()
    if len(df) < 5:
        return 0.0

    closes = df["Close"].values.astype(float)
    opens = df["Open"].values.astype(float)
    volumes = df["Volume"].values.astype(float)

    avg_vol = float(np.mean(volumes))
    if avg_vol <= 0:
        return 0.0

    price_dir = np.sign(closes - opens)
    # Flat candles (close ≈ open) get 0
    price_dir[np.abs(closes - opens) < 1e-8] = 0.0

    norm_vol = volumes / avg_vol
    signed_vol = price_dir * norm_vol

    # Exponentially-weighted average to favour recent activity
    weights = np.exp(np.linspace(0, 1, len(signed_vol)))
    weights /= weights.sum()
    delta = float(np.average(signed_vol, weights=weights))

    # Clamp to [-1, 1]
    return max(-1.0, min(1.0, delta))


def _compute_volume_momentum(data: pd.DataFrame, window: int = 10) -> float:
    """
    Rate of change of smoothed volume.

    Positive = volume expanding (increasing interest).
    Negative = volume contracting (waning interest).
    Return value in [-1, 1].
    """
    df = data.copy()
    if len(df) < window * 2:
        return 0.0

    volumes = df["Volume"].values.astype(float)

    # Simple moving average to smooth
    sma_short = pd.Series(volumes).rolling(window, min_periods=1).mean().values
    sma_long = pd.Series(volumes).rolling(window * 2, min_periods=1).mean().values

    # Rate of change of short-term average vs long-term baseline
    roc = np.mean((sma_short[-window:] - sma_long[-window:]) / np.maximum(sma_long[-window:], 1e-9))

    # Normalise: roc of ±0.5 → ±1.0 (volume doubling/halving)
    return max(-1.0, min(1.0, roc * 2.0))


# ── quick self-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    print("=== orderflow_signals self-test ===")

    # Test with BTC (30 days of 1h data)
    result = compute_orderflow_metrics("BTC", TimeFrame.H1, 30)
    print(f"BTC 30d 1h: {result}")

    # Test with ETH
    result2 = compute_orderflow_metrics("ETH", TimeFrame.H1, 7)
    print(f"ETH 7d 1h:  {result2}")

    # Test empty/no-data path
    result3 = compute_orderflow_metrics("NONEXISTENT999", TimeFrame.H1, 1)
    print(f"NoData:     {result3}")

    print("=== orderflow_signals self-test complete ===")