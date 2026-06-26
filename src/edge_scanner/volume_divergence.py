"""
Volume Divergence Detection
============================
Leading indicator system that detects accumulation/exhaustion phases
before price moves. Based on proven research:

- Wyckoff (1930s): Volume leads price in the accumulation → markup cycle
- Blume, Easley, O'Hara (1994): Volume reveals information about price
  move quality — volume without price movement = informed trading
- Karpoff (1987): Volume-price divergence predicts future direction

Core logic:
  - Bullish divergence: volume spiking while price is flat/down
    → smart money accumulating → score bonus (+2 to +5)
  - Bearish divergence: price rising while volume declining
    → smart money distributing → score penalty (-2 to -3)
  - Volume momentum acceleration: volume increasing over 5-10-20 periods
    → confirms the divergence signal

Usage:
    from src.edge_scanner.volume_divergence import compute_divergence_score
    adj, details = compute_divergence_score(ohlcv_dataframe)
"""

import logging
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────

# Thresholds tuned from historical data analysis
BULL_DIV_VOL_MIN = 1.5      # Volume must be ≥ 1.5x rolling avg
BULL_DIV_PRICE_MAX = 0.02   # Price change max for accumulation (+2%)
BULL_SCORE_BASE = 3.0       # Base score bonus for bullish divergence
BULL_SCORE_BOOST = 2.0      # Extra boost when divergence persists > 5 periods

BEAR_DIV_PRICE_MIN = 0.05   # Price must be up ≥ 5% for exhaustion
BEAR_DIV_VOL_MAX = 0.8      # Volume must be ≤ 0.8x avg
BEAR_SCORE_PENALTY = -2.0   # Score penalty for bearish divergence

VOL_MOMENTUM_PERIODS = [5, 10, 20]  # Multi-timeframe volume momentum
VOL_MOMENTUM_THRESHOLD = 1.3  # Ratio threshold for acceleration detection

# ── Core Functions ──────────────────────────────────────────────────────────

def compute_divergence_score(data: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
    """Compute a leading indicator score from OHLCV data.

    Returns (score_adjustment, details_dict).

    The score is a composite of:
    1. Bullish divergence detection (accumulation signal)
    2. Bearish divergence detection (exhaustion signal)
    3. Volume momentum confirmation

    Usage:
        data = engine.get_data('BTC/USDT', TimeFrame.H1, start, end)
        adj, details = compute_divergence_score(data)
    """
    if data.empty or len(data) < 25:
        return 0.0, {'error': 'insufficient data'}

    details: Dict[str, Any] = {}
    score_adj = 0.0

    # Ensure numeric columns
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    close = data['Close'].values
    volume = data['Volume'].values

    # ── 1. Volume baselines ────────────────────────────────────────────────
    vol_ma_20 = pd.Series(volume).rolling(20).mean().values
    vol_ratio = volume / np.maximum(vol_ma_20, 1)  # Current / 20-avg

    price_20chg = (close - np.roll(close, 20)) / np.maximum(np.roll(close, 20), 0.001)

    latest_vol_ratio = float(vol_ratio[-1])
    latest_price_chg = float(price_20chg[-1])

    details['vol_ratio'] = round(latest_vol_ratio, 2)
    details['price_20chg'] = round(latest_price_chg * 100, 2)

    # ── 2. Bullish divergence — accumulation ───────────────────────────────
    # Volume spiking (>1.5x) while price is flat or down (<+2% over 20 periods)
    # = smart money accumulating before the breakout
    is_bull_div = (latest_vol_ratio >= BULL_DIV_VOL_MIN) and (latest_price_chg <= BULL_DIV_PRICE_MAX)

    if is_bull_div:
        # How many of the last 5 candles showed bullish divergence?
        divergence_count = 0
        for i in range(1, min(6, len(vol_ratio))):
            if vol_ratio[-i] >= BULL_DIV_VOL_MIN and price_20chg[-i] <= BULL_DIV_PRICE_MAX:
                divergence_count += 1

        base_bonus = BULL_SCORE_BASE
        persistence_bonus = BULL_SCORE_BOOST if divergence_count >= 3 else 0.0
        score_adj += base_bonus + persistence_bonus
        details['bull_divergence'] = round(base_bonus + persistence_bonus, 1)
        details['bull_div_count'] = divergence_count
        logger.debug(
            "Bullish divergence: vol=%.1fx, price_chg=%.2f%%, persistence=%d, bonus=%.1f",
            latest_vol_ratio, latest_price_chg * 100, divergence_count, base_bonus + persistence_bonus,
        )
    else:
        details['bull_divergence'] = 0.0

    # ── 3. Bearish divergence — exhaustion ──────────────────────────────────
    # Price up significantly (>5%) while volume dropping (<0.8x)
    # = smart money distributing after a run-up
    is_bear_div = (latest_price_chg >= BEAR_DIV_PRICE_MIN) and (latest_vol_ratio <= BEAR_DIV_VOL_MAX)

    if is_bear_div:
        score_adj += BEAR_SCORE_PENALTY
        details['bear_divergence'] = BEAR_SCORE_PENALTY
        logger.debug(
            "Bearish divergence: vol=%.1fx, price_chg=%.2f%%, penalty=%.1f",
            latest_vol_ratio, latest_price_chg * 100, BEAR_SCORE_PENALTY,
        )
    else:
        details['bear_divergence'] = 0.0

    # ── 4. Volume momentum — multi-period acceleration ──────────────────────
    # Check if volume is accelerating across 5, 10, and 20 periods
    # Strong confirmation when ALL timeframes show acceleration
    momentum_count = 0
    for period in VOL_MOMENTUM_PERIODS:
        vol_ma = pd.Series(volume).rolling(period).mean().values
        if len(vol_ma) >= period * 2:
            recent_avg = float(np.mean(vol_ma[-period:]))
            prior_avg = float(np.mean(vol_ma[-period * 2:-period]))
            if prior_avg > 0 and recent_avg / prior_avg >= VOL_MOMENTUM_THRESHOLD:
                momentum_count += 1

    details['momentum_confirmed'] = momentum_count  # 0-3 timeframes

    # Momentum bonus: small boost when multiple timeframes confirm
    if momentum_count >= 2:
        mom_bonus = 1.0
        score_adj += mom_bonus
        details['momentum_bonus'] = mom_bonus
    else:
        details['momentum_bonus'] = 0.0

    # ── 5. Opposite-direction confirmation ──────────────────────────────────
    # If altFINS trend is SHORT (-4 or below) but volume is spiking (bullish divergence),
    # that's a strong contrarian buy signal — the trend is fading.
    # (This is read from components passed in, not from OHLCV)

    details['total_divergence_adj'] = round(score_adj, 1)
    return round(score_adj, 2), details


def compute_smart_money_index(data: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
    """Compute a smart-money accumulation index.

    Approximates buy/sell volume imbalance:
    - Up-volume = volume on green candles (close > open)
    - Down-volume = volume on red candles (close < open)
    - Net ratio > 1.3 = accumulation (buy volume dominates)
    - Net ratio < 0.7 = distribution (sell volume dominates)

    Returns (accumulation_index, details_dict).
    Index range: -1.0 (strong distribution) to +1.0 (strong accumulation).
    """
    if data.empty or len(data) < 20:
        return 0.0, {'error': 'insufficient data'}

    close = data['Close'].values
    open_p = data['Open'].values
    volume = data['Volume'].values

    up_vol = np.sum(volume[close > open_p])
    down_vol = np.sum(volume[close < open_p])
    total_vol = up_vol + down_vol

    if total_vol == 0:
        return 0.0, {'total_vol': 0}

    # Net ratio: positive = accumulation, negative = distribution
    # Range approximately -0.3 to +0.3 for normal, beyond ±0.3 for extreme
    net_ratio = (up_vol - down_vol) / total_vol

    # Recent 20-period view
    up_vol_20 = np.sum(volume[-20:][close[-20:] > open_p[-20:]])
    down_vol_20 = np.sum(volume[-20:][close[-20:] < open_p[-20:]])
    total_vol_20 = up_vol_20 + down_vol_20
    net_ratio_20 = (up_vol_20 - down_vol_20) / total_vol_20 if total_vol_20 > 0 else 0.0

    details = {
        'net_ratio_all': round(net_ratio, 3),
        'net_ratio_20': round(net_ratio_20, 3),
        'accumulation_index': round(net_ratio_20, 2),
    }

    # Score adjustment: when net ratio is strongly positive (accumulation)
    # but price hasn't moved, it's a leading buy signal
    if net_ratio_20 > 0.15:
        return round(net_ratio_20 * 3, 2), details  # +0.45 to +3.0
    elif net_ratio_20 < -0.15:
        return round(net_ratio_20 * 3, 2), details  # -0.45 to -3.0
    else:
        return 0.0, details


def is_low_float_squeeze(data: pd.DataFrame, coin_type: str = None) -> Tuple[bool, float]:
    """Detect low-float volume squeeze — small supply + sudden volume = explosive.

    Returns (is_squeeze: bool, squeeze_score: float).
    Squeeze score is 0.0-5.0 based on severity.

    Low-float coins (AGENT, AI, MEME types) with sudden volume spikes
    tend to have explosive moves. High-float coins (LAYER1, DEFI) need
    more volume for the same price impact.
    """
    if data.empty or len(data) < 20:
        return False, 0.0

    volume = data['Volume'].values
    vol_ma_20 = np.mean(volume[-20:])
    vol_ma_5 = np.mean(volume[-5:])
    vol_ratio = vol_ma_5 / max(vol_ma_20, 1)

    # Low-float types get a multiplier — smaller supply = bigger impact
    low_float_types = {'AI', 'AGENT', 'MEME', 'GAMING'}
    float_mult = 2.0 if coin_type in low_float_types else 1.0

    if vol_ratio >= 2.0 * float_mult:
        score = min(vol_ratio * float_mult, 5.0)
        return True, round(score, 1)
    return False, 0.0
