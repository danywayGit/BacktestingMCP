"""
Dynamic weight adjustment based on market regime detection.

Provides multipliers and adjusted scoring weights for each market regime
state. The weight_adjuster integrates with the regime_detector to produce
dynamically-tuned ScoringConfig-like dicts per scan cycle.

Regime strategies
-----------------
BULL_TRENDING   — Boost trend-following weights, dial back signal feed
BEAR_TRENDING   — Boost signal feed and on-chain flow (smart money)
SIDEWAYS        — Reduce trend weight, boost mean-reversion (RSI)
HIGH_VOLATILITY — Boost TR/ATR breakout detection, reduce everything else
LOW_VOLATILITY  — Boost volume and scanner hits (quiet accumulation)
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Optional

from .scoring_config import ScoringConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regime weight multipliers
# Each multiplier is applied on top of the base ScoringConfig weights.
# A multiplier of 1.0 means no change.
# ---------------------------------------------------------------------------

REGIME_MULTIPLIERS: dict[str, dict[str, float]] = {
    "BULL_TRENDING": {
        "trend_weight": 1.2,
        "volume_relative_weight": 1.1,
        "signal_feed_weight": 0.9,
        "scanner_hit_weight": 1.0,
        "onchain_netflow_weight": 1.0,
        "adx_weight": 1.0,
        "obv_trend_weight": 1.0,
        "medium_term_trend_weight": 1.0,
        "rsi_momentum_weight": 0.8,
        "price_change_1w_weight": 1.0,
        "tr_vs_atr_weight": 1.0,
    },
    "BEAR_TRENDING": {
        "trend_weight": 1.0,
        "volume_relative_weight": 1.0,
        "signal_feed_weight": 1.3,
        "scanner_hit_weight": 1.0,
        "onchain_netflow_weight": 1.2,
        "adx_weight": 1.0,
        "obv_trend_weight": 1.1,
        "medium_term_trend_weight": 1.0,
        "rsi_momentum_weight": 1.0,
        "price_change_1w_weight": 1.0,
        "tr_vs_atr_weight": 1.0,
    },
    "SIDEWAYS": {
        "trend_weight": 0.7,
        "volume_relative_weight": 1.0,
        "signal_feed_weight": 1.0,
        "scanner_hit_weight": 1.0,
        "onchain_netflow_weight": 0.9,
        "adx_weight": 0.5,
        "obv_trend_weight": 1.0,
        "medium_term_trend_weight": 0.6,
        "rsi_momentum_weight": 1.5,
        "price_change_1w_weight": 1.2,
        "tr_vs_atr_weight": 0.8,
    },
    "HIGH_VOLATILITY": {
        "trend_weight": 0.8,
        "volume_relative_weight": 0.8,
        "signal_feed_weight": 0.8,
        "scanner_hit_weight": 0.8,
        "onchain_netflow_weight": 0.8,
        "adx_weight": 1.0,
        "obv_trend_weight": 0.8,
        "medium_term_trend_weight": 0.8,
        "rsi_momentum_weight": 0.8,
        "price_change_1w_weight": 0.8,
        "tr_vs_atr_weight": 1.5,
    },
    "LOW_VOLATILITY": {
        "trend_weight": 1.0,
        "volume_relative_weight": 1.3,
        "signal_feed_weight": 1.0,
        "scanner_hit_weight": 1.2,
        "onchain_netflow_weight": 1.0,
        "adx_weight": 1.0,
        "obv_trend_weight": 1.0,
        "medium_term_trend_weight": 1.0,
        "rsi_momentum_weight": 1.0,
        "price_change_1w_weight": 1.0,
        "tr_vs_atr_weight": 1.0,
    },
    "UNKNOWN": {k: 1.0 for k in [
        "trend_weight", "volume_relative_weight", "signal_feed_weight",
        "scanner_hit_weight", "onchain_netflow_weight", "adx_weight",
        "obv_trend_weight", "medium_term_trend_weight", "rsi_momentum_weight",
        "price_change_1w_weight", "tr_vs_atr_weight",
    ]},
}

REGIME_LABELS: dict[str, str] = {
    "BULL_TRENDING": "Bullish Trending ↗",
    "BEAR_TRENDING": "Bearish Trending ↘",
    "SIDEWAYS": "Sideways / Ranging ↔",
    "HIGH_VOLATILITY": "High Volatility ⚡",
    "LOW_VOLATILITY": "Low Volatility 💤",
    "UNKNOWN": "Unknown Regime ❓",
}


def get_adjusted_weights(config: ScoringConfig, regime: str) -> dict[str, float]:
    """Apply regime multipliers to a ScoringConfig's weights.

    Parameters
    ----------
    config : ScoringConfig
        Base configuration whose weights will be scaled.
    regime : str
        One of 'BULL_TRENDING', 'BEAR_TRENDING', 'SIDEWAYS',
        'HIGH_VOLATILITY', 'LOW_VOLATILITY', or 'UNKNOWN'.

    Returns
    -------
    dict
        Mapping of weight field names to adjusted float values.
        Returns a copy — the original config is never mutated.
    """
    multipliers = REGIME_MULTIPLIERS.get(regime, REGIME_MULTIPLIERS["UNKNOWN"])
    adjusted: dict[str, float] = {}

    for field in (
        "trend_weight",
        "volume_relative_weight",
        "signal_feed_weight",
        "scanner_hit_weight",
        "onchain_netflow_weight",
        "adx_weight",
        "obv_trend_weight",
        "medium_term_trend_weight",
        "rsi_momentum_weight",
        "price_change_1w_weight",
        "tr_vs_atr_weight",
    ):
        base = getattr(config, field, 0.0)
        mult = multipliers.get(field, 1.0)
        adjusted[field] = round(base * mult, 4)

    return adjusted


def compute_regime_adjusted_score(
    base_score: float,
    base_score_components: dict[str, Any],
    regime: str,
    config: Optional[ScoringConfig] = None,
) -> tuple[float, dict[str, Any]]:
    """Apply regime-aware weight adjustments to an already-computed score.

    This function takes the base composite score and its components, then
    re-weights them according to the detected regime. It is designed to be
    called **after** the standard composite.py score is computed, so the
    regime-adjusted score can be stored alongside the base score.

    Parameters
    ----------
    base_score : float
        The original composite score from score_symbol().
    base_score_components : dict
        The 'components' dict from the CandidateScore (contains all
        intermediate values used in the scoring formula).
    regime : str
        Detected market regime string.
    config : ScoringConfig or None
        The config used to compute the base score. If None, the active
        config will be loaded (but the caller should pass it to avoid
        a circular import and ensure correct attribution).

    Returns
    -------
    tuple[float, dict]
        (adjusted_score, adjustment_info)
        adjustment_info contains the regime breakdown for reporting.
    """
    from .scoring_config import ACTIVE_CONFIG

    cfg = config or ACTIVE_CONFIG

    adjusted_weights = get_adjusted_weights(cfg, regime)
    base_weights = {
        field: getattr(cfg, field, 0.0)
        for field in adjusted_weights
    }

    # Compute the score delta by comparing original vs adjusted contributions
    delta = 0.0
    breakdown: dict[str, Any] = {}

    # Trend contribution
    trend_score = base_score_components.get("altfins_trend_score", 0)
    base_trend_contrib = trend_score * base_weights["trend_weight"]
    adj_trend_contrib = trend_score * adjusted_weights["trend_weight"]
    breakdown["trend_adjustment"] = round(adj_trend_contrib - base_trend_contrib, 4)
    delta += breakdown["trend_adjustment"]

    # Volume relative contribution
    vol_rel = base_score_components.get("altfins_volume_relative", 0)
    vol_excess = max(vol_rel - 1.0, 0.0)
    base_vol_contrib = vol_excess * base_weights["volume_relative_weight"]
    adj_vol_contrib = vol_excess * adjusted_weights["volume_relative_weight"]
    breakdown["volume_adjustment"] = round(adj_vol_contrib - base_vol_contrib, 4)
    delta += breakdown["volume_adjustment"]

    # Signal feed contribution
    feed_dir = base_score_components.get("altfins_signal_feed")
    if feed_dir == "BULLISH":
        base_feed = base_weights["signal_feed_weight"]
        adj_feed = adjusted_weights["signal_feed_weight"]
    elif feed_dir == "BEARISH":
        base_feed = -base_weights["signal_feed_weight"]
        adj_feed = -adjusted_weights["signal_feed_weight"]
    else:
        base_feed = adj_feed = 0.0
    breakdown["signal_feed_adjustment"] = round(adj_feed - base_feed, 4)
    delta += breakdown["signal_feed_adjustment"]

    # On-chain netflow contribution
    netflow = base_score_components.get("onchain_netflow_ratio")
    if netflow is not None:
        base_onchain = netflow * base_weights["onchain_netflow_weight"]
        adj_onchain = netflow * adjusted_weights["onchain_netflow_weight"]
        breakdown["onchain_adjustment"] = round(adj_onchain - base_onchain, 4)
        delta += breakdown["onchain_adjustment"]

    # Extended signal adjustments (ADX, RSI, OBV, price_change, tr_vs_atr)
    # These are less common — only compute if they had non-zero contribution
    ext_fields = {
        "adx": "adx_weight",
        "obv_trend_pct": "obv_trend_weight",
        "medium_term_trend": "medium_term_trend_weight",
        "rsi14": "rsi_momentum_weight",
        "price_change_1w_pct": "price_change_1w_weight",
        "tr_vs_atr": "tr_vs_atr_weight",
    }
    for component_key, weight_field in ext_fields.items():
        val = base_score_components.get(component_key)
        if val is not None and weight_field in base_weights:
            base_ext = val * base_weights[weight_field]
            adj_ext = val * adjusted_weights[weight_field]
            adj_key = f"{component_key}_adjustment"
            breakdown[adj_key] = round(adj_ext - base_ext, 4)
            delta += breakdown[adj_key]

    adjusted_score = round(base_score + delta, 2)

    adjustment_info = {
        "regime": regime,
        "regime_label": REGIME_LABELS.get(regime, regime),
        "base_score": base_score,
        "adjusted_score": adjusted_score,
        "delta": round(delta, 2),
        "adjusted_weights": adjusted_weights,
        "breakdown": breakdown,
    }

    return adjusted_score, adjustment_info


def apply_regime_adjustment_to_config(
    config: ScoringConfig,
    regime: str,
) -> ScoringConfig:
    """Return a NEW ScoringConfig with regime-adjusted weights.

    This creates a frozen copy with the adjusted weights applied.
    The original config is never modified.

    Parameters
    ----------
    config : ScoringConfig
        Base configuration.
    regime : str
        Detected market regime.

    Returns
    -------
    ScoringConfig
        A new ScoringConfig with adjusted weights (version is suffixed).
    """
    adjusted = get_adjusted_weights(config, regime)
    params = config.to_dict()
    params.update(adjusted)
    params["version"] = f"{config.version}+{regime.lower()}"
    params["description"] = f"{config.description} | Regime-adjusted: {REGIME_LABELS.get(regime, regime)}"
    return ScoringConfig(**params)