"""
Order-flow weight manager — defines weight fields that would be added to
ScoringConfig for volume-profile and order-flow scoring, and provides the
scoring entry-point that the composite scanner can call.

The weights defined here are **candidates** for inclusion on ScoringConfig.
Because the main agent is adding them there, this module uses ``getattr``
with sensible defaults so it works immediately regardless of whether
``ScoringConfig`` has been updated yet.

Usage (once ScoringConfig has the fields)::

    from src.edge_scanner.orderflow_weight_manager import compute_orderflow_score

    extra_score, components = compute_orderflow_score(symbol, tf, days, config)

Or via :func:`apply_orderflow_to_score` to integrate into an incremental score.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from config.settings import TimeFrame
from src.edge_scanner.scoring_config import ScoringConfig
from src.integrations.orderflow_signals import compute_orderflow_metrics

logger = logging.getLogger(__name__)

# ── Weight field definitions (defaults for when ScoringConfig doesn't have them)
#    These mirror the field names that should be added to the ScoringConfig dataclass.

VOLUME_PROFILE_VPVR_WEIGHT: float = 0.15
"""Weight for HVN/LVN VPVR score contribution."""

VOLUME_DELTA_WEIGHT: float = 0.10
"""Weight for buy/sell volume delta contribution."""

VOLUME_MOMENTUM_WEIGHT: float = 0.05
"""Weight for volume momentum (rate-of-change) contribution."""

ORDERFLOW_CONFIRMATION_WEIGHT: float = 0.10
"""Extra bonus when multiple order-flow signals agree on direction."""

# All order-flow fields (for introspection / auto-enable logic)
ORDERFLOW_WEIGHT_FIELDS: Tuple[str, ...] = (
    "volume_profile_vpvr_weight",
    "volume_delta_weight",
    "volume_momentum_weight",
    "orderflow_confirmation_weight",
)


# ── public API ───────────────────────────────────────────────────────────────


def compute_orderflow_score(
    symbol: str,
    timeframe: TimeFrame,
    lookback_days: int,
    config: Optional[ScoringConfig] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Fetch order-flow metrics for *symbol* and combine them using *config*'s
    order-flow weights (falling back to module-level defaults when the config
    doesn't have the fields yet).

    Parameters
    ----------
    symbol : str
        altFINS-style symbol (e.g. ``"BTC"``).
    timeframe : TimeFrame
        Bar interval for OHLCV data.
    lookback_days : int
        How many days of data to fetch.
    config : ScoringConfig or None
        Scoring configuration to read weights from.  ``None`` uses the module
        defaults for all order-flow weights (scoring disabled).

    Returns
    -------
    (extra_score, components_dict)
        extra_score : float
            The combined order-flow score contribution to add to the composite.
        components_dict : dict
            Individual metric values for logging/diagnostics.
    """
    # ── fetch raw metrics ──────────────────────────────────────────────
    metrics = compute_orderflow_metrics(symbol, timeframe, lookback_days)

    # ── resolve weights ────────────────────────────────────────────────
    vpvr_w = _get_weight(config, "volume_profile_vpvr_weight", VOLUME_PROFILE_VPVR_WEIGHT)
    delta_w = _get_weight(config, "volume_delta_weight", VOLUME_DELTA_WEIGHT)
    momentum_w = _get_weight(config, "volume_momentum_weight", VOLUME_MOMENTUM_WEIGHT)
    confirm_w = _get_weight(config, "orderflow_confirmation_weight", ORDERFLOW_CONFIRMATION_WEIGHT)

    # If all weights are effectively zero, skip computation
    if abs(vpvr_w) < 1e-9 and abs(delta_w) < 1e-9 and abs(momentum_w) < 1e-9 and abs(confirm_w) < 1e-9:
        return 0.0, metrics

    # ── combine ────────────────────────────────────────────────────────
    extra = 0.0
    vpvr_contrib = (metrics["volume_profile_hvn"] + metrics["volume_profile_lvn"]) / 2.0
    extra += vpvr_contrib * vpvr_w

    delta_contrib = metrics["volume_delta"]
    extra += delta_contrib * delta_w

    momentum_contrib = metrics["volume_momentum_score"]
    extra += momentum_contrib * momentum_w

    # ── confirmation bonus ─────────────────────────────────────────────
    # Count how many order-flow metrics agree on direction (positive vs negative).
    signs = [
        np_sign(metrics["volume_profile_hvn"]),
        np_sign(metrics["volume_profile_lvn"]),
        np_sign(metrics["volume_delta"]),
        np_sign(metrics["volume_momentum_score"]),
    ]
    non_zero = [s for s in signs if s != 0]
    if len(non_zero) >= 3 and len(set(non_zero)) == 1:
        # Strong agreement among order-flow signals
        bonus = confirm_w * (1.0 if non_zero[0] > 0 else -1.0)
        extra += bonus
        metrics["orderflow_agreement"] = len(non_zero)
    elif len(non_zero) >= 2 and len(set(non_zero)) == 1:
        # Moderate agreement
        bonus = confirm_w * 0.5 * (1.0 if non_zero[0] > 0 else -1.0)
        extra += bonus
        metrics["orderflow_agreement"] = len(non_zero)
    else:
        metrics["orderflow_agreement"] = 0

    # Round for cleanliness
    extra = round(extra, 4)
    return extra, metrics


def apply_orderflow_to_score(
    current_score: float,
    symbol: str,
    timeframe: TimeFrame,
    lookback_days: int,
    config: Optional[ScoringConfig] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Convenience wrapper: compute order-flow contribution and add it to
    *current_score*.

    Equivalent to::

        extra, comp = compute_orderflow_score(...)
        return current_score + extra, comp
    """
    extra, components = compute_orderflow_score(symbol, timeframe, lookback_days, config)
    return current_score + extra, components


# ── helpers ──────────────────────────────────────────────────────────────────


def _get_weight(config: Optional[ScoringConfig], field: str, default: float) -> float:
    """Read a weight from *config* if it has the field, else return *default*."""
    if config is None:
        return 0.0  # weights are 0 when no config is given
    return float(getattr(config, field, default))


def np_sign(x: float) -> int:
    """Return -1, 0, or +1 (pure Python, no numpy dependency needed here)."""
    if x > 1e-9:
        return 1
    if x < -1e-9:
        return -1
    return 0


# ── quick self-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== orderflow_weight_manager self-test ===")

    # Test with a real symbol
    extra, comp = compute_orderflow_score("BTC", TimeFrame.H1, 30)
    print(f"BTC 30d 1h: extra={extra:.4f}, components={comp}")

    extra2, comp2 = compute_orderflow_score("ETH", TimeFrame.H1, 7)
    print(f"ETH 7d 1h:  extra={extra2:.4f}, components={comp2}")

    # With a ScoringConfig that has weights (via getattr fallback)
    from src.edge_scanner.scoring_config import CONFIG_V7_0
    extra3, comp3 = compute_orderflow_score("BTC", TimeFrame.H1, 30, config=CONFIG_V7_0)
    print(f"BTC + V7_0 config: extra={extra3:.4f}, components={comp3}")

    # Test apply wrapper
    base = 5.0
    new_score, comp4 = apply_orderflow_to_score(base, "BTC", TimeFrame.H1, 30, config=CONFIG_V7_0)
    print(f"apply_orderflow: base={base} -> new={new_score:.4f}")

    print("=== orderflow_weight_manager self-test complete ===")