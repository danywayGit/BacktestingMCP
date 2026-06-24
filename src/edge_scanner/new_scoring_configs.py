"""
New scoring configurations that extend the V7.0 series with order-flow
and volume-profile weights enabled.

Usage::

    from src.edge_scanner.new_scoring_configs import CONFIG_V7_1

    # Switch active config
    from src.edge_scanner.scoring_config import ACTIVE_CONFIG, ALL_CONFIGS
    # (main agent handles wiring into ALL_CONFIGS)

.. note::

   The four order-flow weight fields (``volume_profile_vpvr_weight``,
   ``volume_delta_weight``, ``volume_momentum_weight``,
   ``orderflow_confirmation_weight``) are not yet present on the
   ``ScoringConfig`` dataclass when this file is first created; the main
   agent adds them.  To keep this file importable immediately, we construct
   the config from V7.0's dict (which silently drops unknown keys) and then
   attach the new fields as plain attributes — the weight-manager module
   reads them via ``getattr(…, default=0.0)``, so it works regardless.
"""

from __future__ import annotations

from typing import Any, Dict

from src.edge_scanner.scoring_config import ScoringConfig, CONFIG_V7_0

# ---------------------------------------------------------------------------
# CONFIG_V7_1 — V7.0 quality gates + order-flow / volume-profile signals
# ---------------------------------------------------------------------------
# Adds four new weights that the orderflow_weight_manager module reads:
#   volume_profile_vpvr_weight  0.15  — VPVR high/low-volume-node analysis
#   volume_delta_weight         0.10  — buy/sell volume imbalance proxy
#   volume_momentum_weight      0.05  — rate of change of smoothed volume
#   orderflow_confirmation_weight 0.10  — bonus when multiple order-flow
#                                         signals agree on direction
# ---------------------------------------------------------------------------

# 1. Build base from V7.0 dict — unknown keys are silently dropped by from_dict
_v7_1_fields: Dict[str, Any] = CONFIG_V7_0.to_dict()
_v7_1_fields.update(
    {
        "version": "7.1",
        "description": (
            "V7.0_Quality_Gate +_Orderflow: "
            "Adds volume-profile (VPVR HVN/LVN), volume delta, and volume momentum "
            "as additional signal sources. Weights tuned conservatively so they "
            "supplement, not dominate, the existing altFINS + on-chain signals."
        ),
        "notes": (
            "Order-flow weights: volume_profile_vpvr=0.15, "
            "volume_delta=0.10, volume_momentum=0.05, "
            "orderflow_confirmation=0.10. "
            "All other fields identical to V7.0."
        ),
    }
)

CONFIG_V7_1: ScoringConfig = ScoringConfig.from_dict(_v7_1_fields)

# 2. Attach order-flow fields as attributes (even if not yet on the dataclass).
#    ScoringConfig is frozen, so we use object.__setattr__ to bypass the guard.
#    The weight-manager module reads them via getattr(config, field, default=0.0)
_EXTRA_ORDERFLOW_WEIGHTS = {
    "volume_profile_vpvr_weight": 0.15,
    "volume_delta_weight": 0.10,
    "volume_momentum_weight": 0.05,
    "orderflow_confirmation_weight": 0.10,
}
for _field, _val in _EXTRA_ORDERFLOW_WEIGHTS.items():
    object.__setattr__(CONFIG_V7_1, _field, _val)

# When the main agent adds these fields to ScoringConfig, update this file
# to pass them as constructor kwargs instead of attaching after construction.


# ── quick self-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== new_scoring_configs self-test ===")

    cfg = CONFIG_V7_1
    print(f"Version:           {cfg.version}")
    print(f"Description:       {cfg.description[:60]}…")
    print(f"Trend weight:      {cfg.trend_weight}")
    print(f"Volume rel weight: {cfg.volume_relative_weight}")
    print(f"Signal feed weight:{cfg.signal_feed_weight}")
    print(f"On-chain weight:   {cfg.onchain_netflow_weight}")
    print()

    # Order-flow weights (accessed via getattr, so they work at runtime)
    print("--- Order-flow weights (new in V7.1) ---")
    for field in (
        "volume_profile_vpvr_weight",
        "volume_delta_weight",
        "volume_momentum_weight",
        "orderflow_confirmation_weight",
    ):
        val = getattr(cfg, field, "NOT-SET")
        print(f"  {field}: {val}")
    print()

    # Extended scoring
    print(f"ADX weight:       {cfg.adx_weight}")
    print(f"OBV trend weight: {cfg.obv_trend_weight}")
    print(f"MT trend weight:  {cfg.medium_term_trend_weight}")
    print(f"RSI weight:       {cfg.rsi_momentum_weight}")
    print()

    # Filters
    print(f"min_abs_score: {cfg.min_abs_score}")
    print(f"min_trend:     {cfg.min_trend_abs_score}")
    print(f"min_adx:       {cfg.min_adx}")
    print(f"min_vol_rel:   {cfg.min_volume_relative}")
    print(f"min_rsi:       {cfg.min_rsi}")
    print(f"max_rsi:       {cfg.max_rsi}")

    # Cross-check: CONFIG_V7_1 should match V7.0 on all non-orderflow fields
    d1 = cfg.to_dict()
    d0 = CONFIG_V7_0.to_dict()
    differing = {}
    skip_keys = {"version", "description", "created_at", "notes"}
    orderflow_keys = {
        "volume_profile_vpvr_weight",
        "volume_delta_weight",
        "volume_momentum_weight",
        "orderflow_confirmation_weight",
    }
    for k in sorted(set(list(d1.keys()) + list(d0.keys()))):
        if k in skip_keys | orderflow_keys:
            continue
        v1 = d1.get(k)
        v0 = d0.get(k)
        if v1 != v0:
            differing[k] = (v0, v1)

    if differing:
        print(f"\n⚠  Differences from V7_0 (non-orderflow fields): {differing}")
    else:
        print("\n✓ All non-order-flow fields match V7.0 exactly.")
        print(f"  Created at: {cfg.created_at}")

    print("=== new_scoring_configs self-test complete ===")

# ---------------------------------------------------------------------------
# CONFIG_V8_0 — Dynamic Regime Adaptive Configuration
# ---------------------------------------------------------------------------
# Theory: Markets cycle through trending (bull/bear), ranging (sideways),
# and volatility regimes. No single static weight set is optimal across all
# conditions. This config enables ALL extended scoring signals with moderate
# defaults, then delegates fine-tuning to the regime detector + weight adjuster
# which scale each weight by regime-specific multipliers.
#
# Design:
#   - All 11 weight fields are set to non-zero → every signal source can
#     contribute and be dynamically adjusted.
#   - Base thresholds are permissive (|score| >= 2.0) so regime adjustment
#     determines signal quality rather than a static gate.
#   - Requires non-trend confirmation to prevent single-source noise.
#   - No market cap / coin-type restrictions — universe breadth is left to
#     the dynamic selection.
# ---------------------------------------------------------------------------

CONFIG_V8_0 = ScoringConfig(
    version="v8.0",
    description=(
        "DYNAMIC REGIME ADAPTIVE — All extended signals enabled at moderate "
        "defaults. Base thresholds permissive (|score|>=2.0) because regime "
        "adjustment + non-trend confirmation gate determine signal quality. "
        "Designed to be paired with weight_adjuster.py which scales each "
        "weight by regime-specific multipliers (1.2x trend in BULL, 1.5x RSI "
        "in SIDEWAYS, 1.5x TR/ATR in HIGH_VOL, etc.). When no regime detector "
        "is running, functions as a balanced all-signals-enabled baseline. "
        "Criteria: all 11 weights non-zero + min_abs_score=2.0 + "
        "require_non_trend_confirmation=True."
    ),
    # Core weights (moderate defaults)
    trend_weight=0.35,
    volume_relative_weight=1.5,
    volume_relative_cap=3.0,
    signal_feed_weight=2.5,
    scanner_hit_weight=2.0,
    onchain_netflow_weight=1.5,
    # Extended scoring signals (all non-zero for dynamic adjustment)
    adx_weight=0.4,
    obv_trend_weight=0.3,
    medium_term_trend_weight=0.25,
    rsi_momentum_weight=0.4,
    price_change_1w_weight=0.3,
    tr_vs_atr_weight=0.6,
    # Permissive base thresholds (regime adjuster refines quality)
    min_abs_score=2.0,
    short_min_abs_score=2.5,
    min_trend_abs_score=0.0,
    require_non_trend_confirmation=True,
    alert_min_score=7.0,
    alert_require_multi_source=True,
    # Filters (permissive — let dynamic adjustment govern quality)
    min_market_cap_usd=0.0,
    max_market_cap_usd=0.0,
    min_volume_relative=0.0,
    min_adx=0.0,
    min_rsi=0.0,
    max_rsi=100.0,
    coin_type_filter=["ANY"],
    exclude_coin_types=[],
    require_multi_timeframe_alignment=False,
    # Display types needed for extended signals
    display_types_extra=[
        "ADX", "RSI14", "OBV_TREND", "TR_VS_ATR", "ATR",
        "MEDIUM_TERM_TREND", "PRICE_CHANGE_1W",
    ],
)


# Registry of new configs for easy importing
NEW_CONFIGS: dict[str, ScoringConfig] = {
    CONFIG_V7_1.version: CONFIG_V7_1,
    CONFIG_V8_0.version: CONFIG_V8_0,
}