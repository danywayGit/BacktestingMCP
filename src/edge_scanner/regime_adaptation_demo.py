#!/usr/bin/env python3
"""
Regime Adaptation Demo / Verification Script.

Demonstrates the dynamic weight adjustment pipeline:
  1. Detect current market regime for BTC/USDT using real OHLCV data.
  2. Detect multi-symbol consensus (BTC, ETH, SOL).
  3. Show regime-adjusted weights for CONFIG_V8_0.
  4. Compute a regime-adjusted score for a hypothetical signal.
  5. Print a human-readable summary.

Usage:
    cd ~/BacktestingMCP
    venv/bin/python -m src.edge_scanner.regime_adaptation_demo
"""

from __future__ import annotations

import logging
import sys
import os

# Ensure BacktestingMCP root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("regime_adaptation_demo")


def main():
    print("=" * 72)
    print("  Regime Adaptation Demo — Dynamic Weight Adjustment")
    print("=" * 72)
    print()

    # ── Phase 1: Single-symbol regime detection ────────────────────────
    print("─── Phase 1: Single-Symbol Regime Detection (BTC/USDT, H1) ───")
    print()
    from src.edge_scanner.regime_detector import detect_regime, detect_market_context
    from config.settings import TimeFrame

    btc_regime = detect_regime(symbol="BTC/USDT", timeframe=TimeFrame.H1, lookback_days=30)
    _print_regime("BTC/USDT", btc_regime)
    print()

    # ── Phase 2: Multi-symbol consensus ────────────────────────────────
    print("─── Phase 2: Multi-Symbol Consensus ───")
    print()
    context = detect_market_context(timeframe=TimeFrame.H1, lookback_days=30)
    print(f"  Consensus regime:  {context['consensus_regime']}")
    print(f"  Volatility ctx:    {context['volatility_context']}")
    print()
    for sym, res in context["symbols"].items():
        _print_regime(sym, res)
    print()

    # ── Phase 3: Adjusted weights per regime ───────────────────────────
    print("─── Phase 3: Regime-Adjusted Weights for CONFIG_V8_0 ───")
    print()
    from src.edge_scanner.new_scoring_configs import CONFIG_V8_0
    from src.edge_scanner.weight_adjuster import (
        get_adjusted_weights,
        REGIME_LABELS,
    )

    regimes_to_show = ["BULL_TRENDING", "BEAR_TRENDING", "SIDEWAYS", "HIGH_VOLATILITY", "LOW_VOLATILITY"]
    weight_fields = [
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
    ]

    # Collect all adjusted weights
    regime_adjustments = {}
    for regime in regimes_to_show:
        regime_adjustments[regime] = get_adjusted_weights(CONFIG_V8_0, regime)

    # Print header
    header = f"{'Weight':<30s}"
    for r in regimes_to_show:
        header += f"  {r:<18s}"
    print(header)
    print("-" * len(header))

    for field in weight_fields:
        base_val = getattr(CONFIG_V8_0, field, 0.0)
        row = f"{field:<30s}"
        for r in regimes_to_show:
            adj = regime_adjustments[r][field]
            sign = "+" if adj > base_val else (" " if adj == base_val else "-")
            row += f"  {adj:<8.4f} ({sign:<1s})  "
        print(row)

    print()
    print(f"  Base config weights shown in parentheses: ( ) = unchanged, (+) = boosted, (-) = reduced")
    print()

    # ── Phase 4: Regime-adjusted score computation ─────────────────────
    print("─── Phase 4: Regime-Adjusted Score (Hypothetical Example) ───")
    print()
    from src.edge_scanner.weight_adjuster import compute_regime_adjusted_score

    # Hypothetical base score components
    example_components = {
        "altfins_trend_score": 6.0,        # Up (6/10)
        "altfins_volume_relative": 1.8,     # 80% above average
        "altfins_signal_feed": "BULLISH",
        "onchain_netflow_ratio": 0.3,       # mild outflow (bullish)
        "adx": 30.0,
        "rsi14": 58.0,
    }

    base_score = 8.5  # hypothetical composite score

    print(f"  Base score (hypothetical):  {base_score:+.2f}")
    print(f"  Components:")
    for k, v in example_components.items():
        print(f"    {k:<35s} = {v}")
    print()

    for regime in regimes_to_show:
        adj_score, info = compute_regime_adjusted_score(
            base_score, example_components, regime, CONFIG_V8_0
        )
        label = REGIME_LABELS.get(regime, regime)
        delta = info["delta"]
        direction = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        print(f"  {label:<30s}  base={base_score:+.2f} → adj={adj_score:+.2f}  delta={delta:+.2f} {direction}")

    print()
    print("─── Adjustment Breakdown (BULL_TRENDING) ───")
    adj_score, info = compute_regime_adjusted_score(
        base_score, example_components, "BULL_TRENDING", CONFIG_V8_0
    )
    for k, v in info["breakdown"].items():
        if v != 0.0:
            print(f"    {k:<35s} = {v:+.4f}")
    print()

    # ── Phase 5: Summary ───────────────────────────────────────────────
    print("─── Summary ───")
    print()
    print(f"  Detected BTC regime:       {btc_regime['regime']}")
    print(f"  Consensus regime:          {context['consensus_regime']}")
    print(f"  BTC ADX:                   {btc_regime['adx']}")
    print(f"  BTC volatility (ATR%):     {btc_regime['volatility']}%")
    print(f"  BTC price trend (EMA50):   {btc_regime['price_trend']:+.2f}%")
    print(f"  Regime confidence:         {btc_regime['confidence']}")
    print(f"  Active config w/ dynamic:  {CONFIG_V8_0.version}")
    print(f"  Extended signals enabled:  all 11 weights non-zero")
    print()

    if btc_regime["regime"] != "UNKNOWN" and btc_regime["confidence"] > 0.3:
        # Show the adjusted weights for the actual detected regime
        detected = btc_regime["regime"]
        actual_weights = get_adjusted_weights(CONFIG_V8_0, detected)
        label = REGIME_LABELS.get(detected, detected)
        print(f"  Recommended dynamic weights for actual regime ({label}):")
        for field in weight_fields:
            base = getattr(CONFIG_V8_0, field, 0.0)
            adj = actual_weights[field]
            if abs(adj - base) > 0.001:
                pct = ((adj / base) - 1) * 100 if base != 0 else 0
                print(f"    {field:<35s} {base:<8.4f} → {adj:<8.4f}  ({pct:+.1f}%)")
            else:
                print(f"    {field:<35s} {base:<8.4f} → {adj:<8.4f}  (unchanged)")

    print()
    print("=" * 72)
    print("  Demo complete.")
    print("=" * 72)


def _print_regime(label: str, result: dict):
    """Pretty-print a regime detection result."""
    from src.edge_scanner.weight_adjuster import REGIME_LABELS

    regime = result["regime"]
    label_str = REGIME_LABELS.get(regime, regime)
    print(f"  {label:<12s}  {label_str:<30s}  "
          f"ADX={result['adx']:<6.1f}  "
          f"ATR%={result['volatility']:<6.2f}%  "
          f"trend={result['price_trend']:<+7.2f}%  "
          f"conf={result['confidence']:<.2f}")


if __name__ == "__main__":
    main()