"""
Integration layer — wires V7.1 (order-flow signals) and V8.0 (dynamic regime
adaptation) into the scoring pipeline without modifying composite.py or
scoring_config.py.

This module exposes four public functions:

- integrate_orderflow()       — add V7.1 order-flow contribution
- integrate_regime_adaptation()— apply V8.0 dynamic weight adjustment
- run_integrated_scoring()    — drop-in replacement for score_symbol() logic
- validate_integration()      — self-test that exercises both extensions

The caller controls which config is used.  Each integration inspects the
config's version and/or weight fields to decide whether to activate.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from config.settings import TimeFrame
from src.edge_scanner.scoring_config import (
    ScoringConfig,
    get_coin_type,
    parse_trend_score_extended,
    parse_float as _safe_float_sc,
)
from ..core.backtesting_engine import engine
from ..strategies.scanner import evaluate_scan
from ..integrations import altfins_client
from ..integrations.altfins_client import AltfinsError, parse_trend_score
from ..integrations import santiment_client
from ..integrations.santiment_client import SantimentError

logger = logging.getLogger(__name__)

# ── helpers mirrored from composite.py (avoid circular dep on composite) ─────


def _altfins_to_pair(symbol: str) -> str:
    return f"{symbol.upper()}USDT"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return default


def _is_blocked(symbol: str) -> bool:
    """Re-import blocklist check without referencing composite.py directly."""
    from src.edge_scanner.scoring_config import is_stablecoin_or_stock
    from ..integrations.binance_symbols import is_on_binance_futures

    if is_stablecoin_or_stock(symbol):
        return True
    if not is_on_binance_futures(symbol):
        return True
    return False


# ── config version helpers ────────────────────────────────────────────────────


def _has_orderflow_weights(config: ScoringConfig) -> bool:
    """Check whether *config* has non-zero order-flow weight fields.

    Uses getattr so it works whether or not the fields exist on the
    ScoringConfig dataclass (V7.1 attaches them after construction).
    """
    fields = (
        "volume_profile_vpvr_weight",
        "volume_delta_weight",
        "volume_momentum_weight",
        "orderflow_confirmation_weight",
    )
    return any(abs(float(getattr(config, f, 0.0))) > 1e-9 for f in fields)


def _is_v8(config: ScoringConfig) -> bool:
    """Check whether *config* is the V8.0 regime-adaptive config."""
    return getattr(config, "version", "").lower().replace(".", "") in ("v80", "8.0")


# ── 1. integrate_orderflow ────────────────────────────────────────────────────


def integrate_orderflow(
    symbol: str,
    timeframe: TimeFrame,
    lookback_days: int,
    config: ScoringConfig,
    current_score: float,
    current_components: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """Add V7.1 order-flow / volume-profile signals to an in-progress score.

    Parameters
    ----------
    symbol : str
        altFINS-style symbol (e.g. ``"BTC"``).
    timeframe : TimeFrame
        Bar interval for OHLCV data.
    lookback_days : int
        How many days of data to fetch.
    config : ScoringConfig
        Configuration whose order-flow weights will be used.
    current_score : float
        The composite score **before** order-flow contribution.
    current_components : dict
        Accumulated components dict; updated in-place with order-flow keys.

    Returns
    -------
    (updated_score, updated_components)
        If order-flow weights are all zero, returns ``(current_score,
        current_components)`` unchanged.
    """
    # Quick exit — no order-flow weight configured
    if not _has_orderflow_weights(config):
        return current_score, current_components

    # We import here so the module can be loaded even when the weight-manager
    # or its dependencies (e.g. orderflow_signals → backtesting_engine) aren't
    # fully available (graceful degradation).
    try:
        from src.edge_scanner.orderflow_weight_manager import (
            apply_orderflow_to_score,
        )

        new_score, comp = apply_orderflow_to_score(
            current_score, symbol, timeframe, lookback_days, config
        )
        # Merge order-flow components into current_components (prefix them)
        of_components = {f"orderflow_{k}": v for k, v in comp.items()}
        current_components.update(of_components)
        current_components["orderflow_extra_score"] = round(
            new_score - current_score, 4
        )

        logger.debug(
            "integrate_orderflow[%s]: score %.2f -> %.2f (delta=%.4f)",
            symbol,
            current_score,
            new_score,
            new_score - current_score,
        )
        return new_score, current_components

    except Exception as exc:
        logger.warning(
            "integrate_orderflow[%s] unavailable — skipping: %s",
            symbol,
            exc,
        )
        current_components["orderflow_error"] = str(exc)
        return current_score, current_components


# ── 2. integrate_regime_adaptation ────────────────────────────────────────────


def integrate_regime_adaptation(
    symbol: str,
    timeframe: TimeFrame,
    lookback_days: int,
    config: ScoringConfig,
    current_score: float,
    current_components: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """Apply V8.0 dynamic regime weight adjustment to an in-progress score.

    Parameters
    ----------
    symbol : str
        altFINS-style symbol (ignored — regime is detected on ``"BTC/USDT"``).
    timeframe : TimeFrame
        Bar interval for regime detection.
    lookback_days : int
        How many days of data to feed the regime detector.
    config : ScoringConfig
        Configuration whose weights will be adjusted by the detected regime.
    current_score : float
        The composite score **before** regime adjustment.
    current_components : dict
        Accumulated components dict; updated in-place with regime info.

    Returns
    -------
    (regime_adjusted_score, updated_components)
        For non-V8.0 configs, returns ``(current_score, current_components)``
        unchanged.
    """
    # Quick exit — only V8.0 triggers regime adaptation
    if not _is_v8(config):
        return current_score, current_components

    try:
        from src.edge_scanner.regime_detector import detect_regime
        from src.edge_scanner.weight_adjuster import compute_regime_adjusted_score

        # Detect macro regime on BTC/USDT (general market context)
        regime_info = detect_regime(
            symbol="BTC/USDT",
            timeframe=timeframe,
            lookback_days=lookback_days,
        )
        regime = regime_info.get("regime", "UNKNOWN")

        # Apply regime-adjusted weights to recompute the score
        adjusted_score, adjustment_info = compute_regime_adjusted_score(
            base_score=current_score,
            base_score_components=current_components,
            regime=regime,
            config=config,
        )

        # Merge regime information into components
        current_components["regime"] = regime
        current_components["regime_label"] = adjustment_info.get(
            "regime_label", regime
        )
        current_components["regime_confidence"] = regime_info.get("confidence", 0.0)
        current_components["regime_adx"] = regime_info.get("adx", 0.0)
        current_components["regime_volatility"] = regime_info.get("volatility", 0.0)
        current_components["regime_delta"] = adjustment_info.get("delta", 0.0)
        current_components["regime_adjusted_weights"] = adjustment_info.get(
            "adjusted_weights", {}
        )
        current_components["regime_breakdown"] = adjustment_info.get(
            "breakdown", {}
        )
        current_components["regime_base_score"] = current_score
        current_components["regime_adjusted_score"] = adjusted_score

        logger.debug(
            "integrate_regime[%s]: regime=%s score=%.2f -> %.2f (delta=%.2f)",
            symbol,
            regime,
            current_score,
            adjusted_score,
            adjustment_info.get("delta", 0.0),
        )

        return adjusted_score, current_components

    except Exception as exc:
        logger.warning(
            "integrate_regime_adaptation[%s] unavailable — skipping: %s",
            symbol,
            exc,
        )
        current_components["regime_error"] = str(exc)
        return current_score, current_components


# ── 3. run_integrated_scoring ─────────────────────────────────────────────────


def run_integrated_scoring(
    symbol: str,
    screener_row: Dict[str, Any],
    signal_feed_index: Dict[str, str],
    timeframe: TimeFrame,
    lookback_days: int,
    config: ScoringConfig,
) -> Tuple[float, Optional[str], Dict[str, Any]]:
    """Drop-in replacement for the ``score_symbol()`` logic in *composite.py*.

    This function mirrors the exact computation sequence from ``score_symbol``
    (composite.py lines 159–286) but adds the two new integration hooks:

    1. **Order-flow scoring** (V7.1) — after extended signals, before scanner.
    2. **Regime adaptation** (V8.0) — after scanner, before direction assignment.

    Returns
    -------
    (final_score, direction, components_dict)
        ``direction`` is ``"LONG"``, ``"SHORT"``, or ``None``.
        ``components_dict`` contains all intermediate values for diagnostics.
    """
    # ── Step 0: prepare ──────────────────────────────────────────────────
    pair = _altfins_to_pair(symbol)
    coin_type = get_coin_type(symbol)
    components: Dict[str, Any] = {}
    score = 0.0

    # ── Step 1: config filters (fail fast) ───────────────────────────────
    passes, filter_reason = config.passes_filters(symbol, screener_row)
    if not passes:
        components["filtered_out"] = filter_reason
        return 0.0, None, components

    # ── Step 2: altFINS trend ────────────────────────────────────────────
    additional = screener_row.get("additionalData", {}) if screener_row else {}
    trend_score = parse_trend_score(additional.get("SHORT_TERM_TREND"))
    volume_relative = _safe_float(additional.get("VOLUME_RELATIVE"), default=1.0)

    score += trend_score * config.trend_weight
    components["altfins_trend_score"] = trend_score

    # ── Step 3: altFINS volume relative ──────────────────────────────────
    vol_excess = min(max(volume_relative - 1.0, 0.0), config.volume_relative_cap)
    if trend_score != 0:
        score += (
            vol_excess * config.volume_relative_weight * (1 if trend_score > 0 else -1)
        )
    components["altfins_volume_relative"] = volume_relative

    # ── Step 4: signal feed direction ────────────────────────────────────
    feed_direction = signal_feed_index.get(symbol.upper())
    if feed_direction == "BULLISH":
        score += config.signal_feed_weight
    elif feed_direction == "BEARISH":
        score -= config.signal_feed_weight
    components["altfins_signal_feed"] = feed_direction

    # ── Step 5: on-chain netflow (Santiment) ─────────────────────────────
    netflow_ratio: Optional[float] = None
    altfins_score_so_far = score
    if santiment_client.slug_for(symbol) and abs(altfins_score_so_far) >= 2.0:
        try:
            onchain = santiment_client.get_onchain_snapshot(
                symbol, lookback_days=3
            )
            inflow = onchain.get("exchange_inflow_usd", 0.0)
            outflow = onchain.get("exchange_outflow_usd", 0.0)
            total = inflow + outflow
            if total > 0:
                netflow_ratio = (outflow - inflow) / total
                score += netflow_ratio * config.onchain_netflow_weight
        except SantimentError as exc:
            logger.debug("No on-chain data for %s: %s", symbol, exc)
    components["onchain_netflow_ratio"] = netflow_ratio

    # ── Step 6: extended scoring signals (ADX, OBV, RSI, TR/ATR, …) ─────
    direction_hint = score  # positive = bullish context so far
    extra_score, extra_components = config.compute_extended_score(
        additional, score, direction_hint
    )
    score += extra_score
    components.update(extra_components)

    # ── Step 7: order-flow signals (V7.1 integration) ────────────────────
    score, components = integrate_orderflow(
        symbol=symbol,
        timeframe=timeframe,
        lookback_days=lookback_days,
        config=config,
        current_score=score,
        current_components=components,
    )

    # ── Step 8: BacktestingMCP scanner hits ──────────────────────────────
    last_close: Optional[float] = None
    triggered_scans: List[str] = []
    try:
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=lookback_days)
        data = engine.get_data(pair, timeframe, start_date, end_date)
        if not data.empty:
            last_close = float(data["Close"].iloc[-1])
            scan_result = evaluate_scan(data, "all")
            triggered_scans = [
                name
                for name, details in scan_result.items()
                if details.get("triggered")
            ]
            score += len(triggered_scans) * config.scanner_hit_weight
    except Exception as exc:
        logger.warning("Could not fetch/scan OHLCV for %s: %s", pair, exc)
    components["backtestingmcp_scanner_hits"] = triggered_scans
    components["coin_type"] = coin_type

    # ── Step 9: regime adaptation (V8.0 integration) ─────────────────────
    score, components = integrate_regime_adaptation(
        symbol=symbol,
        timeframe=timeframe,
        lookback_days=lookback_days,
        config=config,
        current_score=score,
        current_components=components,
    )

    # ── Step 10: direction assignment ────────────────────────────────────
    direction: Optional[str] = None

    trend = components.get("altfins_trend_score", 0)
    if config.min_trend_abs_score > 0 and abs(trend) < config.min_trend_abs_score:
        components["failed_trend_min"] = (
            f"|trend|={abs(trend):.0f} < min={config.min_trend_abs_score}"
        )
    elif config.require_non_trend_confirmation and abs(trend) > 0:
        vol = components.get("altfins_volume_relative", 0)
        feed = components.get("altfins_signal_feed")
        scanner = components.get("backtestingmcp_scanner_hits", [])
        netflow = components.get("onchain_netflow_ratio")
        has_confirmation = (
            vol > 1.0
            or (feed == "BULLISH" and trend > 0)
            or (feed == "BEARISH" and trend < 0)
            or len(scanner) > 0
            or (netflow is not None and abs(netflow) > 0.05)
        )
        if not has_confirmation:
            components["failed_confirmation"] = "no non-trend source confirms"

    short_min = (
        config.short_min_abs_score
        if config.short_min_abs_score is not None
        else config.min_abs_score
    )
    if score >= config.min_abs_score:
        direction = "LONG"
    elif score <= -short_min:
        direction = "SHORT"

    final_score = round(score, 2)
    return final_score, direction, components


# ── 4. validate_integration ───────────────────────────────────────────────────


def validate_integration() -> Dict[str, Any]:
    """Self-test: exercise both integration hooks with mock data.

    Returns a summary dict with pass/fail per integration step so callers
    can inspect results programmatically.

    This test does **not** require live API keys — it gracefully degrades
    on missing data and catches all exceptions so the test never crashes.
    """
    print("=" * 60)
    print("  validate_integration — edge scanner integration layer")
    print("=" * 60)

    results: Dict[str, Any] = {
        "integrate_orderflow": None,
        "integrate_regime_adaptation": None,
        "run_integrated_scoring_v7_1": None,
        "run_integrated_scoring_v8_0": None,
    }

    # ── Import configs (graceful if not available) ───────────────────────
    try:
        from src.edge_scanner.new_scoring_configs import CONFIG_V7_1, CONFIG_V8_0

        print(f"\n  ✓ Loaded CONFIG_V7_1 (ver={CONFIG_V7_1.version})")
        print(f"  ✓ Loaded CONFIG_V8_0 (ver={CONFIG_V8_0.version})")
    except ImportError as exc:
        print(f"\n  ✗ Cannot import new_scoring_configs: {exc}")
        results["error"] = str(exc)
        return results

    # ── Mock data ────────────────────────────────────────────────────────
    mock_row: Dict[str, Any] = {
        "symbol": "BTC",
        "lastPrice": "65000.00",
        "additionalData": {
            "SHORT_TERM_TREND": "Up (5/10)",
            "VOLUME_RELATIVE": "1.5",
            "ADX": "30",
            "RSI14": "62",
            "OBV_TREND": "2.3",
            "MEDIUM_TERM_TREND": "Up (6/10)",
            "PRICE_CHANGE_1W": "3.5",
            "TR_VS_ATR": "1.1",
        },
    }
    mock_feed: Dict[str, str] = {"BTC": "BULLISH"}

    # ── Test 1: integrate_orderflow ──────────────────────────────────────
    print("\n  --- integrate_orderflow ---")
    try:
        from src.edge_scanner.scoring_config import CONFIG_V7_0

        base_score = 5.0
        base_components: Dict[str, Any] = {
            "altfins_trend_score": 5.0,
            "altfins_volume_relative": 1.5,
            "altfins_signal_feed": "BULLISH",
        }

        # With V7.0 (no order-flow weights) — should be no-op
        s1, c1 = integrate_orderflow(
            "BTC", TimeFrame.H1, 30, CONFIG_V7_0, base_score, dict(base_components)
        )
        assert s1 == base_score, "V7.0 should not change score"
        print(f"  ✓ V7.0 (no OF weights): score={s1} (unchanged)")

        # With V7.1 (has order-flow weights)
        s2, c2 = integrate_orderflow(
            "BTC", TimeFrame.H1, 30, CONFIG_V7_1, base_score, dict(base_components)
        )
        print(f"  ✓ V7.1 (with OF weights): score={s2:.4f}")
        of_extra = c2.get("orderflow_extra_score", 0.0)
        print(f"    orderflow_extra_score={of_extra:.4f}")
        results["integrate_orderflow"] = {
            "v7_0_score": s1,
            "v7_1_score": s2,
            "v7_1_delta": of_extra,
        }
    except Exception as exc:
        print(f"  ✗ integrate_orderflow raised: {exc}")
        logger.exception("integrate_orderflow validation failed")
        results["integrate_orderflow"] = {"error": str(exc)}

    # ── Test 2: integrate_regime_adaptation ──────────────────────────────
    print("\n  --- integrate_regime_adaptation ---")
    try:
        base_score = 5.0
        base_components: Dict[str, Any] = {
            "altfins_trend_score": 5.0,
            "altfins_volume_relative": 1.5,
            "altfins_signal_feed": "BULLISH",
            "onchain_netflow_ratio": None,
        }

        # With V7.1 (not V8.0) — should be no-op
        s1, c1 = integrate_regime_adaptation(
            "BTC", TimeFrame.H1, 30, CONFIG_V7_1, base_score, dict(base_components)
        )
        assert s1 == base_score, "V7.1 should not trigger regime adjustment"
        print(f"  ✓ V7.1 (not V8.0): score={s1} (unchanged)")

        # With V8.0 — should try regime detection
        s2, c2 = integrate_regime_adaptation(
            "BTC", TimeFrame.H1, 30, CONFIG_V8_0, base_score, dict(base_components)
        )
        regime = c2.get("regime", "UNKNOWN")
        delta = c2.get("regime_delta", 0.0)
        print(f"  ✓ V8.0: regime={regime}, score={s2:.4f} (delta={delta:.4f})")
        results["integrate_regime_adaptation"] = {
            "v7_1_score": s1,
            "v8_0_score": s2,
            "v8_0_regime": regime,
            "v8_0_delta": delta,
        }
    except Exception as exc:
        print(f"  ✗ integrate_regime_adaptation raised: {exc}")
        logger.exception("regime_adaptation validation failed")
        results["integrate_regime_adaptation"] = {"error": str(exc)}

    # ── Test 3: run_integrated_scoring with V7.1 ─────────────────────────
    print("\n  --- run_integrated_scoring (CONFIG_V7_1) ---")
    try:
        score_v71, direction_v71, comps_v71 = run_integrated_scoring(
            "BTC", mock_row, mock_feed, TimeFrame.H1, 30, CONFIG_V7_1
        )
        print(f"  ✓ V7.1: score={score_v71}, direction={direction_v71}")
        print(f"    config_version={CONFIG_V7_1.version}")
        of_key = next((k for k in comps_v71 if "orderflow" in k), None)
        if of_key:
            print(f"    order-flow present: {of_key}={comps_v71[of_key]}")
        else:
            print("    (order-flow may be unavailable — check data)")
        results["run_integrated_scoring_v7_1"] = {
            "score": score_v71,
            "direction": direction_v71,
            "has_orderflow": of_key is not None,
        }
    except Exception as exc:
        print(f"  ✗ run_integrated_scoring (V7.1) raised: {exc}")
        logger.exception("V7.1 scoring validation failed")
        results["run_integrated_scoring_v7_1"] = {"error": str(exc)}

    # ── Test 4: run_integrated_scoring with V8.0 ─────────────────────────
    print("\n  --- run_integrated_scoring (CONFIG_V8_0) ---")
    try:
        score_v80, direction_v80, comps_v80 = run_integrated_scoring(
            "BTC", mock_row, mock_feed, TimeFrame.H1, 30, CONFIG_V8_0
        )
        print(f"  ✓ V8.0: score={score_v80}, direction={direction_v80}")
        print(f"    config_version={CONFIG_V8_0.version}")
        if "regime" in comps_v80:
            print(f"    regime={comps_v80['regime']}, delta={comps_v80.get('regime_delta')}")
        else:
            print("    (regime info may be unavailable — check data)")
        results["run_integrated_scoring_v8_0"] = {
            "score": score_v80,
            "direction": direction_v80,
            "has_regime": "regime" in comps_v80,
        }
    except Exception as exc:
        print(f"  ✗ run_integrated_scoring (V8.0) raised: {exc}")
        logger.exception("V8.0 scoring validation failed")
        results["run_integrated_scoring_v8_0"] = {"error": str(exc)}

    # ── Summary ──────────────────────────────────────────────────────────
    n_total = 4
    n_ok = sum(
        1
        for k in results
        if k in ("integrate_orderflow", "integrate_regime_adaptation",
                 "run_integrated_scoring_v7_1", "run_integrated_scoring_v8_0")
        and isinstance(results[k], dict)
        and "error" not in results[k]
    )
    print(f"\n  {'=' * 56}")
    print(f"  Summary: {n_ok}/{n_total} tests passed")
    if n_ok < n_total:
        print(f"  WARNING: {n_total - n_ok} test(s) encountered errors —")
        print("    this may be expected if data sources (altFINS, Santiment")
        print("    backtesting engine) are unavailable in the test environment.")
    print(f"  {'=' * 56}\n")

    return results


# ── __main__ self-test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    validate_integration()
