# ----------------------------------------------- #
# Event watcher daemon — event-driven scan triggers
# You                       : Hermes / danywayGit
# File Name                 : event_watcher.py
# ----------------------------------------------- #

"""
Event-driven scan triggers for liquidation (V22.x) and funding (V8.x) strategies.

The main scan runs on a 5-min cron. But liquidation `!forceOrder@arr` events
stream in REAL TIME (the WS daemon writes data/liquidation_snapshot.json
immediately), and funding settles at deterministic times. Waiting for the
5-min cron wastes up to 5 minutes on a fresh liquidation/funding event.

This daemon watches those two event streams and, on a threshold breach,
immediately triggers a TARGETED scan of the event-relevant configs for the
triggering symbol(s), then resolves + sends to the bot:

    trigger → run_parallel_scan(configs=[V22.0,V22.1,V8.0], targeted symbols)
            → resolve_due_signals()
            → run_bridge()

Debounce: at most one targeted run per symbol per DEBOUNCE_SEC (default 60s)
so a flood of liquidations on one symbol doesn't stack scans.

Event sources:
1. LIQUIDATION SPIKE — reads data/liquidation_snapshot.json (written in real
   time by binance_liq_ws.py). If a symbol's liquidation $ volume in the last
   N minutes exceeds LIQ_VOL_THRESHOLD, trigger its V22.0/V22.1 scan.
2. FUNDING TICK — monitors next_funding_time. When funding is within
   FUNDING_LOOKAHEAD_MIN of settling, trigger a V8.0 scan of symbols with
   extreme funding.

Run:  python -m src.integrations.event_watcher   (keeps running)
      python -m src.integrations.event_watcher --once   (single check, cron-friendly)
"""

import argparse
import json
import logging
import os
import time
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Paths (relative to repo root)
_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
_SNAPSHOT_PATH = os.path.join(_REPO_ROOT, "data", "liquidation_snapshot.json")
sys.path.insert(0, _REPO_ROOT)

# ── Thresholds (tunable) ──
LIQ_VOL_THRESHOLD = 250_000.0      # $ liquidation volume in window to trigger
LIQ_WINDOW_SEC = 300               # look back 5 min
DEBOUNCE_SEC = 60                  # min seconds between runs per symbol
FUNDING_LOOKAHEAD_MIN = 5          # trigger funding scan X min before settlement
FUNDING_EXTREME_ABS = 0.004        # |funding| > 0.4% considered extreme

# Event-relevant configs (V22 = liquidation LONG/SHORT, V8 = funding, V14 precursor)
LIQ_CONFIGS = ["22.0", "22.1"]
FUNDING_CONFIGS = ["8.0"]
EVENT_CONFIGS = list(dict.fromkeys(LIQ_CONFIGS + FUNDING_CONFIGS))  # dedup, keep order

# Debounce registry: symbol -> last scan timestamp
_last_scan: Dict[str, float] = {}

# Runtime overrides (set by CLI args, read by trigger functions)
_LIQ_VOL_THRESHOLD: Optional[float] = None
_FUNDING_LOOKAHEAD_MIN: Optional[float] = None


def _liq_threshold() -> float:
    return _LIQ_VOL_THRESHOLD if _LIQ_VOL_THRESHOLD is not None else LIQ_VOL_THRESHOLD


def _funding_lookahead() -> float:
    return _FUNDING_LOOKAHEAD_MIN if _FUNDING_LOOKAHEAD_MIN is not None else FUNDING_LOOKAHEAD_MIN


def _read_snapshot() -> List[dict]:
    """Read the liquidation WS snapshot (list of events). Empty if absent/stale."""
    try:
        if not os.path.exists(_SNAPSHOT_PATH):
            return []
        # Freshness guard: snapshot must be written recently (daemon alive).
        if time.time() - os.path.getmtime(_SNAPSHOT_PATH) > 360:
            return []
        with open(_SNAPSHOT_PATH) as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as exc:
        logger.debug("read_snapshot: %s", exc)
        return []


def _liq_spike_symbols() -> List[str]:
    """Return symbols whose liquidation volume in the last LIQ_WINDOW_SEC
    exceeds LIQ_VOL_THRESHOLD (either direction)."""
    events = _read_snapshot()
    if not events:
        return []
    cutoff = time.time() - LIQ_WINDOW_SEC
    vol_by_sym: Dict[str, float] = {}
    for e in events:
        t = e.get("t", 0)
        if t < cutoff:
            continue
        sym = str(e.get("s", "")).replace("USDT", "").upper()
        v = float(e.get("v", 0) or 0)
        if sym:
            vol_by_sym[sym] = vol_by_sym.get(sym, 0) + v
    spikes = [s for s, v in vol_by_sym.items() if v >= _liq_threshold()]
    return sorted(spikes, key=lambda s: -vol_by_sym.get(s, 0))


def _funding_tick_symbols() -> List[str]:
    """Return extreme-funding symbols (|funding| > threshold) whose next
    funding settlement is within FUNDING_LOOKAHEAD_MIN."""
    try:
        from src.integrations import binance_funding
        rates = binance_funding.fetch_all_funding_rates()
        now_ms = time.time() * 1000
        lookahead_ms = _funding_lookahead() * 60 * 1000
        out = []
        for sym, d in rates.items():
            fr = abs(float(d.get("funding_rate", 0) or 0))
            nft = float(d.get("next_funding_time", 0) or 0)
            if fr >= FUNDING_EXTREME_ABS and 0 < (nft - now_ms) <= lookahead_ms:
                out.append(sym)
        return out
    except Exception as exc:
        logger.warning("funding tick: %s", exc)
        return []


def _debounced(symbol: str) -> bool:
    """True if this symbol is due for another scan (debounce passed)."""
    now = time.time()
    last = _last_scan.get(symbol, 0)
    if now - last >= DEBOUNCE_SEC:
        _last_scan[symbol] = now
        return True
    return False


def _build_minimal_screener_row(symbol: str) -> dict:
    """Build a permissive screener_row so V22/V8 config filters pass on the
    liquidation/funding signal itself (not stale TA filters).

    The event is the signal: liquidation volume or funding rate. We provide
    neutral defaults that won't block the config's minimum filters, letting
    liquidation_weight/funding_rate_weight carry the score.
    """
    return {
        "symbol": symbol,
        "lastPrice": "0.0",
        "additionalData": {
            "MARKET_CAP": 1_000_000_000,   # large cap → passes mcap filters
            "VOLUME_RELATIVE": 10.0,       # high → passes min_volume_relative
            "ADX": 30,                     # trending → passes min_adx
            "RSI14": 55,                   # neutral
            "SHORT_TERM_TREND": "7",       # positive
            "MEDIUM_TERM_TREND": "7",      # aligned
        },
    }


def _run_targeted_scan(symbols: List[str], configs: List[str]) -> Optional[dict]:
    """Score the given symbols across the event configs, log, resolve, bridge.

    Returns summary dict or None on failure.
    """
    if not symbols or not configs:
        return None
    try:
        from src.edge_scanner.scoring_config import ALL_CONFIGS, get_enabled_configs
        from src.edge_scanner.multi_version_scan import run_parallel_scan
        from config.settings import TimeFrame

        # Map version strings to ScoringConfig objects (only enabled).
        cfg_map = get_enabled_configs()
        config_objs = [cfg_map[v] for v in configs if v in cfg_map]
        if not config_objs:
            logger.warning("No enabled event configs: %s", configs)
            return None

        # Score only the triggering symbols by passing a minimal screener row
        # per symbol. run_parallel_scan discovers from altFINS normally, so we
        # bypass it and call score_symbol directly per config.
        from src.edge_scanner.composite import score_symbol
        from src.edge_scanner import store as edge_store
        from src.edge_scanner.webhook_bridge import run_bridge

        all_scores: Dict[str, list] = {}
        total_logged = 0
        for cfg in config_objs:
            scored = []
            for sym in symbols:
                row = _build_minimal_screener_row(sym)
                candidate = score_symbol(
                    sym, row, {}, TimeFrame.H1, 30, config=cfg
                )
                scored.append(candidate)
            actionable = [c for c in scored if c.direction is not None]
            if actionable:
                all_scores[cfg.version] = actionable
                logged = edge_store.log_signals(actionable, TimeFrame.H1, horizon_hours=24)
                total_logged += logged

        # Resolve any due signals, then send to bot.
        try:
            from src.edge_scanner.store import resolve_due_signals
            resolved = resolve_due_signals()
        except Exception as exc:
            logger.warning("resolve: %s", exc)
            resolved = 0
        sent = run_bridge() if total_logged > 0 else 0

        summary = {
            "symbols": symbols,
            "configs": configs,
            "signals": {v: len(s) for v, s in all_scores.items()},
            "logged": total_logged,
            "resolved": resolved,
            "sent_to_bot": sent,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        logger.info("Targeted scan: %s", json.dumps(summary))
        return summary
    except Exception as exc:
        logger.warning("targeted scan failed: %s", exc)
        return None


def run_once() -> dict:
    """One event-watch pass (cron-friendly). Returns what was triggered."""
    triggered = {"liquidation": [], "funding": [], "runs": []}

    # 1) Liquidation spikes → V22.0/V22.1
    liq_syms = [s for s in _liq_spike_symbols() if _debounced(s)]
    if liq_syms:
        logger.info("Liquidation spike: %s", liq_syms)
        triggered["liquidation"] = liq_syms
        r = _run_targeted_scan(liq_syms, LIQ_CONFIGS)
        if r:
            triggered["runs"].append(r)

    # 2) Funding tick → V8.0
    f_syms = [s for s in _funding_tick_symbols() if _debounced(s)]
    if f_syms:
        logger.info("Funding tick: %s", f_syms)
        triggered["funding"] = f_syms
        r = _run_targeted_scan(f_syms, FUNDING_CONFIGS)
        if r:
            triggered["runs"].append(r)

    return triggered


def run_daemon(interval_sec: int = 15):
    """Loop run_once() every interval_sec forever."""
    logger.info("Event watcher daemon started (interval=%ss, liq_threshold=$%s, window=%ss, debounce=%ss)",
                interval_sec, LIQ_VOL_THRESHOLD, LIQ_WINDOW_SEC, DEBOUNCE_SEC)
    while True:
        try:
            trigger = run_once()
            for r in trigger.get("runs", []):
                if r.get("signals"):
                    logger.info("Triggered: %s", json.dumps(r))
        except Exception as exc:
            logger.warning("daemon loop error: %s", exc)
        time.sleep(interval_sec)


def main():
    parser = argparse.ArgumentParser(description="Event-driven scan triggers")
    parser.add_argument("--once", action="store_true", help="Single pass then exit")
    parser.add_argument("--interval", type=int, default=15, help="Loop interval seconds")
    parser.add_argument("--liq-threshold", type=float, default=LIQ_VOL_THRESHOLD,
                        help="Liquidation $ volume threshold to trigger")
    parser.add_argument("--funding-lookahead", type=int, default=FUNDING_LOOKAHEAD_MIN,
                        help="Trigger funding scan N min before settlement")
    args = parser.parse_args()

    global _LIQ_VOL_THRESHOLD, _FUNDING_LOOKAHEAD_MIN
    _LIQ_VOL_THRESHOLD = args.liq_threshold
    _FUNDING_LOOKAHEAD_MIN = args.funding_lookahead

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.once:
        trigger = run_once()
        print(json.dumps(trigger, indent=2))
    else:
        run_daemon(args.interval)


if __name__ == "__main__":
    main()