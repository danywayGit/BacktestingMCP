"""
Telegram alert sender — proven data only, no guessing.
If resolved signal data exists for a symbol, shows win-rate stats.
If ATR/OHLCV data exists on Binance, shows volatility-based stop & target.
Otherwise shows N/A.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone, timedelta
from typing import List, Optional

import httpx
import pandas as pd
from dotenv import load_dotenv

from .composite import CandidateScore
from .gem_scanner import GemCandidate

load_dotenv()

logger = logging.getLogger(__name__)

TELEGRAM_CHANNEL_ID = -1001482338614  # @CryptoAlertsTradingView
TELEGRAM_API_URL = "https://api.telegram.org"

ALERT_MIN_SCORE = 7.0
HIGH_CONFIDENCE_THRESHOLD = 9.0  # Score ≥ 9.0 = 🔥 high-confidence alert
ALERT_MULTI_SOURCE = True

# Dont re-alert the same symbol+config+direction within this window
ALERT_COOLDOWN_HOURS = 24  # Only alert once per signal per day
# Persistent cache across process restarts (file-backed, JSON)
_ALERT_CACHE_FILE = os.path.join(os.path.dirname(__file__), "../../data/alerted_cache.json")
_alerted_cache = {}  # In-memory cache; load from file on process start

# Load persistent cache from disk on import
import json
_cache_dir = os.path.dirname(_ALERT_CACHE_FILE)
if os.path.isdir(_cache_dir) and os.path.exists(_ALERT_CACHE_FILE):
    try:
        with open(_ALERT_CACHE_FILE) as f:
            raw = json.load(f)
        for k, v in raw.items():
            try:
                _alerted_cache[tuple(k.split("|"))] = datetime.fromisoformat(v)
            except Exception:
                pass
    except Exception:
        pass


def _save_alerted_cache():
    """Persist alert dedup cache to disk."""
    try:
        data = {
            "|".join(k): v.isoformat()
            for k, v in _alerted_cache.items()
        }
        with open(_ALERT_CACHE_FILE, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


# Default risk parameters (only used when actual ATR data exists)
RR_RATIO = 2.0        # risk 1 → reward 2
ATR_MULT_STOP = 1.5   # stop = ATR × 1.5


def _get_atr(symbol: str) -> Optional[float]:
    """Fetch ATR(14) from existing OHLCV data. Returns None if unavailable."""
    try:
        from ..core.backtesting_engine import engine
        from config.settings import TimeFrame
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=30)
        data = engine.get_data(f"{symbol}USDT", TimeFrame.H1, start, end)
        if data.empty or len(data) < 20:
            return None
        high, low, close = data["High"], data["Low"], data["Close"]
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        return float(atr) if pd.notna(atr) else None
    except Exception:
        return None


def _get_winrate(symbol: str, config_version: str = "") -> Optional[dict]:
    """Fetch resolved win-rate for this symbol, both per-config and all-configs.
    
    Returns dict with:
      - per_config: {n, wr, avg_return, avg_win, avg_loss, avg_ttr_win, avg_ttr_loss}
      - all_configs: {n, wr, avg_return, avg_win, avg_loss, avg_ttr_win, avg_ttr_loss}
    Returns None if both per-config and all-configs have < 5 trades.
    """
    def _compute_stats(signals):
        if len(signals) < 3:
            return None
        wins = [s for s in signals if s.get("outcome") == "WIN"]
        losses = [s for s in signals if s.get("outcome") == "LOSS"]
        total = len(signals)
        win_count = len(wins)
        loss_count = len(losses)
        if win_count + loss_count == 0:
            return None
        wr = win_count / (win_count + loss_count) * 100 if (win_count + loss_count) > 0 else 0
        avg_ret = sum(s.get("forward_return_pct", 0) for s in signals) / total
        avg_win = sum(s.get("forward_return_pct", 0) for s in wins) / win_count if win_count else 0
        avg_loss = sum(s.get("forward_return_pct", 0) for s in losses) / loss_count if loss_count else 0
        avg_ttr_win = sum(s.get("time_to_resolve_hours", 0) for s in wins) / win_count if win_count else 0
        avg_ttr_loss = sum(s.get("time_to_resolve_hours", 0) for s in losses) / loss_count if loss_count else 0
        return {
            "n": total, "wr": round(wr, 1),
            "avg_return": round(avg_ret, 2),
            "avg_win": round(avg_win, 2), "avg_loss": round(avg_loss, 2),
            "avg_ttr_win": round(avg_ttr_win, 1), "avg_ttr_loss": round(avg_ttr_loss, 1),
        }
    
    try:
        from ..data.database import db
        from datetime import datetime, timezone, timedelta
        since = datetime.now(timezone.utc) - timedelta(days=90)
        signals = db.get_resolved_edge_signals(since=since)
        symbol_signals = [s for s in signals if s.get("symbol") == symbol.upper()]
        if not symbol_signals:
            return None
        
        DISABLED_VERSIONS = {"2.0", "3.0", "7.0", "7.7"}
        
        per_config = _compute_stats(
            [s for s in symbol_signals if s.get("config_version") == config_version]
        ) if config_version else None
        
        active_signals = [s for s in symbol_signals if s.get("config_version") not in DISABLED_VERSIONS]
        all_configs = _compute_stats(active_signals) if len(active_signals) >= 5 else None
        
        if per_config is None and all_configs is None:
            return None
        return {"per_config": per_config, "all_configs": all_configs}
    except Exception:
        return None


def _is_multi_source(c: CandidateScore) -> bool:
    sources = 0
    comp = c.components
    trend = comp.get("altfins_trend_score", 0)
    if c.direction == "LONG" and trend >= 7:
        sources += 1
    elif c.direction == "SHORT" and trend <= -7:
        sources += 1
    feed = comp.get("altfins_signal_feed")
    if c.direction == "LONG" and feed == "BULLISH":
        sources += 1
    elif c.direction == "SHORT" and feed == "BEARISH":
        sources += 1
    # Vol relative: lowered threshold from 2.0 to 1.5 for earlier detection
    vol = comp.get("altfins_volume_relative", 1.0)
    if vol >= 1.5:
        sources += 1
    # Own volume: relative to 10MA (catches volume accumulation before breakout)
    own_vol = comp.get("volume_relative_10ma", 0)
    if own_vol >= 1.5:
        sources += 1
    # Volume accumulation: increasing volume over 3 candles (building pressure)
    if comp.get("volume_accumulation") and own_vol >= 1.0:
        sources += 1
    if comp.get("backtestingmcp_scanner_hits"):
        sources += 1
    netflow = comp.get("onchain_netflow_ratio")
    if netflow is not None:
        if c.direction == "LONG" and netflow > 0.05:
            sources += 1
        elif c.direction == "SHORT" and netflow < -0.05:
            sources += 1
    # Price above EMA20 + elevated volume = powerful early signal (2 sources)
    above_ema = comp.get("price_above_ema20")
    if above_ema and own_vol >= 1.2:
        sources += 2  # Counts as 2 sources: trend direction + volume confirmation
    return sources >= 2


def _format_alert(c: CandidateScore) -> str:
    """Alert with proven data only. N/A where no data exists."""
    direction_emoji = "🟢" if c.direction == "LONG" else "🔴"
    confidence_marker = " 🔥" if abs(c.composite_score) >= HIGH_CONFIDENCE_THRESHOLD else ""
    comp = c.components

    # Entry price
    price_str = f"${c.last_close:.4f}" if c.last_close else "N/A"

    # ATR stop & target (real OHLCV data if available)
    atr = _get_atr(c.symbol)
    if atr and c.last_close and atr > 0:
        stop_distance = atr * ATR_MULT_STOP
        _rr = c.rr_ratio or RR_RATIO
        if c.direction == "LONG":
            stop = c.last_close - stop_distance
            target = c.last_close + stop_distance * _rr
        else:
            stop = c.last_close + stop_distance
            target = c.last_close - stop_distance * _rr
        stop_str = f"${stop:.4f}"
        target_str = f"${target:.4f}"
        rr_str = f"1:{_rr}"
    else:
        stop_str = "N/A"
        target_str = "N/A"
        rr_str = "N/A"

    # Hit rate (resolved signal data — per-config + all-configs)
    wr = _get_winrate(c.symbol, config_version=c.config_version)
    if wr:
        parts = []
        if wr.get("per_config"):
            p = wr["per_config"]
            parts.append(
                f"{c.config_version}: {p['wr']}% ({p['n']}t, "
                f"⬆{p['avg_win']:+.1f}%⬇{p['avg_loss']:+.1f}%, "
                f"⏱{p['avg_ttr_win']:.0f}h/{p['avg_ttr_loss']:.0f}h)"
            )
        if wr.get("all_configs"):
            a = wr["all_configs"]
            parts.append(
                f"All: {a['wr']}% ({a['n']}t, "
                f"⬆{a['avg_win']:+.1f}%⬇{a['avg_loss']:+.1f}%)"
            )
        wr_str = " | ".join(parts)
    else:
        wr_str = "N/A (< 5 resolved trades)"

    # Build time label
    timeframe_label = comp.get("timeframe", "1h")
    coin_type = comp.get("coin_type", c.coin_type)

    # Source tags — compact for alert
    source_parts = []
    trend = comp.get("altfins_trend_score", 0)
    if abs(trend) >= 7:
        source_parts.append(f"Trend {'+' if trend > 0 else ''}{trend:.0f}")
    feed = comp.get("altfins_signal_feed")
    if feed:
        source_parts.append(f"Signal {feed.title()}")
    vol = comp.get("altfins_volume_relative", 1.0)
    if vol >= 1.5:
        source_parts.append(f"Vol {vol:.1f}x")
    scanner_hits = comp.get("backtestingmcp_scanner_hits", [])
    if scanner_hits:
        source_parts.append("TA breakout")
    netflow = comp.get("onchain_netflow_ratio")
    if netflow is not None:
        source_parts.append("On-chain")

    # Funding rate data (V8.0)
    funding_rate = comp.get("funding_rate")
    funding_momentum = comp.get("funding_momentum")
    if funding_rate is not None:
        source_parts.append(f"Funding {funding_rate:+.3f}%")
    if funding_momentum is not None:
        source_parts.append(f"Mom {funding_momentum:+.2f}")

    sources_str = " · ".join(source_parts) if source_parts else "—"

    # Position sizing based on score tier × R:R adjustment
    score = abs(c.composite_score)
    # Base size from confidence tier
    if score >= 9.0:
        base_pct = 2.0
    elif score >= 8.0:
        base_pct = 1.0
    elif score >= 7.0:
        base_pct = 0.5
    else:
        base_pct = 0.0
    # R:R adjustment: higher R:R → larger position (capped)
    # Kelly-inspired: at R:R 2.0 = 1×, at R:R 1.0 = 0.5×, at R:R 3.0 = 1.5×
    rr_adjust = min(c.rr_ratio or RR_RATIO, 4.0) / 2.0 if c.rr_ratio else 1.0
    risk_pct = round(base_pct * rr_adjust, 1)
    risk_pct_str = f"{risk_pct}" if risk_pct > 0 else "—"
    sizing_note = f" ×{rr_adjust:.1f}R" if rr_adjust != 1.0 else ""
    sizing_icon = "🔒" if risk_pct > 0 else ""

    lines = [
        f"{direction_emoji} *{c.symbol}* — {c.direction} ({c.config_version}){confidence_marker}",
        f"┌ Entry: `{price_str}`",
        f"├ Stop:  `{stop_str}`",
        f"├ 🎯 Tgt: `{target_str}`  R:R `{rr_str}`",
        f"├ Position Risk: {risk_pct_str}% {sizing_icon}{sizing_note}",
        f"└ Time:  `{timeframe_label}` · Type: `{coin_type}`",
        f"Score: `{abs(c.composite_score):+.2f}`  |  {sources_str}",
        f"Resolved: {wr_str}",
    ]
    return "\n".join(lines)


def send_alerts(candidates: List[CandidateScore], dry_run: bool = False) -> int:
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        logger.warning("TELEGRAM_BOT_TOKEN not set — skipping alerts")
        return 0

    triggered = [
        c for c in candidates
        if c.direction is not None
        and abs(c.composite_score) >= ALERT_MIN_SCORE
        and (not ALERT_MULTI_SOURCE or _is_multi_source(c))
    ]

    # Dedup: dont re-alert same symbol+config+direction within cooldown window
    # Also skip if an unresolved signal already exists for this symbol+config
    now = datetime.now(timezone.utc)
    deduped = []
    for c in triggered:
        key = (c.symbol, c.config_version, c.direction)
        # Check cooldown
        last_alerted = _alerted_cache.get(key)
        if last_alerted and (now - last_alerted).total_seconds() < ALERT_COOLDOWN_HOURS * 3600:
            continue
        # Check if already has an unresolved signal that was SENT to the bot
        # (a just-logged PENDING signal from this scan should still alert)
        try:
            from ..data.database import db
            unresolved = db.get_pending_edge_signal(
                symbol=c.symbol, direction=c.direction,
                config_version=c.config_version
            )
            if unresolved and unresolved.get("webhook_sent_at"):
                continue
        except Exception:
            pass
        _alerted_cache[key] = now
        _save_alerted_cache()
        deduped.append(c)
    triggered = deduped

    if not triggered:
        return 0

    sent = 0
    for c in triggered:
        message = _format_alert(c)
        if dry_run:
            logger.info("DRY RUN alert:\n%s", message)
            sent += 1
            continue
        try:
            resp = httpx.post(
                f"{TELEGRAM_API_URL}/bot{bot_token}/sendMessage",
                json={
                    "chat_id": TELEGRAM_CHANNEL_ID,
                    "text": message,
                    "parse_mode": "Markdown",
                },
                timeout=10.0,
            )
            resp.raise_for_status()
            sent += 1
            logger.info("Alert sent for %s (score=%+.2f)", c.symbol, c.composite_score)
        except Exception as exc:
            logger.error("Failed to send alert for %s: %s", c.symbol, exc)

    return sent


def send_gem_report(candidates: List[GemCandidate]) -> int:
    """Send a gem scanner report to Telegram."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        logger.warning("TELEGRAM_BOT_TOKEN not set — skipping gem alerts")
        return 0

    from .gem_scanner import format_gem_report
    message = format_gem_report(candidates, top_n=30)

    try:
        resp = httpx.post(
            f"{TELEGRAM_API_URL}/bot{bot_token}/sendMessage",
            json={
                "chat_id": TELEGRAM_CHANNEL_ID,
                "text": message,
                "parse_mode": "Markdown",
            },
            timeout=10.0,
        )
        resp.raise_for_status()
        logger.info("Gem report sent to Telegram")
        return 1
    except Exception as exc:
        logger.error("Failed to send gem report: %s", exc)
        return 0
