"""Persist composite-scanner signals and resolve their forward outcomes.

This is the self-validation loop: every signal the scanner flags gets
logged with an entry price, then checked again after `horizon_hours` to
see whether price actually moved the predicted direction. Without this,
the composite score is just an opinion -- this is what turns it into a
measured win-rate.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Optional

import pandas as pd

from config.settings import TimeFrame
from ..core.backtesting_engine import engine
from ..data.database import db
from .composite import CandidateScore
from ..edge_scanner.scoring_config import ALL_CONFIGS

logger = logging.getLogger(__name__)

DEFAULT_HORIZON_HOURS = 24
# Forward moves smaller than this are treated as noise, not a real win/loss.
MIN_MOVE_PCT = 0.3
# Round-trip transaction costs (entry fee + exit fee + slippage)
# Binance USDT-M Futures taker: 0.04% per trade
COST_BPS = 11  # 4bps entry + 4bps exit + 3bps slippage = 11bps = 0.11%


def _get_atr(pair: str, timeframe: TimeFrame) -> float:
    """Fetch ATR(14) from recent OHLCV data. Returns 0.0 if unavailable."""
    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=30)
        data = engine.get_data(pair, timeframe, start, end)
        if data.empty or len(data) < 20:
            return 0.0
        high, low, close = data["High"], data["Low"], data["Close"]
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        return float(atr) if pd.notna(atr) else 0.0
    except Exception:
        return 0.0


def _compute_atr_stop_target(
    symbol: str,
    direction: str,
    entry_price: float,
    timeframe: TimeFrame,
    atr_stop_mult: float,
    rr_ratio: float,
) -> tuple[float, float]:
    """Compute ATR-based stop and target prices for a signal.

    Returns (target_price, stop_price). If ATR is unavailable, returns
    fallback using a fixed 2% expected move.
    """
    pair = f"{symbol.upper()}USDT"
    atr = _get_atr(pair, timeframe)
    if atr > 0:
        stop_distance = atr * atr_stop_mult
        if direction == "LONG":
            stop_price = round(entry_price - stop_distance, 8)
            target_price = round(entry_price + stop_distance * rr_ratio, 8)
        else:
            stop_price = round(entry_price + stop_distance, 8)
            target_price = round(entry_price - stop_distance * rr_ratio, 8)
    else:
        move = entry_price * 0.02
        if direction == "LONG":
            stop_price = round(entry_price - move * atr_stop_mult, 8)
            target_price = round(entry_price + move * rr_ratio, 8)
        else:
            stop_price = round(entry_price + move * atr_stop_mult, 8)
            target_price = round(entry_price - move * rr_ratio, 8)
    return target_price, stop_price


def log_signals(scores: List[CandidateScore], timeframe: TimeFrame, horizon_hours: int = DEFAULT_HORIZON_HOURS) -> int:
    """Persist the actionable (LONG/SHORT) signals from a scan cycle.

    Deduplicates per config version: if a PENDING signal already exists for the
    same symbol + direction + config_version, it's updated in-place rather than
    creating a duplicate. Different config versions may have separate PENDING
    signals for the same symbol — this enables fair win-rate comparison between
    configs during evolution analysis.

    For each signal, ATR-based target and stop prices are computed from OHLCV
    data and stored alongside the entry price. At resolution time, the actual
    HIGH/LOW of the tracking window is checked against these levels — a signal
    resolves as WIN if price hit the target, LOSS if it hit the stop, and FLAT
    only if neither level was reached during the full horizon.
    """
    logged = 0
    updated = 0
    for score in scores:
        if score.direction is None or score.last_close is None:
            continue

        # Compute ATR-based stop and target prices for this signal
        config = ALL_CONFIGS.get(score.config_version)
        if config is not None:
            atr_stop_mult = config.atr_stop_mult
            rr_ratio = config.rr_ratio
        else:
            # Fallback to sensible defaults if config version not found
            atr_stop_mult = 1.5
            rr_ratio = 2.0
        target_price, stop_price = _compute_atr_stop_target(
            score.symbol, score.direction, score.last_close, timeframe,
            atr_stop_mult, rr_ratio,
        )

        # Dedup per config-version: same symbol+direction+config = update, not insert
        existing = db.get_pending_edge_signal(score.symbol, score.direction, score.config_version)
        if existing is not None:
            db.update_edge_signal(
                signal_id=existing["id"],
                composite_score=score.composite_score,
                entry_price=score.last_close,
                components=score.components,
            )
            updated += 1
            continue

        db.insert_edge_signal(
            symbol=score.symbol,
            pair=score.pair,
            timeframe=timeframe.value,
            direction=score.direction,
            composite_score=score.composite_score,
            components=score.components,
            entry_price=score.last_close,
            horizon_hours=horizon_hours,
            config_version=score.config_version,
            coin_type=score.coin_type,
            target_price=target_price,
            stop_price=stop_price,
        )
        logged += 1

    if updated > 0:
        logger.info("Dedup: updated %d existing PENDING signals (same config, same symbol)", updated)
    return logged


def _fetch_window_data(pair: str, timeframe: TimeFrame, start: datetime, end: datetime):
    """Fetch OHLCV data over a time window.

    Falls back to Binance live API when local DB has no data.
    Returns (dataframe, high, low, close) or (None, None, None, None) on failure.
    """
    try:
        data = engine.get_data(pair, timeframe, start, end)
        if data.empty:
            # Fallback: fetch from Binance Futures klines API directly
            import httpx
            import pandas as pd
            import math
            import time as _time
            # Map TimeFrame to Binance interval strings
            tf_name = timeframe.name if hasattr(timeframe, 'name') else str(timeframe)
            interval_map = {
                'H1': '1h', 'H4': '4h', 'H12': '12h',
                'D1': '1d', 'M15': '15m', 'M30': '30m',
                'M5': '5m', 'M1': '1m',
            }
            interval = interval_map.get(tf_name, '1h')
            # Binance symbol format: remove slash, keep USDT suffix
            api_symbol = pair.replace('/', '')
            if not api_symbol.endswith('USDT'):
                base = pair.split('/')[0] if '/' in pair else pair.split('USDT')[0] if 'USDT' in pair else pair
                api_symbol = base + 'USDT'
            params = {
                'symbol': api_symbol,
                'interval': interval,
                'startTime': int(math.floor(start.timestamp() * 1000)),
                'endTime': int(math.ceil(end.timestamp() * 1000)),
                'limit': 500,
            }
            url = 'https://fapi.binance.com/fapi/v1/klines'
            # Try once, with rate-limit retry
            for attempt in range(2):
                try:
                    resp = httpx.get(url, params=params, timeout=10.0)
                    if resp.status_code == 429:
                        # Rate limited — read Retry-After header
                        retry_after = resp.headers.get('Retry-After')
                        if retry_after:
                            wait = float(retry_after) + 5
                        else:
                            wait = 65  # Binance default: 60s + 5s buffer
                        logger.warning(
                            "Binance rate limited for %s, waiting %.0fs (attempt %d/2)",
                            api_symbol, wait, attempt + 1,
                        )
                        _time.sleep(wait)
                        continue
                    if resp.status_code != 200:
                        logger.debug(
                            "Binance klines returned %d for %s: %s",
                            resp.status_code, api_symbol, resp.text[:200],
                        )
                        # Try Binance Spot API as fallback (coins may be Spot-only)
                        if attempt == 0:
                            url = 'https://api.binance.com/api/v3/klines'
                            continue
                        break
                    klines = resp.json()
                    if klines and len(klines) > 0:
                        data = pd.DataFrame(klines).iloc[:, :6]
                        data.columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
                        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
                        data.set_index('timestamp', inplace=True)
                        data = data.astype(float)
                    break  # Success or non-retryable error
                except httpx.TimeoutException:
                    logger.debug("Binance klines timeout for %s (attempt %d/2)", api_symbol, attempt + 1)
                    if attempt == 0:
                        _time.sleep(1)
                        continue
                    break
                except Exception:
                    logger.debug("Binance klines failed for %s (attempt %d/2)", api_symbol, attempt + 1)
                    if attempt == 0:
                        _time.sleep(1)
                        continue
                    break
        if data.empty:
            return None, None, None, None
        return (data, float(data["High"].max()), float(data["Low"].min()), float(data["Close"].iloc[-1]))
    except Exception:
        return None, None, None, None


def _find_first_hit_hours(data, entry_time: datetime, level: float, direction: str, hit_type: str) -> float:
    """Find the first bar where target or stop was breached.

    Returns the hours from entry_time to the first hit bar.
    hit_type: 'target' (check High for LONG, Low for SHORT)
              'stop'  (check Low for LONG, High for SHORT)
    """
    if data is None or data.empty:
        return 0.0
    for idx, row in data.iterrows():
        bar_time = idx if isinstance(idx, datetime) else entry_time
        if hit_type == "target":
            if direction == "LONG" and row["High"] >= level:
                return (bar_time - entry_time).total_seconds() / 3600
            elif direction == "SHORT" and row["Low"] <= level:
                return (bar_time - entry_time).total_seconds() / 3600
        else:  # stop
            if direction == "LONG" and row["Low"] <= level:
                return (bar_time - entry_time).total_seconds() / 3600
            elif direction == "SHORT" and row["High"] >= level:
                return (bar_time - entry_time).total_seconds() / 3600
    return 0.0


def resolve_due_signals() -> int:
    """Resolve every PENDING signal whose horizon has elapsed.

    Uses target/stop-based resolution: checks if price hit the target or stop
    DURING the tracking window (using HIGH/LOW), not just at the endpoint.
    This correctly captures trades that hit profit target mid-window then
    retraced — those would be WINs under the old close-only logic.
    """
    from ..integrations.binance_symbols import is_on_binance_futures
    resolved = 0
    now = datetime.now(timezone.utc)
    for signal in db.get_pending_edge_signals():
        # Skip symbols not on Binance Futures — prevents API timeouts
        if not is_on_binance_futures(signal.get("symbol", "")):
            db.resolve_edge_signal(signal["id"], 0.0, 0.0, "FLAT", time_to_resolve_hours=24.0)
            resolved += 1
            continue
        entry_time = datetime.fromisoformat(signal["entry_time"])
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)
        due_at = entry_time + timedelta(hours=signal["horizon_hours"])
        if now < due_at:
            continue

        timeframe = TimeFrame(signal["timeframe"])
        target_price = signal.get("target_price")
        stop_price = signal.get("stop_price")
        entry_price = signal["entry_price"]
        direction = signal["direction"]

        # Fetch the full OHLCV window from entry to now
        df, window_high, window_low, exit_price = _fetch_window_data(
            signal["pair"], timeframe, entry_time, due_at,
        )

        if window_high is None:
            logger.warning("Could not resolve signal %s (%s): no OHLCV data", signal["id"], signal["pair"])
            continue

        outcome = "FLAT"  # default
        raw_return_pct = 0.0
        directional_return_pct = 0.0
        time_to_resolve_hours = 0.0

        # Check target/stop hits DURING the window
        if target_price is not None:
            if direction == "LONG" and window_high >= target_price:
                directional_return_pct = ((target_price - entry_price) / entry_price) * 100
                outcome = "WIN"
                time_to_resolve_hours = _find_first_hit_hours(df, entry_time, target_price, direction, "target")
            elif direction == "SHORT" and window_low <= target_price:
                directional_return_pct = ((entry_price - target_price) / entry_price) * 100
                outcome = "WIN"
                time_to_resolve_hours = _find_first_hit_hours(df, entry_time, target_price, direction, "target")

        if stop_price is not None and outcome == "FLAT":
            if direction == "LONG" and window_low <= stop_price:
                directional_return_pct = ((stop_price - entry_price) / entry_price) * 100
                outcome = "LOSS"
                time_to_resolve_hours = _find_first_hit_hours(df, entry_time, stop_price, direction, "stop")
            elif direction == "SHORT" and window_high >= stop_price:
                directional_return_pct = ((entry_price - stop_price) / entry_price) * 100
                outcome = "LOSS"
                time_to_resolve_hours = _find_first_hit_hours(df, entry_time, stop_price, direction, "stop")

        # If neither target nor stop was hit, fall back to endpoint comparison
        if outcome == "FLAT" and exit_price is not None:
            raw_return_pct = (exit_price - entry_price) / entry_price * 100
            directional_return_pct = raw_return_pct if direction == "LONG" else -raw_return_pct
            if directional_return_pct > MIN_MOVE_PCT:
                outcome = "WIN"
            elif directional_return_pct < -MIN_MOVE_PCT:
                outcome = "LOSS"
            else:
                outcome = "FLAT"

        # Subtract round-trip transaction costs from return
        cost_pct = COST_BPS / 100.0  # e.g. 11 bps = 0.11%
        net_return_pct = directional_return_pct - cost_pct

        # Re-classify outcome based on net (after-costs) return
        if outcome == "WIN":
            if net_return_pct <= 0:
                outcome = "FLAT"
                directional_return_pct = 0.0
            else:
                directional_return_pct = net_return_pct
        elif outcome == "LOSS":
            directional_return_pct = directional_return_pct - cost_pct  # Costs make loss worse
        elif outcome == "FLAT" and abs(net_return_pct) > MIN_MOVE_PCT:
            if net_return_pct > 0:
                outcome = "WIN"
                directional_return_pct = net_return_pct
            else:
                outcome = "LOSS"
                directional_return_pct = net_return_pct
        else:
            directional_return_pct = net_return_pct  # Still FLAT, but record the true net

        db.resolve_edge_signal(signal["id"], exit_price or 0.0, directional_return_pct, outcome,
                               time_to_resolve_hours=time_to_resolve_hours)
        resolved += 1
    return resolved


def performance_report(group_by: str = "symbol", min_n: int = 5, since_days: int = 90, breakeven: bool = True) -> pd.DataFrame:
    """Win-rate / avg return grouped by symbol, hour-of-day, direction, config version, or coin type.

    When group_by='config', adds breakeven WR column based on each config's
    rr_ratio. Breakeven WR = 1 / (1 + R:R). Configs with WR above breakeven
    are profitable; configs below are losing money despite raw win rate.
    """
    since = datetime.now(timezone.utc) - timedelta(days=since_days)
    rows = db.get_resolved_edge_signals(since=since)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)

    if group_by == "hour":
        df["group"] = df["entry_time"].dt.hour
    elif group_by == "direction":
        df["group"] = df["direction"]
    elif group_by == "config":
        df["group"] = df.get("config_version", "v1.0")
    elif group_by == "coin_type":
        df["group"] = df.get("coin_type", "OTHER")
    else:
        df["group"] = df["symbol"]

    df["is_win"] = (df["outcome"] == "WIN").astype(int)
    summary = (
        df.groupby("group")
        .agg(n=("outcome", "size"), win_rate=("is_win", "mean"), avg_return_pct=("forward_return_pct", "mean"))
        .reset_index()
    )
    summary["win_rate"] = (summary["win_rate"] * 100).round(1)
    summary["avg_return_pct"] = summary["avg_return_pct"].round(3)
    summary = summary[summary["n"] >= min_n].sort_values("win_rate", ascending=False)

    # Add breakeven WR column when grouping by config
    if breakeven and group_by == "config":
        from src.edge_scanner.scoring_config import ALL_CONFIGS
        config_rr = {}
        for cfg_obj in ALL_CONFIGS.values():
            ver_str = str(cfg_obj.version)
            config_rr[ver_str] = cfg_obj.rr_ratio
            config_rr["v" + ver_str] = cfg_obj.rr_ratio

        be_rows = []
        for _, row in summary.iterrows():
            g = row["group"]
            rr = config_rr.get(g)
            if rr is None:
                be_rows.append(None)
            else:
                be_wr = round(1 / (1 + rr) * 100, 1)
                be_rows.append(be_wr)
        summary["be_wr"] = be_rows
        summary["status"] = summary.apply(
            lambda r: "🟢" if r["be_wr"] and r["win_rate"] >= r["be_wr"] else ("🔴" if r["be_wr"] else "—"),
            axis=1,
        )

    return summary
