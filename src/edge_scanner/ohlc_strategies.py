"""OHLC-native signal engine for OHLC-driven ScoringConfigs (V23/V24 family).

The composite path (composite.py / multi_version_scan.py) scores symbols from
altFINS screener rows + on-chain + funding — it is irrelevant for pure-OHLC
strategies like MA-cross or RSI mean-reversion, which are defined entirely by
price/volume history. This module generates those signals directly from OHLCV
and returns the same CandidateScore objects the rest of the pipeline consumes
(store.log_signals, webhook_bridge), so an OHLC config drops into the existing
flow with zero downstream changes.

Timeframe handling: strategy_search.py proved that raw multi-hour storage is
untrustworthy (4h column inconsistent) — we always resample trusted 1h bars to
the config's TF. Entry fires on the latest CLOSED resampled bar's signal, using
the strategy's +1/-1 rule; SL/TP are left to store.py from the config's
atr_stop_mult / rr_ratio.

Dedup: a Strategy must not re-emit the SAME cross-bar signal every scan cycle.
We skip a (config,symbol,direction) if a row was already SENT for that
(config,symbol) within a short window, so a fresh cross sends once and a repeat
of the same bar is suppressed. Genuinely new crosses (a different bar) pass.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

from config.settings import TimeFrame
from .scoring_config import ScoringConfig, get_coin_type
from .composite import CandidateScore, _pair_to_altfins

logger = logging.getLogger(__name__)

# Min hours a `sent` signal (any outcome) blocks re-firing the same setup.
_SENT_DEDUP_HOURS = 6.0

# Conviction score assigned to a fired OHLC signal (LONG/SHORT both positive,
# direction carried separately — matches the score convention).
_DEFAULT_SCORE = 8.0


# ── indicators (numpy, ported 1:1 from scripts/strategy_search.py) ─────────
def _ema(x, n: int) -> np.ndarray:
    x = np.asarray(x, float)
    out = np.empty_like(x)
    a = 2.0 / (n + 1)
    s = x[0]
    out[0] = s
    for i in range(1, len(x)):
        s = a * x[i] + (1 - a) * s
        out[i] = s
    return out


def _rsi(close, n: int = 14) -> np.ndarray:
    c = np.asarray(close, float)
    d = np.diff(c)
    gain = np.where(d > 0, d, 0.0)
    loss = np.where(d < 0, -d, 0.0)
    ag = _ema(gain, n)
    al = _ema(loss, n)
    rs = np.where(al == 0, 100.0, ag / np.where(al == 0, 1e-12, al))
    return np.concatenate([[50.0], 100 - 100 / (1 + rs)])


def _atr(h, l, c, n: int = 14) -> np.ndarray:
    h = np.asarray(h, float)
    l = np.asarray(l, float)
    c = np.asarray(c, float)
    pc = np.roll(c, 1)
    pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return np.concatenate([[tr[0]], _ema(tr, n)[1:]])


def _donchian_hi(c, n: int = 20) -> np.ndarray:
    c = np.asarray(c, float)
    out = np.full_like(c, np.nan)
    for i in range(n - 1, len(c)):
        out[i] = c[i - n + 1:i + 1].max()
    return out


def _donchian_lo(c, n: int = 20) -> np.ndarray:
    c = np.asarray(c, float)
    out = np.full_like(c, np.nan)
    for i in range(n - 1, len(c)):
        out[i] = c[i - n + 1:i + 1].min()
    return out


def _rolling_mean(x, n: int) -> np.ndarray:
    x = np.asarray(x, float)
    out = np.full_like(x, np.nan)
    s = np.cumsum(np.insert(x, 0, 0))
    out[n - 1:] = (s[n:] - s[:-n]) / n
    return out


# ── strategy types: return +1/-1/0 per bar (identical to strategy_search) ──
def _strat_trend(o, h, l, c, v, p=None):
    e = _ema(c, (p or {}).get("ema", 50))
    return np.where(c > e, 1.0, np.where(c < e, -1.0, 0.0))


def _strat_ma_cross(o, h, l, c, v, p=None):
    p = p or {}
    f = _ema(c, p.get("fast", 20))
    s = _ema(c, p.get("slow", 50))
    sig = np.zeros(len(c))
    for i in range(1, len(c)):
        if f[i - 1] <= s[i - 1] and f[i] > s[i]:
            sig[i] = 1
        elif f[i - 1] >= s[i - 1] and f[i] < s[i]:
            sig[i] = -1
    return sig


def _strat_meanrev(o, h, l, c, v, p=None):
    p = p or {}
    r = _rsi(c, p.get("rsi", 14))
    e = _ema(c, 50)
    sig = np.zeros(len(c))
    for i in range(1, len(c)):
        if r[i] < p.get("oversold", 30) and c[i] > e[i]:
            sig[i] = 1
        elif r[i] > p.get("overbought", 70) and c[i] < e[i]:
            sig[i] = -1
    return sig


def _strat_breakout(o, h, l, c, v, p=None):
    p = p or {}
    n = p.get("chan", 20)
    hi = _donchian_hi(c, n)
    lo = _donchian_lo(c, n)
    vr = p.get("vol", 0.0)
    av = _rolling_mean(np.asarray(v, float), 10)
    sig = np.zeros(len(c))
    for i in range(1, len(c)):
        vok = True if vr <= 0 else (i < len(av) and np.isfinite(av[i]) and v[i] > (av[i] * vr))
        if c[i] > hi[i - 1] and vok:
            sig[i] = 1
        elif c[i] < lo[i - 1] and vok:
            sig[i] = -1
    return sig


def _strat_momentum(o, h, l, c, v, p=None):
    p = p or {}
    roc_back = p.get("roc", 5)
    thr = p.get("thr", 1.0)
    vr = p.get("vol", 1.2)
    roc = np.zeros(len(c))
    pc = np.roll(c, roc_back)
    pc[:roc_back] = c[:roc_back]
    roc = (c - pc) / np.where(pc == 0, 1e-12, pc) * 100
    av = _rolling_mean(np.asarray(v, float), 10)
    avv = av.copy()
    for i in range(len(v)):
        if not np.isfinite(avv[i]):
            avv[i] = v[i]
    sig = np.zeros(len(c))
    for i in range(1, len(c)):
        if i < len(avv) and v[i] > (avv[i] * vr) and roc[i] > thr:
            sig[i] = 1
        elif i < len(avv) and v[i] > (avv[i] * vr) and roc[i] < -thr:
            sig[i] = -1
    return sig


def _strat_rangeratio(o, h, l, c, v, p=None):
    p = p or {}
    K = p.get("K", 60)
    hh_max = p.get("hh", 0.45)
    ll_min = p.get("ll", 0.50)
    sig = np.zeros(len(c))
    for i in range(K + 2, len(c)):
        seg_h = h[i - K:i]
        seg_l = l[i - K:i]
        hh = int((np.asarray(seg_h[1:]) > np.asarray(seg_h[:-1])).sum())
        ll = int((np.asarray(seg_l[1:]) < np.asarray(seg_l[:-1])).sum())
        hh_r = hh / float(K)
        ll_r = ll / float(K)
        if hh_r < hh_max or ll_r >= ll_min:
            sig[i] = 1
    return sig


TYPES = {
    "trend": _strat_trend,
    "ma_cross": _strat_ma_cross,
    "meanrev": _strat_meanrev,
    "breakout": _strat_breakout,
    "momentum": _strat_momentum,
    "rangeratio": _strat_rangeratio,
}

# TF hours -> TimeFrame for ATR lookups store.py needs on the config's own TF.
TF_HOURS_TO_ENUM = {1: TimeFrame.H1, 4: TimeFrame.H4, 12: TimeFrame.H12, 24: TimeFrame.D1}


def resample(rows: List[tuple], hours: int) -> List[tuple]:
    """rows: list[(ts,o,h,l,c,v)] sorted by ts. Bucket aligned to `hours`x3600."""
    out = []
    cur = []
    period = hours * 3600
    cur_key = None
    for ts, o, h, l, c, v in rows:
        key = ts - (ts % period)
        if cur_key is None:
            cur_key = key
        if key == cur_key:
            cur.append((ts, o, h, l, c, v))
        else:
            out.append((cur_key, cur[0][1], max(x[2] for x in cur),
                        min(x[3] for x in cur), cur[-1][4], sum(x[5] for x in cur)))
            cur = [(ts, o, h, l, c, v)]
            cur_key = key
    if cur:
        out.append((cur_key, cur[0][1], max(x[2] for x in cur),
                    min(x[3] for x in cur), cur[-1][4], sum(x[5] for x in cur)))
    return out


# ├─ OHLC data access ─────────────────────────────────────────────────────────
def _load_1h(conn, symbol: str, base_symbol: str) -> List[tuple]:
    """Load trusted 1h bars for a symbol from market_data, merging both era
    blocks (recent `SYMUSDT` ~97d and older block that OTHER tools write under
    `SYMUSDTUSDT` — the exact dual-key quirk documented in the backtesting
    skill). strategy_search.load_full() reads the same table the same way."""
    seen = set()
    merged: List[tuple] = []
    for key in (symbol, base_symbol + "USDTUSDT"):
        try:
            rows = conn.execute(
                "SELECT timestamp, open, high, low, close, volume FROM market_data "
                "WHERE symbol=? AND timeframe='1h' ORDER BY timestamp", (key,)).fetchall()
        except Exception:
            rows = []
        for r in rows:
            ts = r[0]
            # de-dupe same timestamp across both keys
            try:
                t_int = int(ts)
            except (TypeError, ValueError):
                t_int = int(r[2]) if str(r[0])[0].isdigit() else hash(str(ts))
            keyed = (t_int, symbol)
            if keyed in seen:
                continue
            seen.add(keyed)
            # normalize ts to epoch seconds
            try:
                e = float(ts)
                ts_out = int(e)
            except (TypeError, ValueError):
                import datetime as _dt
                date_part = str(ts).replace("T", " ").split(".")[0]
                try:
                    ts_out = int(_dt.datetime.strptime(date_part, "%Y-%m-%d %H:%M:%S").replace(tzinfo=_dt.timezone.utc).timestamp())
                except Exception:
                    continue
            merged.append((ts_out, float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])))
    merged.sort(key=lambda x: x[0])
    # de-dup consecutive same-ts rows
    out = []
    last = None
    for m in merged:
        if m[0] != last:
            out.append(m)
            last = m[0]
    return out


def _fetch_ohlc_tf(pair: str, tf_hours: int, lookback_days: int) -> List[tuple]:
    """Trusted approach: read 1h bars from market_data (exactly like the OOS
    validator), resample to tf_hours. No reliance on engine.get_data's symbol
    resolution, which has no coverage for these user-scoped pairs."""
    base = pair[:-4] if pair.upper().endswith("USDT") else pair
    import sqlite3, os
    _db = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "data", "crypto.db")
    try:
        conn = sqlite3.connect(_db)
        rows = _load_1h(conn, pair, base)
        conn.close()
    except Exception as exc:
        logger.warning("_fetch_ohlc_tf: db read failed for %s: %s", pair, exc)
        return []
    if len(rows) < 6:
        return []
    # trim to lookback window
    last_ts = rows[-1][0]
    cutoff = last_ts - lookback_days * 86400
    rows = [r for r in rows if r[0] >= cutoff]
    return resample(rows, tf_hours)


def _recently_sent(symbol: str, config_version: str, direction: str, hours: float = _SENT_DEDUP_HOURS) -> bool:
    """True if this (config,symbol,direction) has an UNRESOLVED sent signal
    recently (a live/just-sent position). Once resolved (WIN/LOSS/FLAT written)
    the slot is free — a genuinely NEW transition may fire again. A sent row
    whose outcome is still NULL means the bot may be holding the position."""
    try:
        import sqlite3, os, datetime as _dt
        db = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                          "data", "crypto.db")
        if not os.path.exists(db):
            return False
        conn = sqlite3.connect(db)
        rows = conn.execute(
            "SELECT webhook_sent_at FROM edge_signals WHERE config_version=? AND symbol=? AND direction=? "
            "AND webhook_sent_at IS NOT NULL AND outcome IS NULL ORDER BY id DESC LIMIT 3",
            (config_version, symbol, direction)).fetchall()
        conn.close()
        now = _dt.datetime.now(_dt.timezone.utc)
        for r in rows:
            try:
                sent = _dt.datetime.fromisoformat(r[0])
                if sent.tzinfo is None:
                    sent = sent.replace(tzinfo=_dt.timezone.utc)
                age_h = (now - sent).total_seconds() / 3600.0
                if age_h <= hours:
                    return True
                break  # rows are DESC by id; once we pass the window, stop
            except Exception:
                return True  # unparseable → treat as recent to be safe
        return False
    except Exception as exc:
        logger.debug("dedup lookup failed %s: %s", symbol, exc)
        return False


def generate_for_config(cfg: ScoringConfig, lookback_days: int = 120) -> List[CandidateScore]:
    """Generate OHLC-native candidate scores for an OHLC config.

    Only fires when the strategy returns a direction on the LATEST closed bar,
    and only if that (config,symbol,direction) wasn't just sent. Returns
    CandidateScore objects (direction set, score=_DEFAULT_SCORE) for each fired
    symbol — identical to what composite produces so the rest of the pipeline is
    unchanged.
    """
    strat = TYPES.get(cfg.ohlc_strategy)
    if strat is None:
        logger.warning("config %s: unknown ohlc_strategy %r", cfg.version, cfg.ohlc_strategy)
        return []
    tf_hours = cfg.ohlc_tf_hours or 1
    symbols = cfg.symbol_whitelist or []
    if not symbols:
        logger.warning("config %s (ohlc): empty symbol_whitelist — nothing to scan", cfg.version)
        return []

    scores: List[CandidateScore] = []
    for sym in symbols:
        pair = f"{sym.upper()}USDT"
        rows = _fetch_ohlc_tf(pair, tf_hours, lookback_days)
        if len(rows) < 60:
            logger.debug("config %s: insufficient data for %s (%d bars)", cfg.version, pair, len(rows))
            continue
        o = [x[1] for x in rows]
        h = [x[2] for x in rows]
        l = [x[3] for x in rows]
        c = [x[4] for x in rows]
        v = [x[5] for x in rows]
        oa = np.asarray(o, float); ha = np.asarray(h, float)
        la = np.asarray(l, float); ca = np.asarray(c, float)
        va = np.asarray(v, float)
        sig = strat(oa, ha, la, ca, va, cfg.ohlc_params)
        # Fire only on a FRESH transition: ma_cross/trend return a PERSISTENT
        # +1 while the fast EMA stays above slow (until the next cross), so
        # checking the raw last signal would re-emit the SAME cross every
        # 15-min scan. Only a bar where the signal just turned nonzero from
        # zero is a genuinely new setup. Reset-by-direction not needed: a
        # transition is any 0->(+/-1) edge on the latest bar.
        last = sig[-1]
        if last == 0 or sig[-2] == last:
            continue
        direction = "LONG" if last > 0 else "SHORT"
        # hard direction restriction from the config
        allowed = set(cfg.allowed_directions or ["LONG", "SHORT"])
        if direction not in allowed:
            logger.debug("config %s: %s fired %s but config allows %s — blocked",
                         cfg.version, pair, direction, sorted(allowed))
            continue
        if _recently_sent(sym, cfg.version, direction):
            logger.info("config %s: %s %s already sent recently — skip (dedup)", cfg.version, direction, sym)
            continue
        last_close = float(ca[-1])
        scores.append(CandidateScore(
            symbol=sym.upper(),
            pair=pair,
            composite_score=_DEFAULT_SCORE,
            direction=direction,
            last_close=last_close,
            components={
                "ohlc": True,
                "strategy": cfg.ohlc_strategy,
                "tf_hours": tf_hours,
                "params": dict(cfg.ohlc_params or {}),
                "signal_at_close": last_close,
            },
            config_version=cfg.version,
            coin_type=get_coin_type(sym.upper()),
            rr_ratio=cfg.rr_ratio,
        ))
    return scores