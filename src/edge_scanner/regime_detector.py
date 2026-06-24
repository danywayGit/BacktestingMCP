"""
Market regime detector for the edge scanner.

Uses OHLCV from the backtesting engine to classify the current market
regime into one of five states. The detected regime drives dynamic weight
adjustment in weight_adjuster.py.

Regimes
-------
BULL_TRENDING   — price above EMAs, ADX > 25, trending upward
BEAR_TRENDING   — price below EMAs, ADX > 25, trending downward
SIDEWAYS        — ADX <= 25, price between EMAs, low trending conviction
HIGH_VOLATILITY — ATR % of price > 90th percentile over lookback
LOW_VOLATILITY  — ATR % of price < 10th percentile over lookback
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import TimeFrame

logger = logging.getLogger(__name__)

# Default symbols for regime detection (BTC sets macro context)
REGIME_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]


def _compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average Directional Index (ADX) from OHLCV data.

    Uses Wilder's smoothing. Returns a Series aligned to the input index.
    First `period` values are NaN.
    """
    high = df["High"].values.astype(float)
    low = df["Low"].values.astype(float)
    close = df["Close"].values.astype(float)

    # True Range
    prev_close = np.roll(close, 1)
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - prev_close),
            np.abs(low - prev_close),
        ),
    )

    # +DM and -DM
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # Wilder's smoothing
    alpha = 1.0 / period
    tr_smooth = np.full_like(tr, np.nan)
    plus_smooth = np.full_like(plus_dm, np.nan)
    minus_smooth = np.full_like(minus_dm, np.nan)

    tr_smooth[period] = np.mean(tr[1 : period + 1])
    plus_smooth[period] = np.mean(plus_dm[1 : period + 1])
    minus_smooth[period] = np.mean(minus_dm[1 : period + 1])

    for i in range(period + 1, len(tr)):
        tr_smooth[i] = tr_smooth[i - 1] + alpha * (tr[i] - tr_smooth[i - 1])
        plus_smooth[i] = plus_smooth[i - 1] + alpha * (plus_dm[i] - plus_smooth[i - 1])
        minus_smooth[i] = minus_smooth[i - 1] + alpha * (minus_dm[i] - minus_smooth[i - 1])

    # +DI and -DI
    plus_di = 100.0 * plus_smooth / tr_smooth
    minus_di = 100.0 * minus_smooth / tr_smooth

    # DX and ADX
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = np.full_like(dx, np.nan)
    adx[period * 2] = np.mean(dx[period : period * 2 + 1])
    for i in range(period * 2 + 1, len(dx)):
        adx[i] = adx[i - 1] + alpha * (dx[i] - adx[i - 1])

    return pd.Series(adx, index=df.index)


def _compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def _compute_atr_pct(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR as a percentage of close price."""
    high = df["High"].values.astype(float)
    low = df["Low"].values.astype(float)
    close = df["Close"].values.astype(float)

    prev_close = np.roll(close, 1)
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - prev_close),
            np.abs(low - prev_close),
        ),
    )
    tr_series = pd.Series(tr, index=df.index)
    atr = tr_series.rolling(period).mean()
    return atr / close * 100.0  # ATR as % of price


def detect_regime(
    symbol: str = "BTC/USDT",
    timeframe: TimeFrame = TimeFrame.H1,
    lookback_days: int = 30,
) -> dict:
    """Detect the current market regime from OHLCV data.

    Parameters
    ----------
    symbol : str
        Trading pair symbol, e.g. "BTC/USDT" (default).
    timeframe : TimeFrame
        Candle timeframe for analysis (default: H1).
    lookback_days : int
        How many days of history to fetch (default: 30).

    Returns
    -------
    dict
        {
            'regime': str,
            'adx': float,
            'volatility': float,   # latest ATR as % of price
            'price_trend': float,  # % change of EMA50 over last 5 periods
            'confidence': float,    # 0.0 – 1.0
        }
    """
    from ..core.backtesting_engine import engine

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=lookback_days)

    data = engine.get_data(symbol, timeframe, start_date, end_date)
    if data.empty:
        logger.warning("No data returned for %s, returning unknown regime", symbol)
        return {
            "regime": "UNKNOWN",
            "adx": 0.0,
            "volatility": 0.0,
            "price_trend": 0.0,
            "confidence": 0.0,
        }

    # Ensure numeric columns
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    close = data["Close"]

    # ── Technical indicators ────────────────────────────────────────────
    ema20 = _compute_ema(close, 20)
    ema50 = _compute_ema(close, 50)
    ema200 = _compute_ema(close, 200) if len(close) >= 200 else close * 0 + close.mean()

    adx_series = _compute_adx(data)
    atr_pct_series = _compute_atr_pct(data)

    # Latest values
    latest_adx = float(adx_series.iloc[-1]) if not adx_series.empty and not np.isnan(adx_series.iloc[-1]) else 0.0
    latest_atr_pct = float(atr_pct_series.iloc[-1]) if not atr_pct_series.empty and not np.isnan(atr_pct_series.iloc[-1]) else 0.0
    latest_close = float(close.iloc[-1])
    latest_ema20 = float(ema20.iloc[-1])
    latest_ema50 = float(ema50.iloc[-1])
    latest_ema200 = float(ema200.iloc[-1])

    # Price trend (EMA50 slope over last 5 periods)
    if len(ema50) >= 5 and not np.isnan(ema50.iloc[-5]):
        price_trend = (ema50.iloc[-1] - ema50.iloc[-5]) / ema50.iloc[-5] * 100.0
    else:
        price_trend = 0.0

    # ── Volatility percentile ───────────────────────────────────────────
    valid_atr = atr_pct_series.dropna()
    if len(valid_atr) >= 20:
        vol_percentile = (valid_atr > latest_atr_pct).mean()
    else:
        vol_percentile = 0.5

    # ── Regime classification ───────────────────────────────────────────
    # Check volatility extremes first (can override trend regimes)
    if vol_percentile >= 0.90:
        regime = "HIGH_VOLATILITY"
    elif vol_percentile <= 0.10:
        regime = "LOW_VOLATILITY"
    elif latest_adx > 25:
        # Trending — check direction
        if latest_close > latest_ema20 > latest_ema50 and price_trend > 0:
            regime = "BULL_TRENDING"
        elif latest_close < latest_ema20 < latest_ema50 and price_trend < 0:
            regime = "BEAR_TRENDING"
        else:
            # ADX says trending but EMAs are ambiguous — check short-term
            if price_trend > 0.5:
                regime = "BULL_TRENDING"
            elif price_trend < -0.5:
                regime = "BEAR_TRENDING"
            else:
                regime = "SIDEWAYS"
    else:
        regime = "SIDEWAYS"

    # ── Confidence ──────────────────────────────────────────────────────
    # Confidence is based on ADX strength and data sufficiency
    adx_confidence = min(latest_adx / 50.0, 1.0)  # ADX=50 → full confidence
    data_quality = min(len(data) / 200.0, 1.0)    # 200+ candles → full confidence
    confidence = round(adx_confidence * 0.6 + data_quality * 0.4, 2)

    return {
        "regime": regime,
        "adx": round(latest_adx, 1),
        "volatility": round(latest_atr_pct, 2),
        "price_trend": round(price_trend, 2),
        "confidence": confidence,
    }


def detect_market_context(
    symbols: Optional[list[str]] = None,
    timeframe: TimeFrame = TimeFrame.H1,
    lookback_days: int = 30,
) -> dict:
    """Aggregate regime across multiple top symbols for a consensus view.

    Parameters
    ----------
    symbols : list[str] | None
        Symbols to analyse (default: BTC/USDT, ETH/USDT, SOL/USDT).

    Returns
    -------
    dict
        {
            'consensus_regime': str,
            'symbols': {symbol: regime_result},
            'volatility_context': 'HIGH' | 'NORMAL' | 'LOW',
        }
    """
    if symbols is None:
        symbols = REGIME_SYMBOLS

    results = {}
    for sym in symbols:
        try:
            results[sym] = detect_regime(sym, timeframe, lookback_days)
        except Exception as exc:
            logger.warning("Failed to detect regime for %s: %s", sym, exc)
            results[sym] = {
                "regime": "UNKNOWN",
                "adx": 0.0,
                "volatility": 0.0,
                "price_trend": 0.0,
                "confidence": 0.0,
            }

    # Consensus: majority vote, weighted by confidence
    regime_votes: dict[str, float] = {}
    for res in results.values():
        reg = res["regime"]
        conf = res["confidence"]
        regime_votes[reg] = regime_votes.get(reg, 0.0) + conf

    consensus_regime = max(regime_votes, key=regime_votes.get) if regime_votes else "UNKNOWN"

    # Volatility context
    volatilities = [r["volatility"] for r in results.values() if r["volatility"] > 0]
    avg_vol = sum(volatilities) / len(volatilities) if volatilities else 0.0
    if avg_vol > 3.0:
        vol_context = "HIGH"
    elif avg_vol < 0.5:
        vol_context = "LOW"
    else:
        vol_context = "NORMAL"

    return {
        "consensus_regime": consensus_regime,
        "symbols": results,
        "volatility_context": vol_context,
    }