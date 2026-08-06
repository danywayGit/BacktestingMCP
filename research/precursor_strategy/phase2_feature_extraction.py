"""
Phase 2 — Precursor Feature Extraction for BTCUSDT / ETHUSDT

For each detected event, examines the K candles immediately preceding the event
and extracts numeric indicators, price action features, and futures-specific data.

Outputs: precursor_features.csv

Usage:
    python research/precursor_strategy/phase2_feature_extraction.py
"""

import sys, os, csv, json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple
from collections import Counter

import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.core.backtesting_engine import BacktestingEngine
from config.settings import TimeFrame

# ── Parameters ───────────────────────────────────────────────────────────
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
TIMEFRAME = TimeFrame.M15
LOOKBACK_CANDLES_VALUES = [10, 20, 50]  # K values to test: 2.5h, 5h, 12.5h
LOOKBACK_DAYS = 365
OUTPUT_DIR = Path(__file__).parent
EVENTS_FILE = OUTPUT_DIR / "events.csv"


def load_events() -> List[Dict]:
    """Load events from Phase 1."""
    events = []
    with open(EVENTS_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append(row)
    return events


def _compute_rsi(close: pd.Series, period: int = 14) -> float:
    diff = close.diff()
    gain = diff.clip(lower=0).rolling(period).mean()
    loss = (-diff.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty and pd.notna(rsi.iloc[-1]) else 50.0


def _compute_macd(close: pd.Series) -> Tuple[float, float, float]:
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    hist = macd - signal
    return float(macd.iloc[-1] if not macd.empty else 0), \
           float(signal.iloc[-1] if not signal.empty else 0), \
           float(hist.iloc[-1] if not hist.empty else 0)


def extract_features(
    df: pd.DataFrame,
    event: Dict,
    K: int,
) -> Optional[Dict]:
    """Extract precursor features from K candles before an event."""
    # Find the event start index in the dataframe
    event_start_ts = pd.Timestamp(event["event_start_ts"]).tz_localize(None)
    df_index = pd.DatetimeIndex(df.index).tz_localize(None)

    # Find the closest candle to event start
    idx_positions = df_index.get_indexer([event_start_ts], method="ffill")
    event_idx = idx_positions[0]

    if event_idx < K:
        return None  # Not enough pre-event data

    # Extract pre-event window: K candles before event start
    pre_start = event_idx - K
    pre_end = event_idx

    pre_close = df['Close'].iloc[pre_start:pre_end].values
    pre_high = df['High'].iloc[pre_start:pre_end].values
    pre_low = df['Low'].iloc[pre_start:pre_end].values
    pre_vol = df['Volume'].iloc[pre_start:pre_end].values
    pre_timestamps = df.index[pre_start:pre_end]

    pre_close_s = pd.Series(pre_close)
    pre_high_s = pd.Series(pre_high)
    pre_low_s = pd.Series(pre_low)
    pre_vol_s = pd.Series(pre_vol)

    # ── Price Action Features ──
    # Consecutive candle streaks
    # Price changes
    price_changes = np.diff(pre_close)
    consec_green = 0
    consec_red = 0
    for i in range(len(pre_close) - 1, 0, -1):
        if pre_close[i] > pre_close[i - 1]:
            consec_green += 1
            consec_red = 0
        elif pre_close[i] < pre_close[i - 1]:
            consec_red += 1
            consec_green = 0
        if consec_green > 0 and consec_red > 0:
            break

    # Higher highs / lower lows
    hh_count = sum(1 for i in range(1, len(pre_high)) if pre_high[i] > pre_high[i-1])
    ll_count = sum(1 for i in range(1, len(pre_low)) if pre_low[i] < pre_low[i-1])

    # ── Volatility Features ──
    # ATR over lookback window
    tr = pd.concat([
        pd.Series(pre_high) - pd.Series(pre_low),
        (pd.Series(pre_high) - pd.Series(pre_close).shift(1)).abs(),
        (pd.Series(pre_low) - pd.Series(pre_close).shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = float(tr.tail(14).mean()) if len(tr) >= 14 else 0
    atr_pct = (atr / pre_close[-1] * 100) if pre_close[-1] > 0 else 0

    # Bollinger Band width
    bb_mid = float(np.mean(pre_close[-20:])) if len(pre_close) >= 20 else pre_close[-1]
    bb_std = float(np.std(pre_close[-20:])) if len(pre_close) >= 20 else 0
    bb_width = bb_std * 4 / bb_mid * 100 if bb_mid > 0 else 0

    # Historical BB width percentile (vs trailing 100 periods)
    if len(pre_close) >= 120:
        hist_bb_widths = []
        for i in range(100):
            segment = pre_close[max(0, i-20):i] if i >= 20 else pre_close[:i]
            if len(segment) >= 5:
                s_mean = np.mean(segment)
                s_std = np.std(segment)
                hist_bb_widths.append(s_std * 4 / s_mean * 100 if s_mean > 0 else 0)
        bb_pctile = sum(1 for bw in hist_bb_widths if bw <= bb_width) / len(hist_bb_widths) * 100 if hist_bb_widths else 50
    else:
        bb_pctile = 50

    # ── Volume Features ──
    vol_ratio = pre_vol[-1] / np.mean(pre_vol) if np.mean(pre_vol) > 0 else 1.0
    vol_zscore = (pre_vol[-1] - np.mean(pre_vol)) / np.std(pre_vol) if np.std(pre_vol) > 0 else 0

    # Volume-price divergence
    price_trend = pre_close[-1] - pre_close[0]
    vol_trend = pre_vol[-1] - pre_vol[0]
    vol_div = 0
    if price_trend > 0 and vol_trend < 0:
        vol_div = -1  # bearish divergence
    elif price_trend < 0 and vol_trend > 0:
        vol_div = 1  # bullish divergence

    # ── Momentum Features ──
    rsi = _compute_rsi(pre_close_s)
    macd, signal, hist = _compute_macd(pre_close_s)
    macd_hist_slope = hist - _compute_macd(pre_close_s.iloc[:-1])[2] if len(pre_close) > 1 else 0

    # RSI slope
    if len(pre_close) >= 20:
        rsi_first_half = _compute_rsi(pre_close_s.iloc[:len(pre_close)//2])
        rsi_second_half = _compute_rsi(pre_close_s.iloc[len(pre_close)//2:])
        rsi_slope = rsi_second_half - rsi_first_half
    else:
        rsi_slope = 0

    # ── Cross-asset Features ──
    # Check if the other symbol was moving before this event
    other_symbol_return = 0
    # This would need data from the other symbol — simplified for now

    # ── Time Features ──
    try:
        event_dt = datetime.fromisoformat(event["event_start_ts"].replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        event_dt = datetime.now()
    hour_of_day = event_dt.hour
    day_of_week = event_dt.weekday()

    return {
        "event_id": event["event_id"],
        "lookback_K": K,
        # Price action
        "direction_streak_len": consec_green if float(event["pct_move"]) > 0 else consec_red,
        "hh_ratio": round(hh_count / K, 2) if K > 0 else 0,
        "ll_ratio": round(ll_count / K, 2) if K > 0 else 0,
        # Volatility
        "atr_pct": round(atr_pct, 4),
        "bb_width_pct": round(bb_width, 2),
        "bb_width_pctile": round(bb_pctile, 1),
        # Volume
        "volume_ratio": round(vol_ratio, 2),
        "volume_zscore": round(vol_zscore, 2),
        "volume_divergence": vol_div,
        # Momentum
        "rsi_level": round(rsi, 1),
        "rsi_slope": round(rsi_slope, 1),
        "macd_hist_slope": round(macd_hist_slope, 4),
        # Time
        "hour_of_day_utc": hour_of_day,
        "day_of_week": day_of_week,
        # Visual labels (placeholder — Phase 2b)
        "chart_pattern_labels": "",
        "chart_pattern_confidence": 0,
        "chart_notes": "",
        # Control flag
        "is_control": False,
    }


def extract_control_features(df: pd.DataFrame, events: List[Dict], K: int) -> List[Dict]:
    """Extract features from NON-event candles for control comparison."""
    event_starts = set()
    for e in events:
        ts = pd.Timestamp(e["event_start_ts"]).tz_localize(None)
        idx_positions = pd.DatetimeIndex(df.index).tz_localize(None).get_indexer([ts], method="ffill")
        if idx_positions[0] >= 0:
            event_starts.add(idx_positions[0])

    # Sample non-event candles (points far from any event)
    control_features = []
    step = max(1, len(df) // 100)  # Sample ~100 control points

    for i in range(K, len(df) - K, step):
        # Skip if this is within K candles of an event
        if any(abs(i - es) < K for es in event_starts):
            continue

        # Create a synthetic event
        ctrl_event = {
            "event_id": f"CTRL-{i:06d}",
            "event_start_ts": str(df.index[i]).split(".")[0],
            "pct_move": "0",
        }

        feats = extract_features(df, ctrl_event, K)
        if feats:
            feats["is_control"] = True
            feats["event_id"] = f"CTRL-{i:06d}"
            control_features.append(feats)

    return control_features


def write_features_csv(all_features: List[Dict], output_path: Path):
    """Write features to CSV."""
    if not all_features:
        print("  No features to write.")
        return

    fieldnames = [
        "event_id", "lookback_K", "is_control",
        "direction_streak_len", "hh_ratio", "ll_ratio",
        "atr_pct", "bb_width_pct", "bb_width_pctile",
        "volume_ratio", "volume_zscore", "volume_divergence",
        "rsi_level", "rsi_slope", "macd_hist_slope",
        "hour_of_day_utc", "day_of_week",
        "chart_pattern_labels", "chart_pattern_confidence", "chart_notes",
        # Add funding_rate, open_interest later
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_features)
    print(f"  Wrote {len(all_features)} rows to {output_path}")


if __name__ == "__main__":
    print(f"\n  ── Phase 2: Precursor Feature Extraction ──")
    print(f"  Lookback K values: {LOOKBACK_CANDLES_VALUES} candles")
    print()

    engine = BacktestingEngine()
    events = load_events()
    print(f"  Loaded {len(events)} events from {EVENTS_FILE}")

    all_features = []

    for symbol in SYMBOLS:
        sym_events = [e for e in events if e["symbol"] == symbol]
        if not sym_events:
            continue

        print(f"\n  Processing {symbol} ({len(sym_events)} events)...")

        # Fetch full OHLCV data
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=LOOKBACK_DAYS)
        df = engine.get_data(symbol, TIMEFRAME, start, end)
        if df.empty:
            print(f"  ERROR: No data for {symbol}")
            continue

        print(f"  Got {len(df)} candles")

        for K in LOOKBACK_CANDLES_VALUES:
            print(f"    K={K} ({K*15//60}h{'' if K*15%60==0 else f'{K*15%60}m'})...")

            # Event features
            event_features = []
            for event in sym_events:
                feats = extract_features(df, event, K)
                if feats:
                    event_features.append(feats)

            # Control features
            ctrl_features = extract_control_features(df, sym_events, K)

            all_features.extend(event_features)
            all_features.extend(ctrl_features)

            print(f"      Events: {len(event_features)} | Controls: {len(ctrl_features)}")

    # Write combined output
    write_features_csv(all_features, OUTPUT_DIR / "precursor_features.csv")

    print(f"\n  Phase 2 complete. Ready for Phase 3 — Precursor Mining.")
    print(f"  Features saved to: {OUTPUT_DIR / 'precursor_features.csv'}")
    print()