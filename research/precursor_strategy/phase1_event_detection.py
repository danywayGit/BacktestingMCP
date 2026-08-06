"""
Phase 1 — Event Detection for BTCUSDT / ETHUSDT

Detects all historical episodes where price moved >=5% within a defined rolling window.
Outputs events.csv with one row per distinct move episode.

Usage:
    python research/precursor_strategy/phase1_event_detection.py
"""

import sys, os, json, csv
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.core.backtesting_engine import BacktestingEngine
from src.data.timeframe_converter import TimeFrame

# ── Configurable parameters ──────────────────────────────────────────────
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
TIMEFRAME = TimeFrame.H1                 # 1h candles
W_CANDLES = 24                           # Rolling window: 24h (24 × 1h) — catch larger moves
MOVE_THRESHOLD_PCT = 5.0                 # 5% move threshold
LOOKBACK_DAYS = 365                      # 1 year of data
OUTPUT_DIR = Path(__file__).parent


def detect_events(symbol: str, engine: BacktestingEngine) -> List[Dict]:
    """Detect all 5%+ move events for a symbol using rolling W-candle windows."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=LOOKBACK_DAYS)

    print(f"  Fetching {LOOKBACK_DAYS}d of {TIMEFRAME.value} data for {symbol}...")
    df = engine.get_data(symbol, TIMEFRAME, start, end)
    if df.empty or len(df) < 100:
        print(f"  ERROR: Insufficient data for {symbol} ({len(df)} rows)")
        return []

    print(f"  Got {len(df)} candles. Scanning for {MOVE_THRESHOLD_PCT}%+ moves over {W_CANDLES}-candle windows...")

    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    timestamps = df.index.values

    # Rolling window: for each candle i, compute forward return over W_CANDLES
    # pct_change[i] = (close[i+W] - close[i]) / close[i] * 100
    raw_events = []  # List of (start_idx, end_idx, direction, pct_move, max_intra_pct)

    for i in range(len(df) - W_CANDLES):
        entry_price = close[i]
        window_high = high[i:i + W_CANDLES].max()
        window_low = low[i:i + W_CANDLES].min()
        exit_price = close[i + W_CANDLES - 1]

        up_move = (window_high - entry_price) / entry_price * 100
        down_move = (entry_price - window_low) / entry_price * 100
        actual_return = (exit_price - entry_price) / entry_price * 100

        if up_move >= MOVE_THRESHOLD_PCT:
            max_intra = max(
                (high[j] - entry_price) / entry_price * 100
                for j in range(i, i + W_CANDLES)
            )
            raw_events.append({
                "start_idx": i,
                "end_idx": i + W_CANDLES - 1,
                "direction": "up",
                "pct_move": round(actual_return, 2),
                "max_intra_window_pct": round(max_intra, 2),
                "start_price": entry_price,
                "end_price": exit_price,
            })
        elif down_move >= MOVE_THRESHOLD_PCT:
            max_intra = min(
                (low[j] - entry_price) / entry_price * 100
                for j in range(i, i + W_CANDLES)
            )
            raw_events.append({
                "start_idx": i,
                "end_idx": i + W_CANDLES - 1,
                "direction": "down",
                "pct_move": round(actual_return, 2),
                "max_intra_window_pct": round(max_intra, 2),
                "start_price": entry_price,
                "end_price": exit_price,
            })

    # Deduplicate overlapping events
    # If two windows overlap for the same direction, merge them into one event
    # (keep the earliest start, latest end)
    if not raw_events:
        return []

    # Sort by start_idx
    raw_events.sort(key=lambda e: e["start_idx"])

    merged_events = []
    current = raw_events[0]

    for event in raw_events[1:]:
        # Overlap: same direction and windows overlap or touch
        if (event["direction"] == current["direction"]
                and event["start_idx"] <= current["end_idx"] + 1):
            # Merge: extend current end if event ends later
            if event["end_idx"] > current["end_idx"]:
                current["end_idx"] = event["end_idx"]
                current["end_price"] = event["end_price"]
                current["pct_move"] = event["pct_move"]  # Final return of merged event
            # Keep larger max_intra
            if abs(event["max_intra_window_pct"]) > abs(current["max_intra_window_pct"]):
                current["max_intra_window_pct"] = event["max_intra_window_pct"]
        else:
            merged_events.append(current)
            current = event
    merged_events.append(current)

    # Convert to output format
    events = []
    for i, e in enumerate(merged_events):
        start_ts = str(timestamps[e["start_idx"]]).split('.')[0].replace('T', ' ')
        end_ts = str(timestamps[e["end_idx"]]).split('.')[0].replace('T', ' ')
        duration = e["end_idx"] - e["start_idx"] + 1

        events.append({
            "event_id": f"{symbol.split('USDT')[0]}-{i+1:04d}",
            "symbol": symbol,
            "timeframe": TIMEFRAME.value,
            "window_candles_W": W_CANDLES,
            "direction": e["direction"],
            "event_start_ts": start_ts,
            "event_end_ts": end_ts,
            "start_price": round(e["start_price"], 2),
            "end_price": round(e["end_price"], 2),
            "pct_move": e["pct_move"],
            "duration_candles": duration,
            "max_intra_window_pct": e["max_intra_window_pct"],
            "notes": "",
        })

    return events


def write_events_csv(events: List[Dict], output_path: Path):
    """Write events to CSV."""
    if not events:
        print("  No events to write.")
        return

    fieldnames = [
        "event_id", "symbol", "timeframe", "window_candles_W", "direction",
        "event_start_ts", "event_end_ts", "start_price", "end_price",
        "pct_move", "duration_candles", "max_intra_window_pct", "notes"
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(events)
    print(f"  Wrote {len(events)} events to {output_path}")


def print_summary(all_events: Dict[str, List[Dict]]):
    """Print a summary of detected events."""
    print(f"\n{'='*70}")
    print(f"  EVENT DETECTION SUMMARY")
    print(f"  Timeframe: {TIMEFRAME.value} | Window: {W_CANDLES}h | Threshold: {MOVE_THRESHOLD_PCT}%")
    print(f"  Lookback: {LOOKBACK_DAYS} days")
    print('='*70)

    for symbol, events in all_events.items():
        sym = symbol.split('USDT')[0]
        up = [e for e in events if e["direction"] == "up"]
        down = [e for e in events if e["direction"] == "down"]

        pcts = [abs(e["pct_move"]) for e in events]
        durations = [e["duration_candles"] for e in events]

        print(f"\n  {sym}:")
        print(f"    Total events: {len(events)}")
        print(f"    UP moves: {len(up)} | DOWN moves: {len(down)}")
        if pcts:
            print(f"    Magnitude: min={min(pcts):.1f}% median={pd.Series(pcts).median():.1f}% max={max(pcts):.1f}%")
        if durations:
            print(f"    Duration: min={min(durations)}h median={pd.Series(durations).median():.0f}h max={max(durations)}h")

        # Timing distribution
        if events:
            hours = [int(e["event_start_ts"].split()[1].split(':')[0]) for e in events]
            hourly_dist = pd.Series(hours).value_counts().sort_index()
            top_hours = hourly_dist.head(5)
            print(f"    Most common start hours (UTC): {dict(top_hours.to_dict())}")


if __name__ == "__main__":
    print(f"\n  ── Phase 1: Event Detection ──")
    print(f"  Parameters: timeframe={TIMEFRAME.value}, W={W_CANDLES}h, threshold={MOVE_THRESHOLD_PCT}%")
    print()

    engine = BacktestingEngine()
    all_events = {}

    for symbol in SYMBOLS:
        events = detect_events(symbol, engine)
        all_events[symbol] = events

    # Write combined events.csv
    combined = []
    for events in all_events.values():
        combined.extend(events)
    write_events_csv(combined, OUTPUT_DIR / "events.csv")

    # Print summary
    print_summary(all_events)

    print(f"\n  [HUMAN CHECKPOINT] — Review event list above before proceeding to Phase 2.")
    print(f"  Events saved to: {OUTPUT_DIR / 'events.csv'}")
    print()