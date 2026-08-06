"""
Phase 3 — Precursor Mining

Compares feature distributions of event vs control candles to find what
actually precedes 5%+ moves. Reports effect sizes, not just p-values.

Usage:
    python research/precursor_strategy/phase3_precursor_mining.py
"""

import sys, csv, json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUTPUT_DIR = Path(__file__).parent
FEATURES_FILE = OUTPUT_DIR / "precursor_features.csv"


def load_features() -> List[Dict]:
    rows = []
    with open(FEATURES_FILE, "r") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def cohen_d(vals1: List[float], vals2: List[float]) -> float:
    """Cohen's d effect size: (mean1 - mean2) / pooled_std."""
    if len(vals1) < 2 or len(vals2) < 2:
        return 0
    m1, m2 = np.mean(vals1), np.mean(vals2)
    s1, s2 = np.std(vals1, ddof=1), np.std(vals2, ddof=1)
    pooled = np.sqrt(((len(vals1)-1)*s1**2 + (len(vals2)-1)*s2**2) / (len(vals1)+len(vals2)-2))
    if pooled == 0:
        return 0
    return (m1 - m2) / pooled


def analyze_features(rows: List[Dict], symbol: str, direction: str, K: int) -> List[Dict]:
    """Compare event vs control features for a specific symbol/direction/K."""
    numeric_features = [
        "direction_streak_len", "hh_ratio", "ll_ratio",
        "atr_pct", "bb_width_pct", "bb_width_pctile",
        "volume_ratio", "volume_zscore",
        "rsi_level", "rsi_slope", "macd_hist_slope",
        "hour_of_day_utc", "day_of_week",
    ]

    # Filter rows
    event_rows = [
        r for r in rows
        if r["is_control"] == "False" and
        r["lookback_K"] == str(K) and
        symbol in r["event_id"]  # BTC-XXXX or ETH-XXXX
    ]
    # Need direction from events.csv — for now just use all events
    # (we'll split by direction later when we load events.csv too)

    ctrl_rows = [
        r for r in rows
        if r["is_control"] == "True" and
        r["lookback_K"] == str(K) and
        r["volume_ratio"]  # has data (control rows have no symbol field, use any)
    ]

    # Filter by symbol for controls (no symbol field, so we estimate)
    # For now: events with symbol prefix, controls are all

    results = []
    for feat in numeric_features:
        ev_vals = []
        for r in event_rows:
            try:
                v = float(r[feat]) if r[feat] else 0
                ev_vals.append(v)
            except (ValueError, TypeError):
                pass

        ctrl_vals = []
        for r in ctrl_rows:
            try:
                v = float(r[feat]) if r[feat] else 0
                ctrl_vals.append(v)
            except (ValueError, TypeError):
                pass

        if len(ev_vals) < 3 or len(ctrl_vals) < 3:
            continue

        d = cohen_d(ev_vals, ctrl_vals)
        ev_median = np.median(ev_vals)
        ctrl_median = np.median(ctrl_vals)

        results.append({
            "feature": feat,
            "event_median": round(ev_median, 2),
            "control_median": round(ctrl_median, 2),
            "effect_size": round(d, 3),
            "event_n": len(ev_vals),
            "control_n": len(ctrl_vals),
            "direction": "higher" if ev_median > ctrl_median else "lower",
        })

    results.sort(key=lambda r: abs(r["effect_size"]), reverse=True)
    return results


def print_results(all_results: Dict[str, List[Dict]]):
    print(f"\n{'='*70}")
    print(f"  PHASE 3 — PRECURSOR MINING RESULTS")
    print(f"  Comparing event features vs control set")
    print(f"  Effect size: Cohen's d (positive = higher before events)")
    print('='*70)

    for key, results in sorted(all_results.items()):
        symbol, K = key.split("_K")
        print(f"\n  {symbol} | K={K}h lookback | {len(results)} features tested")
        print(f"  {'Feature':<22} {'Event Med':<10} {'Ctrl Med':<10} {'Effect d':<10} {'N_evt':<6} {'N_ctrl':<6}")
        print(f"  {'-'*64}")

        for r in results[:8]:  # Top 8 features
            arrow = "🟢↑" if r["effect_size"] > 0.3 else ("🔴↓" if r["effect_size"] < -0.3 else "  ")
            print(f"  {arrow} {r['feature']:<20} {r['event_median']:<10} {r['control_median']:<10} {r['effect_size']:<10} {r['event_n']:<6} {r['control_n']:<6}")

        # Highlight strong predictors
        strong = [r for r in results if abs(r["effect_size"]) > 0.5]
        if strong:
            print(f"\n  🔑 Strong predictors (|d| > 0.5):")
            for r in strong:
                print(f"    {r['feature']}: d={r['effect_size']:.2f} ({r['direction']} before events)")
        else:
            print(f"\n  No strong predictors found (|d| <= 0.5 for all features)")


if __name__ == "__main__":
    print(f"\n  ── Phase 3: Precursor Mining ──")
    rows = load_features()
    print(f"  Loaded {len(rows)} feature rows from {FEATURES_FILE}")

    all_results = {}
    for K in [20, 40, 60]:
        for sym_prefix in ["BTC", "ETH"]:
            key = f"{sym_prefix}_K{K}"
            results = analyze_features(rows, sym_prefix, None, K)
            all_results[key] = results

    print_results(all_results)

    # Save report
    report = {
        "phase": "Phase 3 — Precursor Mining",
        "parameters": {"W": "30h (1h candles)", "K_values": [20, 40, 60]},
        "results": {}
    }
    for key, results in all_results.items():
        report["results"][key] = {
            "strong_predictors": [r for r in results if abs(r["effect_size"]) > 0.5],
            "all_features": results
        }

    # Generate config suggestions
    print(f"\n{'='*70}")
    print(f"  CONFIG SUGGESTIONS FROM PRECURSOR MINING")
    print('='*70)
    for key, results in sorted(all_results.items()):
        strong = [r for r in results if abs(r["effect_size"]) > 0.5]
        if strong:
            print(f"\n  {key}:")
            for r in strong:
                if r["feature"] == "rsi_level":
                    print(f"    RSI sweet spot: ~{r['event_median']:.0f} (control: ~{r['control_median']:.0f})")
                elif r["feature"] == "volume_ratio":
                    print(f"    Volume ratio: ~{r['event_median']:.1f}x (control: ~{r['control_median']:.1f}x)")
                elif r["feature"] == "atr_pct":
                    print(f"    ATR: ~{r['event_median']:.2f}% (control: ~{r['control_median']:.2f}%)")
                elif r["feature"] == "bb_width_pctile":
                    print(f"    BB width percentile: ~{r['event_median']:.0f}% (control: ~{r['control_median']:.0f}%)")
                else:
                    print(f"    {r['feature']}: event={r['event_median']} vs control={r['control_median']} (d={r['effect_size']:.2f})")

    output_file = OUTPUT_DIR / "phase3_precursor_mining.json"
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Full report saved to {output_file}")
    print(f"  [HUMAN CHECKPOINT] — Review precursor list before Phase 4.")
    print()