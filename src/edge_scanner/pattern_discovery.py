"""
Pattern Discovery Engine — reverse-engineers what precedes 5%+ moves on BTC and ETH.

Approach:
1. Fetch OHLCV data for BTCUSDT and ETHUSDT (90+ days of 1h data)
2. Find all instances where price moves 5%+ in a forward window (5-20 candles)
3. For each "big move", extract the preceding N candles and compute indicators
4. Identify common indicator ranges that precede big moves
5. Generate a data-driven scoring config

Usage:
    python -m src.edge_scanner.pattern_discovery
"""

import json, sys, os
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Tuple
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from dataclasses import dataclass, field, asdict
from src.core.backtesting_engine import BacktestingEngine
from src.data.timeframe_converter import TimeFrame


@dataclass
class PreMoveSnapshot:
    """What the market looked like before a big move."""
    symbol: str
    direction: str  # 'UP' or 'DOWN'
    move_pct: float
    move_start: str  # timestamp
    move_end: str

    # Pre-move indicators (N candles before the move)
    pre_rsi: float = 0
    pre_volume_ratio: float = 0  # volume / avg_volume(20)
    pre_atr_pct: float = 0
    pre_ema50_distance_pct: float = 0  # % away from EMA50
    pre_ema200_distance_pct: float = 0
    pre_adx: float = 0
    pre_bb_position: float = 0  # 0=bottom, 0.5=middle, 1=top
    pre_volume_divergence: float = 0  # price vs volume trend
    pre_consecutive_green: int = 0  # bullish candles before move
    pre_consecutive_red: int = 0  # bearish candles before move
    pre_volatility_ratio: float = 0  # current ATR / ATR(50)
    pre_funding_rate: float = 0
    pre_macd_crossover: bool = False
    pre_support_distance_pct: float = 0  # % above nearest support
    pre_resistance_distance_pct: float = 0  # % below nearest resistance


def _compute_rsi(close: pd.Series, period: int = 14) -> float:
    diff = close.diff()
    gain = diff.clip(lower=0).rolling(period).mean()
    loss = (-diff.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty and pd.notna(rsi.iloc[-1]) else 50.0


def _compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    """Compute ADX (trend strength)."""
    plus_dm = high.diff()
    minus_dm = low.diff()
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.where(plus_dm > minus_dm, 0).rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.where(minus_dm > plus_dm, 0).rolling(period).mean() / atr)
    dx = abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan) * 100
    adx = dx.rolling(period).mean()
    return float(adx.iloc[-1]) if not adx.empty and pd.notna(adx.iloc[-1]) else 0.0


def analyze_symbol(symbol: str, engine: BacktestingEngine, lookback_days: int = 180,
                   move_threshold_pct: float = 5.0, lookforward_candles: int = 12,
                   pre_candles: int = 20) -> Tuple[List[PreMoveSnapshot], Dict]:
    """Find all big moves and analyze what preceded them."""
    print(f"\n{'='*60}")
    print(f"  Analyzing {symbol} — {lookback_days} days of 1h data")
    print(f"  Big move threshold: {move_threshold_pct}% | Lookforward: {lookforward_candles}h")
    print(f"  Pre-move window: {pre_candles}h")
    print('='*60)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    df = engine.get_data(symbol, TimeFrame.H1, start, end)

    if df.empty or len(df) < 100:
        print(f"  ERROR: Not enough data for {symbol}")
        return [], {}

    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    volume = df['Volume'].values
    timestamps = df.index.values

    snapshots: List[PreMoveSnapshot] = []

    for i in range(len(df) - lookforward_candles - pre_candles):
        # Look at the forward window: is there a 5%+ move?
        fwd_start = i + pre_candles
        fwd_end = fwd_start + lookforward_candles

        entry_price = close[i + pre_candles - 1]  # last close before the forward window
        fwd_high = high[fwd_start:fwd_end].max()
        fwd_low = low[fwd_start:fwd_end].min()

        up_move = (fwd_high - entry_price) / entry_price * 100
        down_move = (entry_price - fwd_low) / entry_price * 100

        # Determine if this is a big move
        direction = None
        move_pct = 0.0
        move_idx = None  # index where the move happened

        if up_move >= move_threshold_pct:
            direction = 'UP'
            move_pct = up_move
            # Find exact candle where it happened
            for j in range(fwd_start, fwd_end):
                if (high[j] - entry_price) / entry_price * 100 >= move_threshold_pct:
                    move_idx = j
                    break
        elif down_move >= move_threshold_pct:
            direction = 'DOWN'
            move_pct = down_move
            for j in range(fwd_start, fwd_end):
                if (entry_price - low[j]) / entry_price * 100 >= move_threshold_pct:
                    move_idx = j
                    break

        if direction is None or move_idx is None:
            continue

        # Extract pre-move window
        pre_start_idx = i
        pre_end_idx = i + pre_candles
        if pre_end_idx >= len(df):
            continue

        pre_close = close[pre_start_idx:pre_end_idx]
        pre_high = high[pre_start_idx:pre_end_idx]
        pre_low = low[pre_start_idx:pre_end_idx]
        pre_vol = volume[pre_start_idx:pre_end_idx]
        pre_close_series = pd.Series(pre_close)
        pre_high_series = pd.Series(pre_high)
        pre_low_series = pd.Series(pre_low)
        pre_vol_series = pd.Series(pre_vol)

        # Compute indicators for the pre-move window
        try:
            # RSI on the pre-move window
            rsi = _compute_rsi(pre_close_series)

            # Volume ratio (last 3h volume vs avg)
            vol_ratio = pre_vol[-3:].mean() / pre_vol.mean() if pre_vol.mean() > 0 else 1.0

            # ATR %
            tr = pd.concat([
                pre_high_series - pre_low_series,
                (pre_high_series - pd.Series(pre_close).shift(1)).abs(),
                (pre_low_series - pd.Series(pre_close).shift(1)).abs(),
            ], axis=1).max(axis=1)
            atr = float(tr.tail(14).mean()) if len(tr) >= 14 else 0
            atr_pct = (atr / pre_close[-1] * 100) if pre_close[-1] > 0 else 0

            # EMA50 and EMA200 distance
            close_series = pd.Series(close[:pre_end_idx])
            ema50 = float(close_series.ewm(span=50).mean().iloc[-1]) if len(close_series) >= 50 else pre_close[-1]
            ema200 = float(close_series.ewm(span=200).mean().iloc[-1]) if len(close_series) >= 200 else pre_close[-1]
            ema50_dist = (pre_close[-1] - ema50) / ema50 * 100
            ema200_dist = (pre_close[-1] - ema200) / ema200 * 100 if ema200 else 0

            # Bollinger Band position (0=bottom, 1=top)
            bb_mid = float(pre_close_series.tail(20).mean()) if len(pre_close_series) >= 20 else pre_close[-1]
            bb_std = float(pre_close_series.tail(20).std()) if len(pre_close_series) >= 20 else 0
            bb_top = bb_mid + 2 * bb_std if bb_std > 0 else pre_close[-1]
            bb_bot = bb_mid - 2 * bb_std if bb_std > 0 else pre_close[-1]
            bb_pos = (pre_close[-1] - bb_bot) / (bb_top - bb_bot) if (bb_top - bb_bot) > 0 else 0.5

            # Consecutive green/red candles
            consec_green = 0
            consec_red = 0
            for j in range(len(pre_close) - 1, 0, -1):
                if pre_close[j] > pre_close[j - 1]:
                    consec_green += 1
                    consec_red = 0
                elif pre_close[j] < pre_close[j - 1]:
                    consec_red += 1
                    consec_green = 0
                if consec_green > 0 and consec_red > 0:
                    break

            # Volatility ratio (recent ATR vs longer-term ATR)
            long_atr = 0
            if len(close) >= pre_end_idx + 50:
                long_tr = pd.Series(high[max(0, pre_end_idx-50):pre_end_idx]) - pd.Series(low[max(0, pre_end_idx-50):pre_end_idx])
                long_atr = float(long_tr.tail(20).mean()) if len(long_tr) >= 20 else atr
            vol_ratio_vs_atr = atr / long_atr if long_atr > 0 else 1.0

            # Volume divergence (price direction vs volume direction)
            price_change = pre_close[-1] - pre_close[0]
            vol_change = pre_vol[-1] - pre_vol[0]
            vol_div = 0
            if price_change > 0 and vol_change < 0:
                vol_div = -1  # bearish divergence (price up, volume down)
            elif price_change < 0 and vol_change > 0:
                vol_div = 1  # bullish divergence (price down, volume up)

            snapshot = PreMoveSnapshot(
                symbol=symbol,
                direction=direction,
                move_pct=round(move_pct, 2),
                move_start=str(timestamps[fwd_start]).split('.')[0],
                move_end=str(timestamps[min(move_idx, len(timestamps)-1)]).split('.')[0],
                pre_rsi=round(rsi, 1),
                pre_volume_ratio=round(vol_ratio, 2),
                pre_atr_pct=round(atr_pct, 3),
                pre_ema50_distance_pct=round(ema50_dist, 2),
                pre_ema200_distance_pct=round(ema200_dist, 2),
                pre_bb_position=round(bb_pos, 2),
                pre_consecutive_green=consec_green,
                pre_consecutive_red=consec_red,
                pre_volatility_ratio=round(vol_ratio_vs_atr, 2),
                pre_volume_divergence=vol_div,
            )
            snapshots.append(snapshot)

        except Exception as e:
            continue

    print(f"  Found {len(snapshots)} big moves (≥{move_threshold_pct}%)")
    up_count = sum(1 for s in snapshots if s.direction == 'UP')
    down_count = sum(1 for s in snapshots if s.direction == 'DOWN')
    print(f"    UP moves: {up_count} | DOWN moves: {down_count}")

    # Analyze patterns
    analysis = analyze_patterns(snapshots, symbol)
    return snapshots, analysis


def analyze_patterns(snapshots: List[PreMoveSnapshot], symbol: str) -> Dict:
    """Find common patterns in the pre-move data."""
    if not snapshots:
        return {}

    analysis = {
        "symbol": symbol,
        "total_moves": len(snapshots),
        "up_moves": sum(1 for s in snapshots if s.direction == 'UP'),
        "down_moves": sum(1 for s in snapshots if s.direction == 'DOWN'),
        "patterns": {}
    }

    def percentile(values, pct):
        sorted_v = sorted(values)
        idx = int(len(sorted_v) * pct / 100)
        return sorted_v[min(idx, len(sorted_v)-1)]

    # Analyze each indicator
    up_snaps = [s for s in snapshots if s.direction == 'UP']
    down_snaps = [s for s in snapshots if s.direction == 'DOWN']

    for label, snaps in [("UP", up_snaps), ("DOWN", down_snaps)]:
        if not snaps:
            continue
        indicators = {
            "RSI": [s.pre_rsi for s in snaps],
            "Volume_Ratio": [s.pre_volume_ratio for s in snaps],
            "ATR_pct": [s.pre_atr_pct for s in snaps],
            "EMA50_dist_pct": [s.pre_ema50_distance_pct for s in snaps],
            "BB_Position": [s.pre_bb_position for s in snaps],
            "Volatility_Ratio": [s.pre_volatility_ratio for s in snaps],
            "Consecutive_Green": [s.pre_consecutive_green for s in snaps],
            "Consecutive_Red": [s.pre_consecutive_red for s in snaps],
        }
        analysis["patterns"][label] = {}
        for ind_name, vals in indicators.items():
            analysis["patterns"][label][ind_name] = {
                "median": round(percentile(vals, 50), 2),
                "p25": round(percentile(vals, 25), 2),
                "p75": round(percentile(vals, 75), 2),
                "p10": round(percentile(vals, 10), 2),
                "p90": round(percentile(vals, 90), 2),
                "min": round(min(vals), 2),
                "max": round(max(vals), 2),
            }

    # Find what makes UP moves different from DOWN moves
    if up_snaps and down_snaps:
        analysis["discriminating_factors"] = {}
        for ind_name in ["pre_rsi", "pre_volume_ratio", "pre_atr_pct",
                          "pre_ema50_distance_pct", "pre_bb_position",
                          "pre_volatility_ratio"]:
            up_median = percentile([getattr(s, ind_name) for s in up_snaps], 50)
            down_median = percentile([getattr(s, ind_name) for s in down_snaps], 50)
            diff = abs(up_median - down_median)
            if diff > 0.5:  # Significant difference
                analysis["discriminating_factors"][ind_name] = {
                    "up_median": round(up_median, 2),
                    "down_median": round(down_median, 2),
                    "difference": round(diff, 2),
                }

    return analysis


def print_report(analysis: Dict):
    """Print a human-readable report of the analysis."""
    if not analysis:
        print("No data to analyze")
        return

    sym = analysis["symbol"]
    total = analysis["total_moves"]
    up = analysis["up_moves"]
    down = analysis["down_moves"]

    print(f"\n{'='*60}")
    print(f"  PATTERN DISCOVERY REPORT — {sym}")
    print(f"  {total} big moves (≥5%): {up} UP, {down} DOWN")
    print('='*60)

    for label in ["UP", "DOWN"]:
        if label not in analysis.get("patterns", {}):
            continue
        p = analysis["patterns"][label]
        print(f"\n  📈 {label} MOVES — Typical pre-move profile:")
        print(f"  {'Indicator':<25} {'Median':<8} {'25-75% range':<15} {'10-90% range':<15}")
        print(f"  {'-'*63}")
        for ind_name, stats in p.items():
            med = stats["median"]
            iqr = f"{stats['p25']} — {stats['p75']}"
            dec = f"{stats['p10']} — {stats['p90']}"
            print(f"  {ind_name:<25} {med:<8} {iqr:<15} {dec:<15}")

    if "discriminating_factors" in analysis:
        print(f"\n  🔑 KEY DIFFERENCES (UP vs DOWN):")
        for ind_name, stats in analysis["discriminating_factors"].items():
            print(f"    {ind_name:<20}: UP median={stats['up_median']:<8} DOWN median={stats['down_median']:<8} (diff={stats['difference']})")

    # Generate config suggestion
    print(f"\n  💡 SUGGESTED CONFIG PARAMETERS (for {sym}):")
    up_p = analysis.get("patterns", {}).get("UP", {})
    down_p = analysis.get("patterns", {}).get("DOWN", {})

    if up_p:
        rsi = up_p.get("RSI", {})
        vol = up_p.get("Volume_Ratio", {})
        atr = up_p.get("ATR_pct", {})
        print(f"    For LONG signals (expecting UP moves):")
        print(f"      RSI range: {rsi.get('p10', 0)} — {rsi.get('p90', 100)} (sweet spot: {rsi.get('p25', 0)} — {rsi.get('p75', 100)})")
        print(f"      Volume ratio > {vol.get('p25', 0.5)} (median: {vol.get('median', 1.0)}x)")
        print(f"      ATR > {atr.get('p10', 0)}% (median: {atr.get('median', 0.5)}%)")


def generate_config_from_analysis(analysis: Dict) -> Dict:
    """Generate config parameters from the analysis."""
    if not analysis:
        return {}

    up_p = analysis.get("patterns", {}).get("UP", {})

    def mid_range(stats):
        """Get middle of the interquartile range."""
        if not stats:
            return 0
        return round((stats.get('p25', 0) + stats.get('p75', 0)) / 2, 2)

    config = {
        "source": f"Pattern Discovery Engine — {analysis['symbol']}",
        "total_moves_analyzed": analysis['total_moves'],
        "suggested_params": {}
    }

    if up_p:
        rsi = up_p.get("RSI", {})
        vol = up_p.get("Volume_Ratio", {})
        atr = up_p.get("ATR_pct", {})
        bb = up_p.get("BB_Position", {})
        ema50 = up_p.get("EMA50_dist_pct", {})

        config["suggested_params"] = {
            # RSI range (sweet spot for UP moves)
            "min_rsi": max(0, int(rsi.get('p10', 0))),
            "max_rsi": min(100, int(rsi.get('p90', 100))),
            # Volume confirmation
            "min_volume_relative": max(0.1, round(vol.get('p25', 0.5), 1)),
            # Volatility
            "min_atr_pct": max(0.05, round(atr.get('p10', 0.1), 2)),
            # EMA distance from entry
            "max_ema50_distance_pct": abs(round(ema50.get('p75', 5), 1)),
            "min_ema50_distance_pct": -abs(round(ema50.get('p25', -5), 1)),
            # BB position
            "max_bb_position": round(bb.get('p90', 1.0), 1),
            "min_bb_position": round(bb.get('p10', 0), 1),
        }

    return config


if __name__ == "__main__":
    engine = BacktestingEngine()
    all_results = {}

    for sym in ["BTCUSDT", "ETHUSDT"]:
        snaps, analysis = analyze_symbol(
            symbol=sym,
            engine=engine,
            lookback_days=180,
            move_threshold_pct=5.0,
            lookforward_candles=16,
            pre_candles=20,
        )
        all_results[sym] = {
            "snapshots_count": len(snaps),
            "analysis": analysis
        }
        print_report(analysis)

        # Generate config
        config = generate_config_from_analysis(analysis)
        print(f"\n  Generated config: {json.dumps(config, indent=4)}")

        # Save individual results
        output_file = f"results/pattern_discovery_{sym}_{datetime.now().strftime('%Y%m%d')}.json"
        os.makedirs("results", exist_ok=True)
        with open(output_file, "w") as f:
            json.dump({"snapshots": [asdict(s) for s in snaps], "analysis": analysis}, f, indent=2, default=str)
        print(f"  Saved to {output_file}")

    # Save combined
    with open("results/pattern_discovery_combined.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n{'='*60}")
    print("  Combined results saved to results/pattern_discovery_combined.json")
    print('='*60)