#!/usr/bin/env python3
"""exit_geometry_sweep.py — CPU fallback for the GPU MCP exit-geometry sweep.

Purpose: the exit-geometry grid (atr_stop_mult x rr_ratio x max_hold) must NEVER
block on GPU availability. If the 4090's MCP server is asleep/unreachable, run
the SAME sequential position-sim locally (pure numpy, mirrors src/mcp/gpu_server.py
`_sweep_single`), log that the GPU was offline, and return identical-shaped
results. The caller decides based on results, not on which engine produced them.

This mirrors the server's math EXACTLY so GPU/CPU outputs are comparable:
  - entry proxy : close > EMA50 => LONG(+1) else SHORT(-1)
  - SL distance : ATR14 (mean abs diff of first 14 bars, then floor of diffs)
  - resolve     : touch SL => loss(-1R), touch TP => win(+rr), else flat (max_hold)
  - skip icicle : after a position closes, advance i = j+1

Single-arg close per call (like the tool). API:
  sweep_single_close(close, stops, rrs, holds) -> list[dict]  (same shape as GPU)
  main() reads a JSON file {"SYM":[close,...],...} or runs built-in demo.
"""
import json, math, os, sys

import numpy as np


def _ema(arr, period):
    out = np.empty_like(arr, dtype=float)
    alpha = 2.0 / (period + 1.0)
    s = float(arr[0]); out[0] = s
    for i in range(1, len(arr)):
        s = alpha * float(arr[i]) + (1 - alpha) * s
        out[i] = s
    return out


def _signal(close, ema_period=50):
    ema = _ema(close, ema_period)
    return np.where(close > ema, 1.0, -1.0)


def _sweep_single(close, direction, atr_stop_mult, rr_ratio, max_hold):
    n = len(close)
    diffs = np.abs(np.diff(close, prepend=close[0]))
    atr = float(diffs[:14].mean()) if n >= 15 else 1.0
    for i in range(1, min(14, n)):
        atr = max(atr, float(diffs[i]))
    wins = losses = flats = 0
    total_r = 0.0
    i = 0
    while i < n:
        d = direction[i]
        if abs(d) < 0.5:
            i += 1
            continue
        entry = float(close[i])
        sl = entry - d * atr_stop_mult * atr
        tp = entry + d * rr_ratio * atr_stop_mult * atr
        outcome = None
        j = i + 1
        while j < n and (max_hold == 0 or (j - i) <= max_hold):
            # server uses close[j] only (lo=hi=close[j]) — mirror EXACTLY so results match
            lo = float(close[j]); hi = float(close[j])
            if d > 0:
                if lo <= sl: outcome = "L"; break
                if hi >= tp: outcome = "W"; break
            else:
                if hi >= sl: outcome = "L"; break
                if lo <= tp: outcome = "W"; break
            j += 1
        if outcome is None:
            outcome = "F"
        if outcome == "W":
            total_r += rr_ratio; wins += 1
        elif outcome == "L":
            total_r -= 1.0; losses += 1
        else:
            flats += 1
        i = j + 1
    n_t = wins + losses
    return {
        "wins": wins, "losses": losses, "flats": flats,
        "ev_r": round(total_r / n_t, 3) if n_t else 0.0,
        "wr": round(wins / n_t, 3) if n_t else 0.0,
        "trades": n_t, "total_r": round(total_r, 2),
        "atr_stop_mult": atr_stop_mult, "rr_ratio": rr_ratio,
        "max_hold_hours": max_hold,
    }


def sweep_single_close(close, stops=None, rrs=None, holds=None, ema_period=50):
    stops = stops or [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    rrs = rrs or [1.0, 1.1, 1.2, 1.5, 2.0, 2.5, 3.0]
    holds = holds or [0, 12, 24, 48, 72]
    close = np.asarray(close, dtype=float)
    direction = _signal(close, ema_period)
    results = []
    for s in stops:
        for r in rrs:
            for h in holds:
                results.append(_sweep_single(close, direction, s, r, h))
    results.sort(key=lambda x: -x["ev_r"])
    return {"gpu": False, "engine": "cpu", "elapsed_s": None, "n_bars": len(close),
            "results": results, "count": len(results)}


def _load_close(path):
    data = json.load(open(path))
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if isinstance(v, list) and v}
    return {"_": data}


def main():
    pool = {}
    if len(sys.argv) > 1:
        pool = _load_close(sys.argv[1])
    else:
        demo = [2529.1, 2518.68, 2514.78, 2498.97, 2500.86, 2505.53, 2503.44,
                2517.54, 2516.45, 2513.0, 2501.12, 2509.38, 2508.46, 2510.93,
                2512.14, 2511.15, 2497.95, 2507.88, 2506.47, 2514.58]
        pool = {"demo": demo}
    for sym, close in pool.items():
        res = sweep_single_close(close)
        top = res["results"][:5]
        print(f"\n{sym}  n_bars={res['n_bars']}")
        for r in top:
            print(f"  stop{r['atr_stop_mult']} rr{r['rr_ratio']} hold{r['max_hold_hours']:>2}h "
                  f"EV{r['ev_r']:+.3f} WR{r['wr']*100:5.1f}% N{r['trades']}")


if __name__ == "__main__":
    main()