#!/usr/bin/env python3
"""mtf_gated_exit_grid.py — MULTI-TIMEFRAME-gated entry + exit-geometry optimization.

Combines the two findings from the earlier passes:
  - regime sweep  : only the 1.x-6.x/14.0 family survives tough windows
  - MTF test      : 4h+1h trend confirmation is NOT a universal filter, but it
                    RESCUES specific configs (8.0,12.0,14.0,22.0) as a per-config gate
So here we test the concrete tradable setup never tried before: for each target
config's ACTUAL liquid symbols, fire an entry ONLY when the 4h AND 1h EMA-trend
both agree with the signal direction, then simulate the exit on real high/low
with an ATR-based SL/TP under a stop x rr x max-hold grid. This is the natural
next step: MTF gate on entry + tuned exit.

Data      : crypto.db market_data (1h OHLCV, resample for 4h trend).
Engine    : CPU fallback — same math as the 4090 gpu_server (see
            exit_geometry_sweep.py). Safe to run even if the GPU sleeps.
Output    : results/_mtf_gated_grid.json
              per config+symbol -> {aggregate per grid combo over allowed entries}
Compute-only: reads OHLC, writes a JSON. No trading/system changes.
"""
import json, os, sqlite3, math
from collections import defaultdict
import numpy as np

DB = "/home/hermes/BacktestingMCP/data/crypto.db"
OUT = "/home/hermes/BacktestingMCP/results/_mtf_gated_grid.json"
EMA_N = 50
GRID_STOPS = [2.0, 2.5, 3.0, 3.5]
GRID_RRS   = [1.0, 1.2, 1.5, 2.0]
GRID_HOLDS = [0, 24, 48]
MIN_SYMBOL_TRADES = 5   # a combo counts on a symbol only if >= this many entries


def ema(arr):
    a = np.asarray(arr, dtype=float)
    n = len(a)
    if n == 0:
        return np.array([])
    alpha = 2.0 / (EMA_N + 1)
    out = np.empty(n); s = a[0]; out[0] = s
    for i in range(1, n):
        s = alpha * a[i] + (1 - alpha) * s
        out[i] = s
    return out


def slope(em):
    if len(em) < 4:
        return 0.0
    return 1.0 if em[-1] >= float(np.mean(em[-4:-1])) else -1.0


def _atr(high, low, close, period=14):
    n = len(close)
    if n < period + 1:
        return float(np.mean(high[:] - low[:])) or 1.0
    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(np.abs(high[1:] - close[:-1]),
                               np.abs(low[1:] - close[:-1])))
    return float(np.mean(tr[:period]))


def run_mtf_gated(conn, symbol, direction_sign, stops, rrs, holds, ema_period=EMA_N):
    """Simulate MTF-gated strategy on one symbol's 1h OHLC.
    Entry: open a LONG(+1)/SHORT(-1) when both 1h and 4h EMA-slope == direction.
    Exit : SL based on ATR*stop_mult; TP based on rr*stop_mult*ATR; max_hold bars.
    Returns { (stop,rr,hold): {wins,losses,ev_r,wr,trades} } over gated entries.
    """
    rows = conn.execute(
        "SELECT timestamp,open,high,low,close FROM market_data WHERE symbol=? "
        "AND timeframe='1h' ORDER BY timestamp", (symbol,)).fetchall()
    if len(rows) < max(3 * ema_period, 300):
        return None
    tss = [r[0] for r in rows]; opn = np.asarray([r[1] for r in rows], float)
    hi = np.asarray([r[2] for r in rows], float); lo = np.asarray([r[3] for r in rows], float)
    cl = np.asarray([r[4] for r in rows], float)
    # 1h trend slope per bar
    ema1 = ema(cl)
    s1 = np.empty(len(cl))
    for i in range(len(cl)):
        s1[i] = slope(ema1[max(0, i - EMA_N + 1):i + 1])
    # 4h trend slope per bar: bucket closes, EMA over 4h buckets, assign each 1h bar the slope of its 4h bucket
    b4 = {}; order = []
    for i, t in enumerate(tss):
        bb = (t // 14400) * 14400
        if bb not in b4:
            b4[bb] = cl[i]; order.append(bb)
        else:
            b4[bb] = cl[i]
    closes4 = [b4[x] for x in order]
    ema4 = ema(closes4)
    s4_bucket = {}
    for j, bb in enumerate(order):
        s4_bucket[bb] = slope(ema4[max(0, j - EMA_N + 1):j + 1])
    s4 = np.empty(len(cl))
    ptr = 0
    for i, t in enumerate(tss):
        bb = (t // 14400) * 14400
        while ptr < len(order) - 1 and order[ptr + 1] <= bb:
            ptr += 1
        s4[i] = s4_bucket.get(order[ptr], 0.0) if ptr < len(order) else 0.0
    atr = _atr(hi, lo, cl)
    n = len(cl)
    acc = defaultdict(lambda: {"w": 0, "l": 0, "ev": 0.0})
    i = 3 * EMA_N  # warmup: enough 4h bars + 1h bars
    while i < n:
        d = direction_sign
        gated = (s1[i] == d) and (s4[i] == d)
        if not gated:
            i += 1
            continue
        entry = float(opn[i])
        last_j = i + 1
        for stopm in stops:
            for rr in rrs:
                for hold in holds:
                    sl = entry - d * stopm * atr
                    tp = entry + d * rr * stopm * atr
                    j = i + 1
                    outcome = None
                    while j < n and (hold == 0 or (j - i) <= hold):
                        h_j = hi[j]; l_j = lo[j]
                        if d > 0:
                            if l_j <= sl: outcome = "L"; break
                            if h_j >= tp: outcome = "W"; break
                        else:
                            if h_j >= sl: outcome = "L"; break
                            if l_j <= tp: outcome = "W"; break
                        j += 1
                    if outcome is None:
                        outcome = "F"
                    k = (stopm, rr, hold)
                    if outcome == "W":
                        acc[k]["w"] += 1; acc[k]["ev"] += rr
                    elif outcome == "L":
                        acc[k]["l"] += 1; acc[k]["ev"] -= 1.0
                    last_j = max(last_j, j)
        i = last_j + 1  # advance past the resolved bars so entries don't overlap
    out = {}
    for k, a in acc.items():
        n_tr = a["w"] + a["l"]
        out[k] = {"wins": a["w"], "losses": a["l"], "trades": n_tr,
                  "ev_r": round(a["ev"] / n_tr, 3) if n_tr else 0.0,
                  "wr": round(a["w"] / n_tr, 3) if n_tr else 0.0}
    return out


def load_targets():
    p = "/home/hermes/BacktestingMCP/results/_mtf_grid_symbols.json"
    if os.path.exists(p):
        return json.load(open(p))
    # fallback
    return {"8.0": ["DEXEUSDT", "BTWUSDT"]}


def main():
    targets = load_targets()
    conn = sqlite3.connect(DB)
    results = {}
    for cfg, syms in targets.items():
        direction_sign = 1.0  # run LONG-oriented; the gate handles trend
        cfg_res = {}
        for sym in syms:
            r = run_mtf_gated(conn, sym, direction_sign, GRID_STOPS, GRID_RRS, GRID_HOLDS)
            if r:
                cfg_res[sym] = r
        results[cfg] = cfg_res
    conn.close()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    # stringify combo tuple keys for JSON serialization
    stringified = {}
    for cfg, syms in results.items():
        stringified[cfg] = {sym: {"__".join(map(str, k)): a for k, a in r.items()} for sym, r in syms.items()}
    with open(OUT, "w") as f:
        json.dump(stringified, f, indent=2)
    print(f"wrote {OUT}")

    # aggregate: best combo per config across its symbols (combo median EV)
    for cfg, syms in results.items():
        combos = defaultdict(list)
        for sym, r in syms.items():
            for k, a in r.items():
                if a["trades"] >= MIN_SYMBOL_TRADES:
                    combos[k].append(a["ev_r"])
        ranked = [(k, np.median(v), len(v)) for k, v in combos.items() if v]
        ranked.sort(key=lambda x: -x[1])
        print(f"\n=== {cfg} (top combos by median EV, symbols n>={MIN_SYMBOL_TRADES}) ===")
        for k, medev, ns in ranked[:6]:
            print(f"  stop{k[0]} rr{k[1]} hold{k[2]}h  medEV={medev:+.3f}  over {ns} symbols")


if __name__ == "__main__":
    main()