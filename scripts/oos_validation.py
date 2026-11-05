#!/usr/bin/env python3
"""oos_validation.py — OUT-OF-SAMPLE validation of the MTF-gated setup.

Purpose: the MTF-gated exit grid (mtf_gated_exit_grid.py) picked parameter combos
on the SAME data it scored them on — that is in-sample / curve-fit risk. Before
any prop-money proposal, we must prove the edge survives UNSEEN data.

Method (walk-forward split):
  1. For each symbol's 1h history, split at 60% into TRAIN (first 60%) and OOS
     (last 40%). The OOS tail is NEVER seen while picking params.
  2. On TRAIN only, the config picks its best (stop, rr, hold) combo by median EV
     across train symbols (fixed grid, no per-symbol cherry-picking).
  3. Apply that SAME combo to the OOS tail.
  4. Report whether OOS EV is > 0 (edge persists) and compare OOS vs train.

We use the SAME MTF entry gate (4h+1h trend agreement) as the grid, so this is a
true holdout test of today's finding, not a new strategy.

Output: results/_oos_validation.json — per-config {train: {...}, oos: {...}}.
Compute-only; no trading/system changes.
"""
import json, os, sqlite3
from collections import defaultdict
import numpy as np

DB = "/home/hermes/BacktestingMCP/data/crypto.db"
OUT = "/home/hermes/BacktestingMCP/results/_oos_validation.json"
EMA_N = 50
GRID_STOPS = [2.0, 2.5, 3.0, 3.5]
GRID_RRS   = [1.0, 1.2, 1.5, 2.0]
GRID_HOLDS = [0, 24, 48]
TRAIN_FRAC = 0.60
MIN_TRAIN_TRADES = 4   # need enough train entries to trust the picked combo


def ema(arr):
    a = np.asarray(arr, float); n = len(a)
    if n == 0:
        return np.array([])
    alpha = 2.0 / (EMA_N + 1); out = np.empty(n); s = a[0]; out[0] = s
    for i in range(1, n):
        s = alpha * a[i] + (1 - alpha) * s; out[i] = s
    return out


def slope(em):
    if len(em) < 4:
        return 0.0
    return 1.0 if em[-1] >= float(np.mean(em[-4:-1])) else -1.0


def _atr(h, l, c, period=14):
    n = len(c)
    if n < period + 1:
        return float(np.mean(h - l)) or 1.0
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    return float(np.mean(tr[:period]))


def gated_entries(hi, lo, opn, cl, tss, direction_sign=1.0):
    """Return list of (entry_idx, entry_price) gated by 4h AND 1h trend == direction."""
    n = len(cl)
    ema1 = ema(cl); s1 = np.empty(n)
    for i in range(n):
        s1[i] = slope(ema1[max(0, i - EMA_N + 1):i + 1])
    b4 = {}; order = []
    for i, t in enumerate(tss):
        bb = (t // 14400) * 14400
        if bb not in b4:
            b4[bb] = cl[i]; order.append(bb)
        else:
            b4[bb] = cl[i]
    ema4 = ema([b4[x] for x in order]); s4_bucket = {}
    for j, bb in enumerate(order):
        s4_bucket[bb] = slope(ema4[max(0, j - EMA_N + 1):j + 1])
    s4 = np.empty(n); ptr = 0
    for i, t in enumerate(tss):
        bb = (t // 14400) * 14400
        while ptr < len(order) - 1 and order[ptr + 1] <= bb:
            ptr += 1
        s4[i] = s4_bucket.get(order[ptr], 0.0) if ptr < len(order) else 0.0
    entries = []
    i = 3 * EMA_N
    while i < n:
        if s1[i] == direction_sign and s4[i] == direction_sign:
            entries.append((i, float(opn[i])))
            # skip ahead a window to avoid overlapping entries (coarse, fine for validation)
            i += 6
        else:
            i += 1
    return entries


def sim(hi, lo, opn, cl, entries, stopm, rr, hold):
    n = len(cl); atr = _atr(hi, lo, cl)
    w = l = 0; ev = 0.0
    for (i, entry) in entries:
        d = 1.0
        sl = entry - d * stopm * atr; tp = entry + d * rr * stopm * atr
        j = i + 1; outcome = None
        while j < n and (hold == 0 or (j - i) <= hold):
            if d > 0:
                if lo[j] <= sl: outcome = "L"; break
                if hi[j] >= tp: outcome = "W"; break
            else:
                if hi[j] >= sl: outcome = "L"; break
                if lo[j] <= tp: outcome = "W"; break
            j += 1
        if outcome is None:
            outcome = "F"
        if outcome == "W":
            w += 1; ev += rr
        elif outcome == "L":
            l += 1; ev -= 1.0
    nt = w + l
    return {"w": w, "l": l, "trades": nt, "ev_r": round(ev / nt, 3) if nt else 0.0,
            "wr": round(w / nt, 3) if nt else 0.0}


def load_symbols():
    p = "/home/hermes/BacktestingMCP/results/_mtf_grid_symbols.json"
    return json.load(open(p))


def main():
    targets = load_symbols()
    conn = sqlite3.connect(DB)
    results = {}
    # Step 1: on TRAIN, pick per-config best combo (max median EV across train symbols)
    train_combos = defaultdict(lambda: defaultdict(list))  # cfg -> (stop,rr,hold) -> [ev over train symbols]
    for cfg, syms in targets.items():
        for sym in syms:
            rows = conn.execute(
                "SELECT timestamp,open,high,low,close FROM market_data WHERE symbol=? "
                "AND timeframe='1h' ORDER BY timestamp", (sym,)).fetchall()
            if len(rows) < max(3 * EMA_N, 300) * (1 / TRAIN_FRAC):
                continue
            tss = [r[0] for r in rows]; opn = np.asarray([r[1] for r in rows], float)
            hi = np.asarray([r[2] for r in rows], float); lo = np.asarray([r[3] for r in rows], float)
            cl = np.asarray([r[4] for r in rows], float)
            split = int(len(cl) * TRAIN_FRAC)
            entries_tr = gated_entries(hi[:split], lo[:split], opn[:split], cl[:split], tss[:split])
            for stopm in GRID_STOPS:
                for rr in GRID_RRS:
                    for hold in GRID_HOLDS:
                        r = sim(hi[:split], lo[:split], opn[:split], cl[:split], entries_tr, stopm, rr, hold)
                        if r["trades"] >= MIN_TRAIN_TRADES:
                            train_combos[cfg][(stopm, rr, hold)].append(r["ev_r"])
    # pick best combo per config on train by median EV
    best = {}
    for cfg, c in train_combos.items():
        ranked = sorted(c.items(), key=lambda kv: -np.median(kv[1]))
        if ranked:
            best[cfg] = ranked[0][0]
    # Step 2: apply to OOS tail
    for cfg, syms in targets.items():
        if cfg not in best:
            results[cfg] = {"status": "no_train_combo"}
            continue
        stopm, rr, hold = best[cfg]
        tr_evs, oos_evs, oos_trades = [], [], []
        for sym in syms:
            rows = conn.execute(
                "SELECT timestamp,open,high,low,close FROM market_data WHERE symbol=? "
                "AND timeframe='1h' ORDER BY timestamp", (sym,)).fetchall()
            if len(rows) < max(3 * EMA_N, 300) * (1 / TRAIN_FRAC):
                continue
            tss = [r[0] for r in rows]; opn = np.asarray([r[1] for r in rows], float)
            hi = np.asarray([r[2] for r in rows], float); lo = np.asarray([r[3] for r in rows], float)
            cl = np.asarray([r[4] for r in rows], float)
            split = int(len(cl) * TRAIN_FRAC)
            etr = gated_entries(hi[:split], lo[:split], opn[:split], cl[:split], tss[:split])
            eos = gated_entries(hi[split:], lo[split:], opn[split:], cl[split:], tss[split:])
            rtr = sim(hi[:split], lo[:split], opn[:split], cl[:split], etr, stopm, rr, hold)
            ros = sim(hi[split:], lo[split:], opn[split:], cl[split:], eos, stopm, rr, hold)
            if rtr["trades"] >= MIN_TRAIN_TRADES:
                tr_evs.append(rtr["ev_r"])
            if ros["trades"] > 0:
                oos_evs.append(ros["ev_r"]); oos_trades.append(ros["trades"])
        results[cfg] = {
            "chosen_combo": {"stop": stopm, "rr": rr, "hold": hold},
            "train": {"median_ev": round(float(np.median(tr_evs)), 3) if tr_evs else None,
                      "n_symbols": len(tr_evs)},
            "oos": {"median_ev": round(float(np.median(oos_evs)), 3) if oos_evs else None,
                    "n_symbols": len(oos_evs), "total_trades": sum(oos_trades)},
        }
    conn.close()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)

    print(f"wrote {OUT}")
    print("\n=== OUT-OF-SAMPLE VALIDATION (tune on first 60%, replay on unseen 40%) ===")
    print(f"{'cfg':>5} {'combo':>16} {'trainEV':>8} {'OOS_EV':>8} {'OOS#':>6} {'verdict':<12}")
    for cfg in results:
        r = results[cfg]
        if "chosen_combo" not in r:
            print(f"{cfg:>5} ---- no-bench --- ")
            continue
        cb = r["chosen_combo"]
        tr = r["train"]["median_ev"]; oos = r["oos"]["median_ev"]
        nt = r["oos"]["total_trades"]
        ok = (oos is not None) and (oos > 0) and (nt >= 10)
        print(f"{cfg:>5} s{cb['stop']} r{cb['rr']} h{cb['hold']} {tr:>+8} {str(oos):>8} {nt:>6} "
              f"{'PASS*' if ok else ('FALLBACK' if oos in (None,) else ('WEAK' if not ok else ''))}")


if __name__ == "__main__":
    main()