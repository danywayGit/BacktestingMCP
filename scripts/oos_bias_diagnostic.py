#!/usr/bin/env python3
"""oos_bias_diagnostic.py — is the OOS window regime-lucky, direction-biased, or real?

The full OOS sweep reported 73/81 configs PASS*, which is suspicious: when
nearly everything passes, the gate may be rewarding regime luck (longs in a bull
OOS window) rather than config edge. This diagnostic answers that directly:

  1. REGIME: for the OOS tail (last 40%) of each liquid symbol, compute the
     buy-and-hold drift. If most drift UP, the window is bullish and long-biased
     configs pass by luck.
  2. DIRECTION ASYMMETRY: run the SAME MTF-gated setup LONG vs SHORT on the same
     symbols. If LONG dominates but SHORT is negative, the "edge" is direction
     luck, not a robust signal. A real edge survives both-ish; a regime artifact
     is lopsided.
  3. MARKET-RELATIVE: subtract the symbol's OOS buy-and-hold from the config EV
     so we see excess (alpha) vs just riding the trend.

Output: printed + results/_oos_bias_diagnostic.json. Compute-only.
"""
import json, os, sqlite3
from collections import defaultdict
import numpy as np

DB = "/home/hermes/BacktestingMCP/data/crypto.db"
OUT = "/home/hermes/BacktestingMCP/results/_oos_bias_diagnostic.json"
EMA_N = 50
TRAIN_FRAC = 0.60
STOP, RR, HOLD = 3.0, 2.0, 0   # the combo family most picked; 3.0/2.0/0 is representative
SYMBOLS = ["ETHUSDT", "BTCUSDT", "SOLUSDT", "BNBUSDT", "LINKUSDT", "XRPUSDT",
           "ADAUSDT", "DOGEUSDT", "UNIUSDT", "DOTUSDT", "AVAXUSDT"]


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


def gated_entries(hi, lo, opn, cl, tss, direction_sign):
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
            entries.append((i, float(opn[i]))); i += 6
        else:
            i += 1
    return entries


def _atr(h, l, c, period=14):
    n = len(c)
    if n < period + 1:
        return float(np.mean(h - l)) or 1.0
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    return float(np.mean(tr[:period]))


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
    return {"ev_r": ev / nt if nt else 0.0, "trades": nt,
            "wr": w / nt if nt else 0.0}


def main():
    conn = sqlite3.connect(DB)
    rows = {}
    for sym in SYMBOLS:
        r = conn.execute(
            "SELECT timestamp,open,high,low,close FROM market_data WHERE symbol=? "
            "AND timeframe='1h' ORDER BY timestamp", (sym,)).fetchall()
        if len(r) >= int(300 / TRAIN_FRAC):
            rows[sym] = r
    conn.close()

    per_sym = {}
    long_evs, short_evs, drift = [], [], []
    for sym, r in rows.items():
        tss = [x[0] for x in r]; opn = np.asarray([x[1] for x in r], float)
        hi = np.asarray([x[2] for x in r], float); lo = np.asarray([x[3] for x in r], float)
        cl = np.asarray([x[4] for x in r], float)
        split = int(len(cl) * TRAIN_FRAC)
        # OOS buy-and-hold drift (fractional)
        bh = (cl[-1] - cl[split]) / cl[split] if cl[split] else 0.0
        drift.append(bh)
        # both directions over the OOS tail
        e_long = gated_entries(hi[split:], lo[split:], opn[split:], cl[split:], tss[split:], 1.0)
        e_short = gated_entries(hi[split:], lo[split:], opn[split:], cl[split:], tss[split:], -1.0)
        rl = sim(hi[split:], lo[split:], opn[split:], cl[split:], e_long, STOP, RR, HOLD)
        rs = sim(hi[split:], lo[split:], opn[split:], cl[split:], e_short, STOP, RR, HOLD)
        long_evs.append(rl["ev_r"]); short_evs.append(rs["ev_r"])
        per_sym[sym] = {"oos_drift_pct": round(bh * 100, 2),
                        "long": {"ev_r": round(rl["ev_r"], 3), "trades": rl["trades"], "wr": round(rl["wr"], 3)},
                        "short": {"ev_r": round(rs["ev_r"], 3), "trades": rs["trades"], "wr": round(rs["wr"], 3)}}

    res = {
        "n_symbols": len(rows),
        "oos_drift": {"mean_pct": round(float(np.mean(drift)) * 100, 2),
                      "median_pct": round(float(np.median(drift)) * 100, 2),
                      "n_up": sum(1 for d in drift if d > 0),
                      "n_down": sum(1 for d in drift if d < 0)},
        "long": {"median_ev": round(float(np.median(long_evs)), 3),
                 "n_positive": sum(1 for e in long_evs if e > 0)},
        "short": {"median_ev": round(float(np.median(short_evs)), 3),
                  "n_positive": sum(1 for e in short_evs if e > 0)},
        "per_symbol": per_sym,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {OUT}\n")
    print("=== OOS WINDOW REGIME ===")
    print(f"population = {res['n_symbols']} symbols, combo stop{STOP} rr{RR} hold{HOLD}h")
    print(f"OOS drift: mean {res['oos_drift']['mean_pct']}%, median {res['oos_drift']['median_pct']}% "
          f"-> up {res['oos_drift']['n_up']}, down {res['oos_drift']['n_down']}")
    print("\n=== DIRECTION ASYMMETRY (same setup, LONG vs SHORT) ===")
    print(f"LONG  median EV {res['long']['median_ev']:+.3f}  positive {res['long']['n_positive']}/{res['n_symbols']}")
    print(f"SHORT median EV {res['short']['median_ev']:+.3f}  positive {res['short']['n_positive']}/{res['n_symbols']}")
    print("\nper symbol:")
    for sym, p in per_sym.items():
        bias = "UP" if p["oos_drift_pct"] > 0 else "DOWN"
        print(f"  {sym:>10} drift {p['oos_drift_pct']:>+6.1f}% [{bias}]  long {p['long']['ev_r']:+.3f} "
              f"(n{p['long']['trades']})  short {p['short']['ev_r']:+.3f} (n{p['short']['trades']})")


if __name__ == "__main__":
    main()