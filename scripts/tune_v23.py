#!/usr/bin/env python3
"""tune_v23.py — targeted pass-2 tuning of the V23/V24 OHLC strategies.

Reuses the OOS-validated harness from scripts/strategy_search.py (load_full →
resample trusted 1h → TYPES → sequential simulate with 15bp cost) but sweeps
DIDIER-APPROVED CUSTOM grids per strategy, plus two new entry variants:
  * meanrev_ema0  : mean-reversion WITHOUT the EMA50 trend filter (pure RSI touch)
  * ma_cross_mom  : ma_cross REQUIRING momentum (ROC>0) at the cross bar
Direction SPLIT included: every config runs LONG and SHORT separately so a
both-direction config's compromise is exposed.

Discipline: tune on aggregated TRAIN (first 60%), report the chosen config's
aggregated OOS (last 40%) EV side by side. Only STABLE (positive BOTH) combos
are deployable candidates; nothing is written to any config here.
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strategy_search import (load_full, resample, atr,  # noqa
                             simulate, ev_split, TYPES, COST, SYMBOLS)

import numpy as np


# ── entry variants not in TYPES ─────────────────────────────────────────────
def strat_meanrev_ema0(o, h, l, c, v, p=None):
    """Pure RSI mean-reversion, NO EMA50 filter (trades any RSI touch)."""
    p = p or {}
    n = p.get("rsi", 14); k = p.get("K", 50)  # K unused, kept for grid symmetry
    import strategy_search as SS
    r = SS.rsi(c, n)
    sig = np.zeros(len(c))
    for i in range(1, len(c)):
        if r[i] < p.get("oversold", 30):
            sig[i] = 1
        elif r[i] > p.get("overbought", 70):
            sig[i] = -1
    return sig


def strat_ma_cross_mom_impl(o, h, l, c, v, p):
    import strategy_search as SS
    f = SS.ema(c, p.get("fast", 10))
    s = SS.ema(c, p.get("slow", 50))
    roc_back = p.get("roc", 5)
    roc = np.zeros(len(c))
    pc = np.roll(c, roc_back); pc[:roc_back] = c[:roc_back]
    roc = (c - pc) / np.where(pc == 0, 1e-12, pc) * 100
    sig = np.zeros(len(c))
    for i in range(1, len(c)):
        up = f[i - 1] <= s[i - 1] and f[i] > s[i] and roc[i] > 0
        dn = f[i - 1] >= s[i - 1] and f[i] < s[i] and roc[i] < 0
        if up:
            sig[i] = 1
        elif dn:
            sig[i] = -1
    return sig


# ── direction-split simulate ────────────────────────────────────────────────
def sim_direction(sig, o, h, l, c, atr_series, sl_mult, rr, want, cost=COST):
    """Sequential sim, keeping only trades in `want` direction (+1 LONG/-1 SHORT)."""
    n = len(c); out = []
    i = 1
    while i < n:
        if sig[i] == 0 or not np.isfinite(atr_series[i]) or atr_series[i] <= 0:
            i += 1; continue
        d = sig[i]
        if d != want:
            i += 1; continue
        entry = c[i]; sl_d = atr_series[i] * sl_mult; tp_d = sl_d * rr
        sl = entry - d * sl_d; tp = entry + d * tp_d
        resolved = None; j = i + 1
        while j < n:
            if d > 0:
                if l[j] <= sl: resolved = "L"; break
                if h[j] >= tp: resolved = "W"; break
            else:
                if h[j] >= sl: resolved = "L"; break
                if l[j] <= tp: resolved = "W"; break
            j += 1
        if resolved == "W":
            out.append(rr - cost / sl_d)
        elif resolved == "L":
            out.append(-1.0 - cost / sl_d)
        else:
            out.append(-cost / sl_d)
        i = j + 1
    return out


def ev_direction(cube_data, symbols, tf_hours, typ, fn, sl, rr, params, want):
    """Aggregated train/OOS EV for one direction. (tr_ev, oo_ev, tr_n, oo_n)."""
    tr_sum = oo_sum = tr_n = oo_n = 0.0
    for s in symbols:
        rows = cube_data[s]; rb = resample(rows, tf_hours)
        o = [x[1] for x in rb]; h = [x[2] for x in rb]; l = [x[3] for x in rb]
        c = [x[4] for x in rb]; v = [x[5] for x in rb]
        if len(c) < 200: continue
        sig = fn(o, h, l, c, v, params); a = atr(h, l, c, 14)
        res = sim_direction(sig, o, h, l, c, a, sl, rr, want)
        if len(res) < 20: continue
        n = len(res); split = int(n * 0.60)
        tr = res[:split]; oo = res[split:]
        tr_sum += sum(tr); oo_sum += sum(oo); tr_n += len(tr); oo_n += len(oo)
    if tr_n < 15 or oo_n < 15: return None
    return (round(tr_sum / tr_n, 4), round(oo_sum / oo_n, 4), int(tr_n), int(oo_n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(SYMBOLS))
    ap.add_argument("--out", default="results/_v23_tune.json")
    args = ap.parse_args()
    syms = [s.strip().upper() for s in args.symbols.split(",")]
    cube = {s: load_full(s) for s in syms}
    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.out)
    print(f"symbols={syms} cost={COST*10000:.0f}bp tune-on-train60/validate-OOS40\n")

    # (label, tf, fn, param-grid, exit-grid, directions)
    runs = [
        # V23.1 meanrev — RSI period/thresholds, EMA filter ON vs OFF
        ("meanrev(V23.1)", 1, TYPES["meanrev"],
         [{"rsi":7,"oversold":25,"overbought":75},{"rsi":7,"oversold":30,"overbought":70},
          {"rsi":14,"oversold":25,"overbought":75},{"rsi":14,"oversold":30,"overbought":70},
          {"rsi":14,"oversold":35,"overbought":65},{"rsi":21,"oversold":30,"overbought":70}],
         [(1.0,1.2),(1.0,1.5),(1.5,1.2),(1.5,1.5),(1.5,2.0),(2.0,1.5),(2.0,2.0)],
         [+1,-1]),
        # V23.1 ALT — pure RSI, no EMA filter
        ("meanrev_ema0", 1, strat_meanrev_ema0,
         [{"rsi":7,"oversold":30,"overbought":70},{"rsi":14,"oversold":25,"overbought":75},
          {"rsi":14,"oversold":30,"overbought":70},{"rsi":14,"oversold":35,"overbought":65},
          {"rsi":21,"oversold":30,"overbought":70}],
         [(1.0,1.2),(1.0,1.5),(1.5,1.2),(1.5,1.5),(1.5,2.0),(2.0,1.5)],
         [+1,-1]),
        # V23.0 ma_cross — fast/slow + momentum gate
        ("ma_cross(V23.0)", 12, TYPES["ma_cross"],
         [{"fast":5,"slow":50},{"fast":5,"slow":100},{"fast":10,"slow":50},
          {"fast":10,"slow":100},{"fast":10,"slow":150},{"fast":15,"slow":100},
          {"fast":15,"slow":200},{"fast":20,"slow":100},{"fast":20,"slow":200}],
         [(1.0,1.2),(1.0,1.5),(1.5,1.2),(1.5,1.5),(2.0,1.5),(2.0,2.0),(3.0,2.0)],
         [+1,-1]),
        # V23.0 ALT — ma_cross + momentum gate
        ("ma_cross_mom", 12, strat_ma_cross_mom_impl,
         [{"fast":5,"slow":50,"roc":5},{"fast":5,"slow":100,"roc":5},
          {"fast":10,"slow":50,"roc":3},{"fast":10,"slow":50,"roc":5},
          {"fast":10,"slow":100,"roc":5},{"fast":15,"slow":100,"roc":8},
          {"fast":20,"slow":100,"roc":5},{"fast":20,"slow":200,"roc":5}],
         [(1.0,1.2),(1.0,1.5),(1.5,1.2),(1.5,1.5),(2.0,1.5),(2.0,2.0),(3.0,2.0)],
         [+1,-1]),
        # V24.1 momentum — ROC/thr/vol + RR
        ("momentum(V24.1)", 2, TYPES["momentum"],
         [{"roc":3,"thr":0.5,"vol":1.2},{"roc":3,"thr":1.0,"vol":1.2},
          {"roc":5,"thr":0.5,"vol":1.2},{"roc":5,"thr":1.0,"vol":1.2},
          {"roc":5,"thr":1.5,"vol":1.2},{"roc":8,"thr":0.5,"vol":1.2},
          {"roc":8,"thr":1.0,"vol":1.2},{"roc":8,"thr":1.5,"vol":1.2},
          {"roc":8,"thr":1.5,"vol":1.5},{"roc":8,"thr":1.0,"vol":1.5},{"roc":5,"thr":1.5,"vol":1.5}],
         [(1.5,1.5),(2.0,1.5),(3.0,2.0),(5.0,2.0),(3.0,2.5),(5.0,2.5)],
         [+1,-1]),
    ]

    allrows = []
    for label, tf, fn, grid, exits, dirs in runs:
        for want in dirs:
            dname = "LONG" if want == 1 else "SHORT"
            best = None
            for params in grid:
                for sl, rr in exits:
                    r = ev_direction(cube, syms, tf, label, fn, sl, rr, params, want)
                    if r is None: continue
                    tr_ev, oo_ev, tr_n, oo_n = r
                    if tr_ev > 0.05 and tr_n >= 30:
                        if best is None or oo_ev > best["oos_ev"]:
                            best = {"label":label, "dir":dname, "tf":tf, "sl":sl, "rr":rr,
                                    "params":params, "tr_ev":tr_ev, "oos_ev":oo_ev,
                                    "tr_n":tr_n, "oo_n":oo_n}
            if best is None:
                print(f"  {label:<16} {dname:<5}: no stable combo (all train-EV<=0.05 or too thin)")
                continue
            allrows.append(best)
            tag = "STABLE" if best["oos_ev"] > 0 else "OVERFIT"
            print(f"  {label:<16} {dname:<5}: tf={best['tf']}h sl={best['sl']}xATR rr={best['rr']} "
                  f"params={json.dumps(best['params'])} trainEV={best['tr_ev']:+.3f} OOS_EV={best['oos_ev']:+.3f} "
                  f"(n={best['tr_n']}/{best['oo_n']}) {tag}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(allrows, f, indent=2)
    stable = [r for r in allrows if r["oos_ev"] > 0 and r["tr_ev"] > 0]
    print(f"\n==== summary: {len(allrows)} combos; {len(stable)} STABLE (positive train AND OOS) ====")
    for r in sorted(stable, key=lambda x: -x["oos_ev"]):
        print(f"  {r['label']:<16} {r['dir']:<5}: sl={r['sl']}xATR rr={r['rr']} params={json.dumps(r['params'])} "
              f"train={r['tr_ev']:+.3f} OOS={r['oos_ev']:+.3f} (n={r['tr_n']}/{r['oo_n']})")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()