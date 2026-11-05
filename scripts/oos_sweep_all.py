#!/usr/bin/env python3
"""oos_sweep_all.py — sweep EVERY config through the out-of-sample gate.

Generalizes oos_validation.py from the 5 hand-picked configs to the full family:
for each config that has >= MIN_SYMBOLS usable symbols, pick its best MTF-gated
combo on the first 60% (TRAIN) of each symbol, replay on the unseen 40% (OOS),
and classify:
    PASS*  : OOS EV > 0 AND >= MIN_OOS_TRADES total OOS trades
    WEAK   : OOS EV > 0 but too few trades to trust
    OVERFIT: OOS EV <= 0 (positive in sample, dies unseen) — DO NOT ROUTE
    no-data: not enough symbols/history

This is the gate that protects prop money: configs positive in hindsight are
rejected when they fail unseen data (the exact pattern behind the 3 lost
accounts).

Output: results/_oos_sweep_all.json + printed ranking. Compute-only.
"""
import json, os, sqlite3
from collections import defaultdict
import numpy as np

DB = "/home/hermes/BacktestingMCP/data/crypto.db"
OUT = "/home/hermes/BacktestingMCP/results/_oos_sweep_all.json"
EMA_N = 50
GRID_STOPS = [2.0, 2.5, 3.0, 3.5]
GRID_RRS   = [1.0, 1.2, 1.5, 2.0]
GRID_HOLDS = [0, 24, 48]
TRAIN_FRAC = 0.60
MIN_TRAIN_TRADES = 4
MIN_SYMBOLS = 4       # need >= this many symbols with data to judge a config
MIN_OOS_TRADES = 25   # total OOS trades needed to call a config PASS*

DIR_SIGN = 1.0        # +1 LONG, -1 SHORT — set via env OOS_DIR for bias diagnosis


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


def ohlc(sym):
    return sym if sym.endswith("USDT") else sym + "USDT"


def main():
    conn = sqlite3.connect(DB)
    avail = dict(conn.execute("SELECT symbol,COUNT(*) FROM market_data WHERE timeframe='1h' GROUP BY symbol").fetchall())
    # per config: symbols with enough history
    cfg_syms = defaultdict(list)
    rows = conn.execute(
        "SELECT config_version, COALESCE(symbol,pair) sym, COUNT(*) FROM edge_signals WHERE status='RESOLVED' GROUP BY 1,2").fetchall()
    for cfg, sym, sigs in rows:
        s = ohlc(sym)
        if s in avail and avail[s] >= int(max(3 * EMA_N, 300) / TRAIN_FRAC):
            cfg_syms[cfg].append(s)

    results = {}
    n_configs = len(cfg_syms)
    print(f"sweeping {n_configs} configs through OOS gate...")
    for idx, (cfg, syms) in enumerate(cfg_syms.items()):
        if len(syms) < MIN_SYMBOLS:
            results[cfg] = {"status": "no-data", "n_symbols": len(syms)}
            continue
        # pick best combo on TRAIN
        train_evs = defaultdict(list)
        for sym in syms:
            rows = conn.execute(
                "SELECT timestamp,open,high,low,close FROM market_data WHERE symbol=? AND timeframe='1h' ORDER BY timestamp", (sym,)).fetchall()
            tss = [r[0] for r in rows]; opn = np.asarray([r[1] for r in rows], float)
            hi = np.asarray([r[2] for r in rows], float); lo = np.asarray([r[3] for r in rows], float)
            cl = np.asarray([r[4] for r in rows], float)
            split = int(len(cl) * TRAIN_FRAC)
            ent = gated_entries(hi[:split], lo[:split], opn[:split], cl[:split], tss[:split])
            for sm in GRID_STOPS:
                for rr in GRID_RRS:
                    for hd in GRID_HOLDS:
                        r = sim(hi[:split], lo[:split], opn[:split], cl[:split], ent, sm, rr, hd)
                        if r["trades"] >= MIN_TRAIN_TRADES:
                            train_evs[(sm, rr, hd)].append(r["ev_r"])
        if not train_evs:
            results[cfg] = {"status": "no-train", "n_symbols": len(syms)}
            continue
        best = max(train_evs.items(), key=lambda kv: np.median(kv[1]))[0]
        sm, rr, hd = best
        # replay on OOS
        oos_evs = []; oos_trades = []
        for sym in syms:
            rows = conn.execute(
                "SELECT timestamp,open,high,low,close FROM market_data WHERE symbol=? AND timeframe='1h' ORDER BY timestamp", (sym,)).fetchall()
            tss = [r[0] for r in rows]; opn = np.asarray([r[1] for r in rows], float)
            hi = np.asarray([r[2] for r in rows], float); lo = np.asarray([r[3] for r in rows], float)
            cl = np.asarray([r[4] for r in rows], float)
            split = int(len(cl) * TRAIN_FRAC)
            etr = gated_entries(hi[:split], lo[:split], opn[:split], cl[:split], tss[:split])
            eos = gated_entries(hi[split:], lo[split:], opn[split:], cl[split:], tss[split:])
            rtr = sim(hi[:split], lo[:split], opn[:split], cl[:split], etr, sm, rr, hd)
            ros = sim(hi[split:], lo[split:], opn[split:], cl[split:], eos, sm, rr, hd)
            if ros["trades"] > 0:
                oos_evs.append(ros["ev_r"]); oos_trades.append(ros["trades"])
        tot_trades = sum(oos_trades)
        oos_med = float(np.median(oos_evs)) if oos_evs else None
        if oos_med is None:
            status = "no-oos"
        elif oos_med > 0 and tot_trades >= MIN_OOS_TRADES:
            status = "PASS*"
        elif oos_med > 0:
            status = "WEAK"
        else:
            status = "OVERFIT"
        results[cfg] = {"status": status, "n_symbols": len(syms),
                        "chosen_combo": {"stop": sm, "rr": rr, "hold": hd},
                        "oos": {"median_ev": oos_med, "total_trades": tot_trades}}
        if (idx + 1) % 10 == 0:
            print(f"  ...{idx+1}/{n_configs} done")

    conn.close()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)

    order = {"PASS*": 0, "WEAK": 1, "OVERFIT": 2, "no-data": 3, "no-train": 4, "no-oos": 5}
    ranked = sorted(results.items(), key=lambda kv: (order.get(kv[1]["status"], 9), -(
        kv[1].get("oos", {}).get("median_ev") or 0)))
    print(f"\nwrote {OUT}")
    print(f"\n{'cfg':>7} {'verdict':<9} {'syms':>4} {'combo':>15} {'OOS_EV':>8} {'OOS#':>7}")
    for cfg, r in ranked:
        if r["status"] in ("no-data", "no-train", "no-oos"):
            print(f"{cfg:>7} {r['status']:<9} {r.get('n_symbols',0):>4}")
            continue
        cb = r["chosen_combo"]
        print(f"{cfg:>7} {r['status']:<9} {r['n_symbols']:>4} s{cb['stop']} r{cb['rr']} h{cb['hold']:>2} "
              f"{r['oos']['median_ev']:>+8.3f} {r['oos']['total_trades']:>7}")


if __name__ == "__main__":
    main()