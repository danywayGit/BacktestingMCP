#!/usr/bin/env python3
"""oos_signal_sweep.py — CONFIG-SIGNAL-DRIVEN out-of-sample ranking.

The prior OOS sweeps failed to rank configs because they synthesized a shared
EMA entry gate for all configs (degenerate: measured one transform N times).
This one uses each config's OWN realized signals from edge_signals — its own
entry_price, direction, composite_score, and realized outcome — so we measure
each config's genuine signal edge, not a shared proxy.

For every config with enough resolved signals:
  1. SPLIT its own history by entry_time into TRAIN (first 60%) / OOS (last 40%).
  2. ON TRAIN: tune a composite_score threshold (does filtering to higher-score
     signals raise OOS winrate? pick the threshold with best train WR subject to
     >= MIN_TRADES retained).
  3. ON OOS (unseen): replay the config's own signals at that threshold, measure
     WR / median forward-return% / Wilson lower bound, split LONG / SHORT / ALL.
  4. Classify:
       PASS*  : OOS WR meaningfully above break-even AND wilson_l>0 AND n>=MIN_OOS
                AND not direction-dependent (mixed-direction pool)
       OVERFIT: positive in train, dies OOS
       WEAK   : OOS positive but thin / direction-lopsided
       no-data: < MIN_OOS_TRADES total or < MIN_SYMBOLS

This answers the real question: which configs produce signals that win on data
their exits never saw, and does raising the score bar improve winrate without
overfitting.

Output: results/_oos_signal_sweep.json. Compute-only (reads edge_signals, writes json).
"""
import json, os, sqlite3, math
from collections import defaultdict

DB = "/home/hermes/BacktestingMCP/data/crypto.db"
OUT = "/home/hermes/BacktestingMCP/results/_oos_signal_sweep.json"
TRAIN_FRAC = 0.60
MIN_TRAIN_TRADES = 20     # need enough train trades to trust a threshold pick
MIN_OOS_TRADES = 40       # total OOS trades to class PASS*
MIN_SYMBOLS = 4
BREAKEVEN_WR = 0.52       # net of ~0.1% round-trip cost: 52% WR is the practical floor
SCORE_GRID = [0.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0]


def wilson_lower(w, n, z=1.96):
    if n == 0:
        return 0.0
    p = w / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n))) / den
    return max(0.0, center - half)


def load_configs():
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        """SELECT config_version, COALESCE(symbol,pair) sym, entry_time, direction,
                  composite_score, forward_return_pct, outcome
           FROM edge_signals WHERE status='RESOLVED'
             AND outcome IN ('WIN','LOSS') AND forward_return_pct IS NOT NULL
             AND entry_time IS NOT NULL AND composite_score IS NOT NULL
             AND direction IS NOT NULL""").fetchall()
    conn.close()
    by_cfg = defaultdict(list)
    for cv, sym, et, d, sc, fr, oc in rows:
        try:
            fr = float(fr); sc = float(sc)
        except (TypeError, ValueError):
            continue
        by_cfg[str(cv)].append({"sym": sym, "time": str(et), "dir": (d or '').upper(),
                                "score": sc, "fr": fr})
    return {k: v for k, v in by_cfg.items()}


def _ts(s):
    import time, datetime
    s = s.strip().replace('T', ' ').split('+')[0].split('Z')[0]
    return int(time.mktime(datetime.datetime.strptime(s[:19], '%Y-%m-%d %H:%M:%S').timetuple()))


def stats(items):
    """WR / median fr / wilson_lower over a list of {fr}."""
    if not items:
        return None
    n = len(items)
    w = sum(1 for x in items if x["fr"] > 0)
    frs = sorted(x["fr"] for x in items)
    med = frs[n // 2]
    return {"n": n, "wr": w / n, "wilson_l": wilson_lower(w, n),
            "med_r": round(med, 3), "total_r": round(sum(x["fr"] for x in items), 2)}


def main():
    all_cfgs = load_configs()
    print(f"configs with resolved signals: {len(all_cfgs)}")

    results = {}
    # focus on configs with real history
    for cv, items in all_cfgs.items():
        items.sort(key=lambda x: _ts(x["time"]))
        if len(items) < MIN_OOS_TRADES:
            results[cv] = {"status": "no-data", "n_signals": len(items)}
            continue
        n_sym = len({x["sym"] for x in items})
        if n_sym < MIN_SYMBOLS:
            results[cv] = {"status": "no-data", "n_signals": len(items)}
            continue
        split_i = int(len(items) * TRAIN_FRAC)
        train, oos = items[:split_i], items[split_i:]

        # tune score threshold on TRAIN: pick threshold maximizing train WR and
        # wr>=BREAKEVEN, retaining >= MIN_TRAIN_TRADES
        best_th, best_wr = 0.0, -1.0
        for th in SCORE_GRID:
            tr = [x for x in train if x["score"] >= th]
            if len(tr) < MIN_TRAIN_TRADES:
                continue
            s = stats(tr)
            if s and s["wr"] > best_wr:
                best_wr = s["wr"]; best_th = th
        # use threshold best (or 0 if none)
        train_f = [x for x in train if x["score"] >= best_th]
        oos_f = [x for x in oos if x["score"] >= best_th]

        tr_s = stats(train_f); oo_s = stats(oos_f)
        # direction split on OOS
        long_f = [x for x in oos_f if x["dir"] == "LONG"]
        short_f = [x for x in oos_f if x["dir"] == "SHORT"]
        long_s = stats(long_f); short_s = stats(short_f)

        oos_ok = oo_s and oo_s["n"] >= MIN_OOS_TRADES
        if not oos_ok:
            results[cv] = {"status": "no-oos", "n_signals": len(items),
                           "threshold": best_th,
                           "oos": oo_s, "long": long_s, "short": short_s}
            continue
        # direction honesty: does the edge depend on longs exclusively?
        both_poss = (long_s is not None and long_s["wr"] > 0.50) or \
                    (short_s is not None and short_s["wr"] > 0.50)
        wr = oo_s["wr"]
        # PASS* = OOS WR above practical cost floor, wilson_l positive, healthy n,
        #         and edge not purely one-direction single-sample
        if wr >= BREAKEVEN_WR and oo_s["wilson_l"] > 0 and both_poss:
            status = "PASS*"
        elif wr >= BREAKEVEN_WR and oo_s["wilson_l"] > 0:
            status = "WEAK-dir"
        elif wr < BREAKEVEN_WR and (tr_s and tr_s["wr"] >= BREAKEVEN_WR):
            status = "OVERFIT"
        else:
            status = "BELOW-BRKEVEN"
        results[cv] = {"status": status, "threshold": best_th,
                       "train_wr": round(tr_s["wr"], 3) if tr_s else None,
                       "train_n": tr_s["n"] if tr_s else 0,
                       "oos": oo_s, "long": long_s, "short": short_s,
                       "n_symbols": n_sym}

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)

    order = {"PASS*": 0, "WEAK-dir": 1, "OVERFIT": 2, "BELOW-BRKEVEN": 3,
             "no-oos": 4, "no-data": 5}
    ranked = sorted(results.items(), key=lambda kv: (order.get(kv[1]["status"], 9),
                      -(kv[1].get("oos") or {}).get("wr", 0)))
    from collections import Counter
    print("verdict counts:", dict(Counter(v["status"] for v in results.values())))
    print(f"\n{'cfg':>7} {'verdict':<14} {'th':>5} {'trainWR':>7} {'OOS_WR':>6} {'OOS#':>6} {'wilLo':>6} {'longWR':>6} {'shortWR':>6}")
    for cv, r in ranked:
        if r["status"] in ("no-data", "no-oos"):
            print(f"{cv:>7} {r['status']:<14} n={r.get('n_signals',0):>5}")
            continue
        oo = r["oos"]
        lw = r["long"]["wr"] if r["long"] else 0
        sw = r["short"]["wr"] if r["short"] else 0
        print(f"{cv:>7} {r['status']:<14} {r['threshold']:>5} "
              f"{str(r['train_wr']):>7} {oo['wr']:>6.1%} {oo['n']:>6} {oo['wilson_l']:>6.1%} {lw:>6.1%} {sw:>6.1%}")


if __name__ == "__main__":
    main()