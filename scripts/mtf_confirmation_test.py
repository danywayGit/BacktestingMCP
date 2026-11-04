#!/usr/bin/env python3
"""mtf_confirmation_test.py — 1h + 4h multi-timeframe confirmation comparison.

Question: does requiring the higher-timeframe (4h) trend to AGREE with a 1h
signal (and optionally the 1h trend too) raise signal quality, or just trade less?

Method (pure compute, no system/exchange change):
  1. Resample our deep 1h OHLC -> synthetic 4h closes per symbol (bucket by
     aligned 4h boundary, close = last 1h close in the bucket). We do NOT trust
     the raw stored 4h column: an earlier check found it inconsistent with 1h.
  2. For every RESOLVED win/loss signal, compute at its entry-time
        t4 = sign of EMA(50)-slope on the synthetic 4h series
        t1 = sign of EMA(50)-slope on the raw 1h series
     and compare to the signal's LONG/SHORT direction `d` (+1/-1).
  3. Bucket each signal:
        UNFILTERED : all signals
        AGREE_4H   : d == t4
        AGREE_BOTH : d == t4 AND d == t1
  4. Per bucket, per-config regime robustness (wr, wilson_l, avg fwd-return %,
     % positive rolling-50 windows) + bucket-level aggregate win rate, so we can
     tell whether MTF confirmation raises quality or just cuts volume.

Output: results/_mtf_comparison.json. Compute-only.
"""
import json, os, sqlite3, math, time, datetime
from collections import defaultdict
import numpy as np

DB = "/home/hermes/BacktestingMCP/data/crypto.db"
OUT = "/home/hermes/BacktestingMCP/results/_mtf_comparison.json"

EMA_N = 50         # trend EMA bars
MIN_TRADES_CFG = 50
WIN = 50
SLIDE = 25


def _ema_closes(closes):
    a = np.asarray(closes, dtype=float)
    n = len(a)
    if n == 0:
        return np.array([])
    alpha = 2.0 / (EMA_N + 1)
    out = np.empty(n)
    s = a[0]; out[0] = s
    for i in range(1, n):
        s = alpha * a[i] + (1 - alpha) * s
        out[i] = s
    return out


def _slope(ema_arr):
    """+1 if rising (last >= mean of preceding window), -1 falling, 0 flat."""
    if len(ema_arr) < 4:
        return 0.0
    return 1.0 if ema_arr[-1] >= float(np.mean(ema_arr[-4:-1])) else -1.0


def wilson_lower(w, n, z=1.96):
    if n == 0:
        return 0.0
    p = w / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n))) / den
    return max(0.0, center - half)


def ts_of(s):
    s = s.strip().replace('T', ' ').split('+')[0].split('Z')[0]
    dt = datetime.datetime.strptime(s[:19], '%Y-%m-%d %H:%M:%S')
    return int(time.mktime(dt.timetuple()))


def ohlc_name(sym):
    return sym if sym.endswith('USDT') else sym + 'USDT'


def main():
    conn = sqlite3.connect(DB)
    # signals we care about + their symbol set
    sig_rows = conn.execute(
        """SELECT config_version, COALESCE(symbol,pair) sym, entry_time,
                  direction, forward_return_pct, outcome
           FROM edge_signals WHERE status='RESOLVED'
             AND outcome IN ('WIN','LOSS') AND forward_return_pct IS NOT NULL
             AND entry_time IS NOT NULL""").fetchall()
    # min/max entry time to bound the OHLC load to the signal window
    t_lo = ts_of(min((r[2] for r in sig_rows), key=ts_of))
    t_hi = ts_of(max((r[2] for r in sig_rows), key=ts_of))
    ohlc_syms = set(r[0] for r in conn.execute(
        "SELECT DISTINCT symbol FROM market_data WHERE timeframe='1h'"))
    need = {ohlc_name(r[1]) for r in sig_rows} & ohlc_syms
    print(f"signals={len(sig_rows)} symbols_needed={len(need)} window={t_lo}..{t_hi}")

    # per symbol: 1h slope map {1h_ts: t1}, 4h slope map {aligned4h_ts: t4}
    t1_map = {}
    t4_map = {}
    m1_keys = {}
    m4_keys = {}
    for sym in need:
        rows = conn.execute(
            "SELECT timestamp, close FROM market_data WHERE symbol=? AND timeframe='1h' "
            "AND timestamp>=? AND timestamp<=? ORDER BY timestamp",
            (sym, t_lo - 50 * 3600, t_hi)).fetchall()
        if not rows:
            continue
        tss = [r[0] for r in rows]
        closes = [float(r[1]) for r in rows]
        ema1 = _ema_closes(closes)
        m1 = {}
        for i in range(len(tss)):
            if i >= EMA_N - 1:
                m1[tss[i]] = _slope(ema1[max(0, i - EMA_N + 1):i + 1])
        t1_map[sym] = m1
        m1_keys[sym] = list(m1.keys())
        # 4h resample: bucket
        b4 = {}
        order = []
        for i, t in enumerate(tss):
            bb = (t // 14400) * 14400
            if bb not in b4:
                b4[bb] = closes[i]; order.append(bb)
            else:
                b4[bb] = closes[i]
        ema4 = _ema_closes([b4[x] for x in order])
        m4 = {}
        for j, bb in enumerate(order):
            if j >= EMA_N - 1:
                m4[bb] = _slope(ema4[max(0, j - EMA_N + 1):j + 1])
        t4_map[sym] = m4
        m4_keys[sym] = list(m4.keys())
    conn.close()
    print("trend maps built")

    dirmap = {'LONG': 1.0, 'SHORT': -1.0, 'BUY': 1.0, 'SELL': -1.0}

    B = {'UNFILTERED': [], 'AGREE_4H': [], 'AGREE_BOTH': []}
    skipped = 0
    for cfg, sym, et, dirs, fr, oc in sig_rows:
        d = dirmap.get((dirs or '').upper())
        if d is None or ohlc_name(sym) not in t1_map:
            skipped += 1
            continue
        ts = ts_of(et)
        # floor to bar boundaries; fall back to the most recent prior bar (bisect)
        import bisect
        b1 = (ts // 3600) * 3600
        m1 = t1_map[ohlc_name(sym)]
        i1 = bisect.bisect_right(m1_keys[ohlc_name(sym)], b1) - 1
        t1 = m1[m1_keys[ohlc_name(sym)][i1]] if i1 >= 0 else None
        b4 = (ts // 14400) * 14400
        m4 = t4_map[ohlc_name(sym)]
        i4 = bisect.bisect_right(m4_keys[ohlc_name(sym)], b4) - 1
        t4 = m4[m4_keys[ohlc_name(sym)][i4]] if i4 >= 0 else None
        if t1 is None or t4 is None:
            skipped += 1
            continue
        rec = (str(cfg), str(et), float(fr), str(oc), t4, t1)
        B['UNFILTERED'].append(rec)
        if d == t4:
            B['AGREE_4H'].append(rec)
            if d == t1:
                B['AGREE_BOTH'].append(rec)

    print("bucketed:", {k: len(v) for k, v in B.items()}, "skipped:", skipped)

    out = {}
    for bucket, recs in B.items():
        by_cfg = defaultdict(list)
        for cfg, et, fr, oc, t4, t1 in recs:
            by_cfg[cfg].append((et, fr))
        agg_w = agg_l = 0
        cfgs = {}
        for cfg, items in by_cfg.items():
            items.sort(key=lambda x: x[0])
            frs = [x[1] for x in items]
            n = len(frs)
            w = sum(1 for x in frs if x > 0)
            winwins = [sum(frs[i:i + WIN]) for i in range(0, n - WIN + 1, SLIDE)]
            ww = sum(1 for x in winwins if x > 0) / len(winwins) if winwins else 0.0
            cfgs[cfg] = {'n': n, 'wr': round(w / n, 4),
                         'wilson_l': round(wilson_lower(w, n), 4),
                         'avg_r': round(sum(frs) / n, 4),
                         'win_windows': round(ww, 3)}
            agg_w += w; agg_l += (n - w)
        n = agg_w + agg_l
        out[bucket] = {'n_signals': n, 'wr': round(agg_w / n, 4) if n else 0.0,
                       'wilson_l': round(wilson_lower(agg_w, n), 4),
                       'configs': cfgs}

    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"\nwrote {OUT}")
    print("\n=== BUCKET AGGREGATE ===")
    print(f"{'bucket':>10} {'sig#':>6} {'WR%':>6} {'wilsonLo':>8}")
    for b in ('UNFILTERED', 'AGREE_4H', 'AGREE_BOTH'):
        d = out[b]
        print(f"{b:>10} {d['n_signals']:>6} {d['wr']*100:>6.1f} {d['wilson_l']*100:>8.1f}")

    for b in ('UNFILTERED', 'AGREE_4H', 'AGREE_BOTH'):
        cfgs = [(c, m) for c, m in out[b]['configs'].items() if m['n'] >= MIN_TRADES_CFG]
        cfgs.sort(key=lambda x: -x[1]['wilson_l'])
        print(f"\n--- {b}: top configs (n>={MIN_TRADES_CFG}) ---")
        for c, m in cfgs[:6]:
            print(f"  {c:>6} n={m['n']:>5} WR={m['wr']*100:5.1f}% wilLo={m['wilson_l']*100:5.1f}% "
                  f"avgR={m['avg_r']:+.2f}% winWin={m['win_windows']*100:.0f}%")


if __name__ == '__main__':
    main()