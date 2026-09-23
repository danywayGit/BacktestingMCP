#!/usr/bin/env python3
"""Historical backtest of V22 trade mechanics using the liquidation PROXY trigger.

Real historical *liquidation-event* data doesn't exist on Binance (no free
feed, ~30d ratio/OI only). So this backtests V22's actual trade rules —
the same ones the live V22 configs use: wide stop (atr_stop_mult) + RR target
+ forward-OHLC resolution — using the OI-drop+taker-burst proxy clusters
(data/liq_proxy_history.json) as the entry trigger, over real historical 1h OHLC.

This answers: "if the proxy had triggered V22 on these historical bars, what
would the trades have returned?" — an honest historical simulation of the
liquidation-signal hypothesis, with clear 'proxy' labeling.

Metrics per config: sign-corrected EV in R-multiples, win-rate, hit-rate,
avg ±%, n trades, by symbol. Horizons resolve to SL/TP intrabar (conservative:
if both touched, whichever is nearer the open wins).

Usage: python scripts/backtest_liq_proxy.py [--stop 4.0] [--rr 6.0] [--min-dir-signal]
"""
import argparse, bisect, json, os, sqlite3, sys
from collections import defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROXY = os.path.join(REPO, 'data', 'liq_proxy_history.json')
DB = os.path.join(REPO, 'data', 'crypto.db')
HORIZON_H = 96          # max wait for SL/TP (0 = no time stop; use 96h cap)
ATR_PERIOD = 14


def load_ohlc(conn):
    syms = set(r['symbol'] for r in json.load(open(PROXY))['rows'])
    o = {}
    for s in syms:
        rows = conn.execute(
            "SELECT timestamp,open,high,low,close FROM market_data "
            "WHERE timeframe='1h' AND symbol=? ORDER BY timestamp", (s + 'USDT',)).fetchall()
        o[s] = {'ts': [r[0] for r in rows], 'open': [r[1] for r in rows],
                'high': [r[2] for r in rows], 'low': [r[3] for r in rows],
                'close': [r[4] for r in rows]}
    return o


def atr_at(ch, idx):
    if idx < ATR_PERIOD:
        return None
    trs = []
    for i in range(idx - ATR_PERIOD, idx):
        h = ch['high'][i]; l = ch['low'][i]; c = ch['close'][i-1] if i > 0 else ch['close'][i]
        trs.append(max(h - l, abs(h - c), abs(l - c)))
    return sum(trs) / ATR_PERIOD


def sim(cluster, ch, stop_mult, rr, horizon_h):
    """Resolve one proxy-cluster trade. Returns dict or None."""
    ts_ms = cluster['ts']; ts_s = ts_ms / 1000.0
    i = bisect.bisect_left(ch['ts'], ts_s)
    if i >= len(ch['ts']) - 2 or i < ATR_PERIOD:
        return None
    atr = atr_at(ch, i)
    if not atr or atr <= 0:
        return None
    entry = ch['close'][i]
    direction = cluster.get('direction')
    # entry bar close == cluster close; long if buy-heavy, short if sell-heavy
    is_long = direction == 'LONG'
    if direction is None:
        return None  # only trade direction-aligned clusters
    sl = entry - atr * stop_mult if is_long else entry + atr * stop_mult
    tp = entry + atr * stop_mult * rr if is_long else entry - atr * stop_mult * rr
    max_bars = int(horizon_h) if horizon_h else 10**9
    for k in range(i + 1, min(len(ch['ts']), i + 1 + max_bars)):
        o, h, l, c = ch['open'][k], ch['high'][k], ch['low'][k], ch['close'][k]
        if is_long:
            hit_sl = l <= sl
            hit_tp = h >= tp
        else:
            hit_sl = h >= sl
            hit_tp = l <= tp
        if hit_sl and hit_tp:
            # both touched: nearness to open decides (conservative)
            d_sl = abs(o - sl); d_tp = abs(o - tp)
            return {'outcome': 'LOSS' if d_sl <= d_tp else 'WIN', 'exit': sl if d_sl <= d_tp else tp,
                    'bars': k - i, 'entry': entry, 'atr': atr}
        if hit_sl:
            return {'outcome': 'LOSS', 'exit': sl, 'bars': k - i, 'entry': entry, 'atr': atr}
        if hit_tp:
            return {'outcome': 'WIN', 'exit': tp, 'bars': k - i, 'entry': entry, 'atr': atr}
    # time stop: exit at last close
    c_end = ch['close'][min(len(ch['ts'])-1, i + max_bars)]
    ret = (c_end - entry) / entry
    outcome = 'WIN' if (ret > 0) == is_long and abs(ret) > 0 else ('LOSS' if ret != 0 else 'FLAT')
    return {'outcome': outcome, 'exit': c_end, 'bars': max_bars, 'entry': entry, 'atr': atr, 'timeout': True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stop', type=float, default=4.0)
    ap.add_argument('--rr', type=float, default=6.0)
    ap.add_argument('--horizon', type=int, default=HORIZON_H)
    args = ap.parse_args()

    rows = json.load(open(PROXY))['rows']
    clusters = [r for r in rows if r['proxy_cluster'] and r.get('direction')]
    conn = sqlite3.connect(DB)
    ohlc = load_ohlc(conn)
    conn.close()

    print(f"Backtest V22 mechanics on {len(clusters)} PROXY trigger signals "
          f"(stop={args.stop}×ATR, rr={args.rr}, horizon={args.horizon}h, majors-only)\n")

    per_sym = defaultdict(list)
    for c in clusters:
        ch = ohlc.get(c['symbol'])
        if not ch:
            continue
        r = sim(c, ch, args.stop, args.rr, args.horizon)
        if r:
            per_sym[c['symbol']].append({**r, 'time': c['time']})

    # aggregate
    agg = defaultdict(list)
    for sym, trades in per_sym.items():
        for t in trades:
            agg[sym].append(t)
    all_trades = [t for sym in agg for t in agg[sym]]

    wins = sum(1 for t in all_trades if t['outcome'] == 'WIN')
    losses = sum(1 for t in all_trades if t['outcome'] == 'LOSS')
    flats = sum(1 for t in all_trades if t['outcome'] == 'FLAT')
    n = len(all_trades)
    wr = wins / (wins + losses) * 100 if (wins + losses) else 0
    ev_r = (wins * args.rr - losses * 1.0) / n if n else 0
    print(f"TOTAL: {n} trades | W {wins} | L {losses} | F {flats}")
    print(f"  Win-rate: {wr:.1f}%  (n={n})")
    print(f"  EV (R-multiples/trade, rr={args.rr}): {ev_r:+.3f}")
    print(f"  Hit-rate (win or flat, no stop-out): {((wins+flats)/n*100) if n else 0:.1f}%")
    print(f"  Timeout exits (forced at {args.horizon}h): {sum(1 for t in all_trades if t.get('timeout'))}/{n}")
    print("\nPer-symbol:")
    for sym, trades in sorted(agg.items(), key=lambda kv: -len(kv[1])):
        w = sum(1 for t in trades if t['outcome'] == 'WIN')
        l = sum(1 for t in trades if t['outcome'] == 'LOSS')
        f = len(trades) - w - l
        e = (w * args.rr - l * 1.0) / len(trades)
        print(f"  {sym:8} n={len(trades):3} {w}W/{l}L/{f}F  WR={w/(w+l)*100 if w+l else 0:4.0f}%  EV={e:+.2f}R")


if __name__ == '__main__':
    main()