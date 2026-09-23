#!/usr/bin/env python3
"""Option C — Free reconstruction of liquidation-cluster proxy features.

Binance has no public historical liquidation-event feed, but DOES expose ~30 days
of per-symbol history for open interest, taker order-flow, and long/short ratios.
A liquidation cascade leaves a recognizable signature in that data:
  * sharp Open Interest DROP (forced closes unwind positions), and
  * a burst of one-directional aggressive taker volume, and
  * an extreme long/short account ratio (crowded side gets liquidated).

This script reconstructs a "liquidation-cluster proxy" precursor per symbol per
1h bar over the ~30-day Binance retention window, so the V22 trigger hypothesis
(volume bar + imbalance) can be backtested TODAY instead of waiting weeks for the
forward !forceOrder archive to fill. It is a PROXY, not ground-truth liquidations
— honest about that in every field (source='oi_taker_proxy').

Output: data/liq_proxy_history.json  (and data/liq_proxy_history.csv)
Run by cron weekly (Weds); extends/refreshes the window each run.

Usage: python scripts/build_liq_proxy_history.py [--symbols BTCUSDT,ETHUSDT] [--period 1h] [--days 30]
"""
import argparse, csv, httpx, json, math, os, sys, time
from datetime import datetime, timezone

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT_JSON = os.path.join(REPO, 'data', 'liq_proxy_history.json')
OUT_CSV = os.path.join(REPO, 'data', 'liq_proxy_history.csv')

FAPI = 'https://fapi.binance.com/futures/data'
PAGE = 744  # max results per request

# Proxy thresholds calibrated 2026-09-20 to REAL 1h data on BTC/ETH/BNB:
#   OI-drop p95 ≈ 0.52%, max 2.2%      → 0.5% is a genuine forced-unwind bar
#   taker_ratio p90 ≈ 2.07             → 1.8 catches bursts but not all bars
#   |buy_frac-0.5| p95 ≈ 0.11, max .19 → 0.10 = one-directional (0.55 impossible)
OI_DROP_PCT = 0.50     # OI falls ≥0.5% (≈95th pct) → real unwind, not noise
TAKER_BURST = 1.8      # taker vol ≥1.8× trailing avg (≈p90)
TAKER_BIAS = 0.10      # |buy-bias| ≥0.10 of taker vol (≈p95) → one-directional
LS_EXTREME = None      # (informational only)


def ts_iso(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def fetch_series(endpoint, symbol, period, start_time=0, limit=PAGE):
    """Page /futures/data/[endpoint] from (optionally) start_time forward/back;
    returns list of dicts sorted ascending by timestamp."""
    params = {'symbol': symbol, 'period': period, 'limit': limit}
    if start_time:
        params['startTime'] = start_time
    rows = []
    cursor = start_time
    for _ in range(8):  # ~30 days at 1h = up to 744 bars; a few pages
        p = dict(params)
        if cursor:
            p['startTime'] = cursor
        r = httpx.get(f'{FAPI}/{endpoint}', params=p, timeout=20)
        r.raise_for_status()
        d = r.json()
        if not isinstance(d, list) or not d:
            break
        rows.extend(d)
        # move cursor to oldest returned
        next_c = min(x['timestamp'] for x in d)
        if next_c >= cursor or len(d) < limit:
            break
        cursor = next_c
        time.sleep(0.15)
    # dedupe + sort ascending
    seen, uniq = set(), []
    for x in sorted(rows, key=lambda r: r['timestamp']):
        k = x['timestamp']
        if k in seen:
            continue
        seen.add(k)
        uniq.append(x)
    return uniq


def build_symbol(symbol, period):
    """Return list of per-bar liquidation-cluster proxy rows for one symbol."""
    try:
        oi = fetch_series('openInterestHist', symbol, period)
        tk = fetch_series('takerlongshortRatio', symbol, period)
        ls = fetch_series('globalLongShortAccountRatio', symbol, period)
    except Exception as e:
        print(f'  [{symbol}] fetch error: {e}', file=sys.stderr)
        return []
    if not oi:
        return []

    oi_map = {x['timestamp']: float(x['sumOpenInterest']) for x in oi}
    tk_map = {x['timestamp']: (x, float(x['buyVol'] or 0), float(x['sellVol'] or 0)) for x in tk}

    oi_ts = sorted(oi_map)
    out = []
    ls_map = {x['timestamp']: x for x in ls}
    # trailing taker average (20 bars)
    taker_hist = []
    for i, t in enumerate(oi_ts):
        oiv = oi_map[t]
        prev_oiv = oi_map.get(oi_ts[i-1]) if i > 0 else oiv
        oi_drop_pct = (prev_oiv - oiv) / prev_oiv * 100 if prev_oiv else 0.0
        bx = tk_map.get(t)
        tk_ratio = tk_avg = buy_frac = None
        if bx:
            _, bv, sv = bx
            tv = bv + sv
            tk_avg = (sum(x[1] for x in taker_hist[-20:]) /
                      len(taker_hist[-20:])) if taker_hist else tv
            tk_ratio = tv / tk_avg if tk_avg else 0.0
            buy_frac = bv / tv if tv else 0.5
        taker_hist.append((t, (bx[1] + bx[2]) if bx else 0.0))

        # Proxy cluster = OI-drop (forced unwind) + taker-volume burst.
        # Calibration: on majors, |buy_frac-0.5| barely moves (p95=0.11) AND-ing
        # it kills the signal (only 3 bars pass). So cluster is 2-condition;
        # direction + ls_ratio stay informational.
        cluster = (
            oi_drop_pct >= OI_DROP_PCT
            and tk_ratio is not None and tk_ratio >= TAKER_BURST
        )
        # Proxy imbalance sign: heavy SELL taker (buy_frac<0.5) = longs liquidating → bearish
        direction = None
        if cluster:
            if buy_frac is not None and abs(buy_frac - 0.5) >= 0.05:
                direction = 'SHORT' if buy_frac < 0.5 else 'LONG'
        lsr = float(ls_map.get(t, {}).get('longShortRatio', 0) or 0) if ls_map else None
        out.append({
            'symbol': symbol.replace('USDT', ''),
            'time': ts_iso(t),
            'ts': t,
            'oi': round(oiv, 2),
            'oi_drop_pct': round(oi_drop_pct, 3),
            'taker_ratio': round(tk_ratio, 3) if tk_ratio is not None else None,
            'buy_frac': round(buy_frac, 3) if buy_frac is not None else None,
            'ls_ratio': round(lsr, 3) if lsr is not None else None,
            'proxy_cluster': cluster,
            'direction': direction,
            'source': 'oi_taker_proxy',
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', default='', help='comma BTCUSDT list; default = majors file')
    ap.add_argument('--period', default='1h')
    ap.add_argument('--days', type=int, default=30)
    args = ap.parse_args()

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
    else:
        mp = os.path.join(REPO, 'data', 'major_symbols.json')
        if os.path.exists(mp):
            symbols = [s + 'USDT' for s in json.load(open(mp))['symbols_bare']]
        else:
            symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']

    all_rows = []
    for sym in symbols:
        rows = build_symbol(sym, args.period)
        print(f'{sym}: {len(rows)} bars, {sum(1 for r in rows if r["proxy_cluster"])} clusters', flush=True)
        all_rows.extend(rows)
        time.sleep(0.2)

    all_rows.sort(key=lambda r: r['ts'])
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump({'updated': datetime.now(timezone.utc).isoformat(),
                   'period': args.period, 'count': len(all_rows),
                   'rows': all_rows}, f, indent=2)
    avecsv = []
    for r in all_rows:
        rr = dict(r)
        rr.pop('ts', None)
        avecsv.append(rr)
    if avecsv:
        with open(OUT_CSV, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(avecsv[0].keys()))
            w.writeheader()
            w.writerows(avecsv)
    clusters = sum(1 for r in all_rows if r['proxy_cluster'])
    print(f'\nWrote {len(all_rows)} rows ({clusters} clusters) → {OUT_JSON}')
    print(f'Proxy clusters: {clusters} across {len(symbols)} symbols, {args.days}d, {args.period} bars.')
    return 0


if __name__ == '__main__':
    sys.exit(main())