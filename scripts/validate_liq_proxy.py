#!/usr/bin/env python3
"""Cross-validate the OI+taker proxy against real !forceOrder archive events.

The proxy (data/liq_proxy_history.json) flags OI-drop+taker-burst clusters as a
stand-in for real liquidations. This script checks HOW OFTEN a real forward
liquidation event (data/liquidation_history/*.jsonl) falls inside a proxy cluster
hour — i.e. does the proxy actually see the cascades the real feed records?

Scores each proxy cluster by real $ volume and count observed in its 1h bar.
Also flags "phantom" clusters (flagged by proxy, zero real liq) and "missed"
bars (real liq present, proxy silent). Honest framing: precision of the proxy.
Output: data/liq_proxy_validation.json

Usage: python scripts/validate_liq_proxy.py
"""
import json, os, sys
from collections import defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROXY = os.path.join(REPO, 'data', 'liq_proxy_history.json')
HIST_DIR = os.path.join(REPO, 'data', 'liquidation_history')
OUT = os.path.join(REPO, 'data', 'liq_proxy_validation.json')


def load_real_events():
    """{hour_ts: {symbol: {'vol':$, 'n':count}}} from forward archive."""
    by_hour = defaultdict(lambda: defaultdict(lambda: {'vol': 0.0, 'n': 0}))
    if not os.path.isdir(HIST_DIR):
        return by_hour
    for fn in sorted(os.listdir(HIST_DIR)):
        if not fn.endswith('.jsonl'):
            continue
        with open(os.path.join(HIST_DIR, fn)) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                ts = float(e.get('t', 0) or 0)
                hour = int(ts // 3600) * 3600
                sym = str(e.get('s', '')).replace('USDT', '').upper()
                vol = float(e.get('v', 0) or 0)
                by_hour[hour][sym]['vol'] += vol
                by_hour[hour][sym]['n'] += 1
    return by_hour


def main():
    if not os.path.exists(PROXY):
        print('No proxy history — run build_liq_proxy_history.py first.')
        return 1
    rows = json.load(open(PROXY))['rows']
    real = load_real_events()

    # Map symbol->validated rows
    clusters = [r for r in rows if r['proxy_cluster']]
    per_sym = defaultdict(list)
    for r in rows:
        per_sym[r['symbol']].append(r)

    n_clusters = len(clusters)
    n_with_real = 0
    real_vol_hit = 0.0
    missed_bars = 0
    # sample of real-verified clusters
    verified = []

    # Bars with real liq events per symbol/hour.
    # NOTE units: proxy 'ts' is Binance milliseconds; real event 't' is seconds.
    # real dict keys (from load_real_events) are SECONDS hours → convert to ms.
    real_bars = set()
    for h, syms in real.items():
        for s in syms:
            real_bars.add((s, h * 1000))  # s-hour → ms

    for r in clusters:
        h_ms = r['ts']                       # ms
        sym = r['symbol']
        liq = real.get(round(h_ms / 1000), {}).get(sym)  # look up by s-hour
        if liq:
            n_with_real += 1
            real_vol_hit += liq['vol']
            verified.append({**r, 'real_events': liq['n'], 'real_vol_usd': round(liq['vol'], 1)})
        else:
            verified.append({**r, 'real_events': 0, 'real_vol_usd': 0.0})

    # Missed: bars that had real liq volume but proxy didn't cluster
    cluster_bars = set((r['symbol'], r['ts']) for r in clusters)  # ms
    for (s, h) in real_bars:
        if (s, h) not in cluster_bars and real[h // 1000][s]['vol'] >= 250_000:
            missed_bars += 1

    precision = (n_with_real / n_clusters) if n_clusters else 0
    result = {
        'updated': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        'n_proxy_clusters': n_clusters,
        'n_clusters_with_real_liq': n_with_real,
        'precision': round(precision, 3),
        'sum_real_vol_on_clusters_usd': round(real_vol_hit, 1),
        'missed_real_liq_bars_ge250k': missed_bars,
        'verified': verified,
    }
    with open(OUT, 'w') as f:
        json.dump(result, f, indent=2)

    print(f'Proxy clusters: {n_clusters}')
    print(f'  with real liquidation in same bar: {n_with_real} (precision {precision*100:.1f}%)')
    print(f'  summed real $ volume on flagged bars: ${real_vol_hit:,.0f}')
    print(f'  real liq bars (≥$250k) the proxy MISSED: {missed_bars}')
    print(f'Wrote {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())