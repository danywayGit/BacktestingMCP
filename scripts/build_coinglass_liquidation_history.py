#!/usr/bin/env python3
"""Option B — Coinglass liquidation HISTORY backfill + forward poll.

Fetches per-pair long/short liquidation $ history from Coinglass
(`/api/futures/liquidation/history`) for the 40 major symbols, and stores it
to data/coinglass_liquidations.json for V22 backtest/reference.

NOTE: requires a Coinglass API key WITH a plan that includes liquidation/history
(minimum per docs = Hobbyist ≥4h interval, but the live API on some keys returns
401 "Upgrade plan" — if so, no data is written and the reason is printed).
The forward !forceOrder archive is the fallback ground truth until/unless this
works.

Data model (per symbol): list of {time_ms, long_liquidation_usd, short_liquidation_usd}
across the deepest available interval.

Usage: python scripts/build_coinglass_liquidation_history.py [--interval 4h] [--daily]
"""
import argparse, json, os, sys, time
import httpx

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = os.path.join(REPO, 'data', 'coinglass_liquidations.json')
BASE = 'https://open-api-v4.coinglass.com'
HIST_EP = '/api/futures/liquidation/history'


def get_key():
    # env first, then .env
    k = os.environ.get('COINGLASS_API_KEY')
    if k:
        return k
    try:
        with open(os.path.join(REPO, '.env')) as f:
            for line in f:
                line = line.strip()
                if line.startswith('COINGLASS_API_KEY='):
                    return line.split('=', 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return None


def fetch_symbol(key, symbol, interval, limit=1000):
    r = httpx.get(BASE + HIST_EP, headers={'CG-API-KEY': key},
                  params={'exchange': 'Binance', 'symbol': symbol,
                          'interval': interval, 'limit': limit}, timeout=25)
    j = r.json()
    if j.get('code') != '0' or j.get('data') is None:
        return None, j.get('msg') or j.get('code'), r.status_code
    return j['data'], None, r.status_code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--interval', default='4h', help='1m..1d etc')
    ap.add_argument('--symbols', default='', help='comma BTCUSDT')
    args = ap.parse_args()

    key = get_key()
    if not key:
        print('ERROR: no COINGLASS_API_KEY found (env or BacktestingMCP/.env)')
        return 2
    print(f'key present: {key[:6]}...')

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
    else:
        mp = os.path.join(REPO, 'data', 'major_symbols.json')
        if os.path.exists(mp):
            symbols = [s + 'USDT' for s in json.load(open(mp))['symbols_bare']]
        else:
            symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']

    all_data = {}
    ok = blocked = errored = 0
    for sym in symbols:
        data, err, code = fetch_symbol(key, sym, args.interval)
        if data is not None:
            all_data[sym] = data
            ok += 1
            print(f'  {sym}: {len(data)} pts', flush=True)
        else:
            msg = f'{err} (HTTP {code})'
            if '401' in msg or 'Upgrade' in msg:
                blocked += 1
            else:
                errored += 1
            print(f'  {sym}: {msg}', flush=True)
        time.sleep(0.25)

    print(f'\nOK={ok}  plan-blocked={blocked}  other-errors={errored}  of {len(symbols)}')
    if ok == 0:
        print('NO data written — Coinglass plan lacks liquidation/history access '
              '(401 Upgrade plan). The forward !forceOrder archive remains the '
              'ground truth. Upgrade the key plan to enable B.')
        return 1
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump({'updated': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   'interval': args.interval, 'exchange': 'Binance',
                   'symbols_ok': ok, 'data': all_data}, f, indent=2)
    print(f'Wrote {OUT} ({ok} symbols, {args.interval})')
    return 0


if __name__ == '__main__':
    sys.exit(main())