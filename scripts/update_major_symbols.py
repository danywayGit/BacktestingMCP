#!/usr/bin/env python3
"""Weekly major-crypto symbol list updater for the Liquidation (V22) strategies.

Fetches Binance USDT-M Futures 24h tickers, keeps the top-N by 24h quote
volume (i.e. the genuinely major, liquid perp symbols), excludes stablecoins,
wrapped tokens, stocks, and inactive pairs. Writes bare symbol names
(no USDT suffix) to data/major_symbols.json — used by ScoringConfig
.use_dynamic_majors (V22.0/V22.1) to restrict firing to majors only.

Run by cron weekly. No API key required (public endpoint).
"""
import json, os, sys, time
import httpx

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'BacktestingMCP'))
sys.path.insert(0, REPO_ROOT)

TOP_N = int(os.environ.get('MAJORS_TOP_N', '50'))          # keep top 50 by volume
MIN_USDT_VOLUME = float(os.environ.get('MAJORS_MIN_VOL', '20_000_000'))  # $20M+/24h
OUT = os.path.join(REPO_ROOT, 'data', 'major_symbols.json')

from src.edge_scanner.scoring_config import is_stablecoin_or_stock, get_coin_type

# Additional hard exclusions (modern Binance is swapping in odd near-stocks;
# is_stablecoin_or_stock handles wrapped+stocks+stables, EXTRA_EXCLUDE stays
# as a manual net for anything the helpers miss)
EXTRA_EXCLUDE = {'BTW', 'EUL', 'EIGEN', 'MORPHOUSDT', 'DGB'}

TICKER_URL = 'https://fapi.binance.com/fapi/v1/ticker/24hr'


def fetch_tickers() -> list:
    r = httpx.get(TICKER_URL, timeout=20)
    r.raise_for_status()
    return r.json()


def main() -> int:
    try:
        tickers = fetch_tickers()
    except Exception as exc:
        print(f'ERROR fetching Binance tickers: {exc}', file=sys.stderr)
        return 1

    rows = []
    for t in tickers:
        sym = str(t.get('symbol', ''))                     # e.g. BTCUSDT
        if not sym.endswith('USDT'):
            continue
        base = sym[:-4]
        qv = float(t.get('quoteVolume', 0) or 0)
        if qv < MIN_USDT_VOLUME:
            continue
        if is_stablecoin_or_stock(base):
            continue
        if base in EXTRA_EXCLUDE:
            continue
        # Only REAL crypto: Binance is listing tokenized stocks (XAU/XAG/SOXL/
        # MSTR/SNDK/CL/G...) and volume-farm spam (龙虾/PUMP). A symbol only
        # qualifies as a "major crypto" if the scanner knows its coin type
        # (LAYER1/LAYER2/DEFI/MEME/AI/INFRA/GAMING). Unknown == junk → drop.
        if get_coin_type(base) == 'OTHER':
            continue
        rows.append({'base': base, 'symbol': sym, 'quoteVolume_24h': round(qv, 2)})

    rows.sort(key=lambda r: -r['quoteVolume_24h'])
    majors = rows[:TOP_N]
    bare = [r['base'] for r in majors]

    payload = {
        'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'count': len(bare),
        'min_quoteVolume_24h': MIN_USDT_VOLUME,
        'top_n': TOP_N,
        'symbols_bare': bare,
        'symbols_usdt': [r['symbol'] for r in majors],
        'detail': majors,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(payload, f, indent=2)

    # Compact digest for the Telegram Liquidation (V22) topic / weekly log.
    print(f"🔄 *Majors list updated weekly — {payload['updated_utc']}*")
    print(f"  Count: {len(bare)} | 24h vol ≥ ${MIN_USDT_VOLUME:,.0f} (top {TOP_N} by vol)")
    print(f"  Symbols: {', '.join(bare)}")
    print("  V22 params: 22.0 LONG 1.5×ATR/rr1.2 | 22.1 SHORT 1.5×ATR/rr2.0 | majors-only (22.2 DISABLED)")
    print("  Config: placed in " + OUT)
    return 0


if __name__ == '__main__':
    sys.exit(main())