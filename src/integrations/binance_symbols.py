"""
Cached Binance Futures symbol list — used to filter candidates
to only tradeable USDT-M perpetuals.

This prevents the scanner from scoring, logging, and tracking
symbols that don't exist on Binance Futures.
"""

from __future__ import annotations

import logging
import os
from typing import FrozenSet, Optional

logger = logging.getLogger(__name__)

# Cache: populated on first call to get_binance_futures_symbols()
_BINANCE_FUTURES: Optional[FrozenSet[str]] = None


def get_binance_futures_symbols() -> FrozenSet[str]:
    """Return all USDT-M perpetual base symbols available on Binance Futures.

    Cached in memory for the lifetime of the process.
    Falls back to a static list if the exchange is unreachable.
    """
    global _BINANCE_FUTURES
    if _BINANCE_FUTURES is not None:
        return _BINANCE_FUTURES

    # Fallback static list for when network is unavailable
    # Core ~150 most liquid USDT-M perpetuals
    _FALLBACK: FrozenSet[str] = frozenset({
        "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX", "DOT",
        "LINK", "MATIC", "UNI", "SHIB", "LTC", "ATOM", "ETC", "XLM",
        "BCH", "TRX", "FIL", "APT", "ARB", "OP", "NEAR", "INJ", "SUI",
        "SEI", "TIA", "IMX", "PEPE", "FLOKI", "BONK", "WIF", "RUNE",
        "AAVE", "CRV", "MKR", "SNX", "COMP", "YFI", "SUSHI", "CAKE",
        "GMX", "DYDX", "PENDLE", "LDO", "EIGEN", "ENA", "ETHFI", "JTO",
        "ONDO", "ICP", "EGLD", "ALGO", "FTM", "FLOW", "KSM", "ZIL",
        "VET", "HBAR", "EOS", "XTZ", "ANKR", "IOST", "HOT", "DGB",
        "WAVES", "OMG", "BAT", "ZRX", "KNC", "BAND", "NMR", "STMX",
        "RSR", "QNT", "ZEC", "DASH", "XMR", "SAND", "MANA", "AXS",
        "GALA", "ENJ", "CHZ", "THETA", "TFUEL", "IOTX", "FET", "AGIX",
        "OCEAN", "GRT", "RLC", "CVC", "SKL", "API3", "TRB", "AUDIO",
        "REN", "COTI", "ALICE", "TLM", "DENT", "CKB", "CELO", "NKN",
        "AR", "CTSI", "POWR", "BICO", "DEXE", "XVS", "AERO", "JUP",
        "PYTH", "W", "ALT", "TIA", "STRK", "ZRO", "BLAST", "MANTA",
        "PIXEL", "PRIME", "PORTAL", "OMNI", "DYM", "SAGA", "TNSR",
        "WLD", "ENA", "ETHFI", "AEVO", "ZETA", "NMT",
    })

    try:
        import ccxt
        ex = ccxt.binanceusdm({
            'timeout': 10000,
            'enableRateLimit': True,
        })
        markets = ex.load_markets()
        symbols: set[str] = set()
        for sym, info in markets.items():
            # Binance USDT-M perpetuals: type='swap', linear=True, ends with :USDT
            if sym.endswith(':USDT') and info.get('swap') and info.get('linear'):
                base = sym.replace('/USDT:USDT', '')
                symbols.add(base)
        if symbols:
            _BINANCE_FUTURES = frozenset(symbols)
            logger.info("Binance Futures symbols loaded: %d", len(symbols))
            return _BINANCE_FUTURES
    except Exception as exc:
        logger.warning("Could not fetch Binance Futures markets: %s — using fallback list", exc)

    _BINANCE_FUTURES = _FALLBACK
    logger.info("Binance Futures symbols (fallback): %d", len(_FALLBACK))
    return _BINANCE_FUTURES


def is_on_binance_futures(symbol: str) -> bool:
    """Check if a symbol is tradeable on Binance Futures USDT-M."""
    return symbol.upper() in get_binance_futures_symbols()
