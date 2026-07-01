"""
Binance Spot and Futures symbol lists — used to filter candidates to only tradeable USDT pairs.
"""

import logging
import os
from typing import FrozenSet, Optional

logger = logging.getLogger(__name__)

# Cache for Futures: populated on first call to get_binance_futures_symbols()
_BINANCE_FUTURES: Optional[FrozenSet[str]] = None

# Cache for Spot: populated on first call to get_binance_spot_symbols()
_BINANCE_SPOT: Optional[FrozenSet[str]] = None

# Fallback static list for Futures when network is unavailable
_FUTURES_FALLBACK: FrozenSet[str] = frozenset({
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

# Fallback static list for Spot when network is unavailable - top 300 spot USDT pairs
_SPOT_FALLBACK: FrozenSet[str] = frozenset({
    "BTC", "ETH", "USDT", "BNB", "XRP", "ADA", "SOL", "DOGE", "TRX",
    "DOT", "MATIC", "LTC", "LINK", "BCH", "XLM", "ATOM", "VET",
    "FIL", "ICP", "THETA", "ETC", "ALGO", "KLAY", "XTZ", "EOS",
    "AAVE", "UNI", "MKR", "COMP", "YFI", "SNX", "SUSHI", "KSM",
    "LUNA", "AVAX", "FTM", "NEAR", "IOTX", "RUNE", "SNZ", "HOT",
    "ZIL", "ONT", "DASH", "WAVES", "ZEN", "ONT", "BAND", "REN",
    "MKR", "BAT", "COTI", "ALICE", "TLM", "DENT", "CKB", "CELO", "NKN",
    "AR", "CTSI", "POWR", "BICO", "DEXE", "XVS", "AERO", "JUP",
    "PYTH", "W", "ALT", "TIA", "STRK", "ZRO", "BLAST", "MANTA",
    "PIXEL", "PRIME", "PORTAL", "OMNI", "DYM", "SAGA", "TNSR",
    "WLD", "ENA", "ETHFI", "AEVO", "ZETA", "NMT",
})


def get_binance_futures_symbols() -> FrozenSet[str]:
    """Return all USDT-M perpetual base symbols available on Binance Futures.

    Cached in memory for the lifetime of the process.
    Falls back to a static list if the exchange is unreachable.
    """
    global _BINANCE_FUTURES
    if _BINANCE_FUTURES is not None:
        return _BINANCE_FUTURES

    # Fallback static list for when network is unavailable
    _FALLBACK: FrozenSet[str] = _FUTURES_FALLBACK

    try:
        import ccxt
        ex = ccxt.binanceusdm({  # USDT-M futures
            'timeout': 10000,
            'enableRateLimit': True,
        })
        markets = ex.load_markets()
        symbols: set[str] = set()
        for sym, info in markets.items():
            # Binance USDT-M perpetuals: type='swap', linear=True, ends with :USDT
            if sym.endswith(':USDT') and info.get('swap') and info.get('linear'):
                base = sym.split('/')[0]  # e.g. BTC/USDT:USDT -> BTC
                symbols.add(base)
        if symbols:
            _BINANCE_FUTURES = frozenset(symbols)
            logger.info("Binance Futures symbols loaded: %d", len(symbols))
            return _BINANCE_FUTURES
    except Exception as exc:
        logger.warning("Could not fetch Binance Futures markets: %s — using fallback list", exc)

    _BINANCE_FUTURES = _FUTURES_FALLBACK
    logger.info("Binance Futures symbols (fallback): %d", len(_FUTURES_FALLBACK))
    return _BINANCE_FUTURES


def get_binance_spot_symbols() -> FrozenSet[str]:
    """Return all USDT spot base symbols available on Binance Spot.

    Cached in memory for the lifetime of the process.
    Falls back to a static list if the exchange is unreachable.
    """
    global _BINANCE_SPOT
    if _BINANCE_SPOT is not None:
        return _BINANCE_SPOT

    # Fallback static list for when network is unavailable
    _FALLBACK: FrozenSet[str] = _SPOT_FALLBACK

    try:
        import ccxt
        ex = ccxt.binance({  # Spot
            'timeout': 10000,
            'enableRateLimit': True,
        })
        markets = ex.load_markets()
        symbols: set[str] = set()
        for sym, info in markets.items():
            # Binance Spot USDT pairs: ends with /USDT, spot
            if sym.endswith('/USDT') and info.get('spot'):
                base = sym.split('/')[0]  # e.g. BTC/USDT -> BTC
                symbols.add(base)
        if symbols:
            _BINANCE_SPOT = frozenset(symbols)
            logger.info("Binance Spot symbols loaded: %d", len(symbols))
            return _BINANCE_SPOT
    except Exception as exc:
        logger.warning("Could not fetch Binance Spot markets: %s — using fallback list", exc)

    _BINANCE_SPOT = _SPOT_FALLBACK
    logger.info("Binance Spot symbols (fallback): %d", len(_SPOT_FALLBACK))
    return _BINANCE_SPOT


def is_on_binance_futures(symbol: str) -> bool:
    """Check if a symbol is tradeable on Binance Futures USDT-M."""
    return symbol.upper() in get_binance_futures_symbols()


def is_on_binance_spot(symbol: str) -> bool:
    """Check if a symbol is tradeable on Binance Spot USDT."""
    return symbol.upper() in get_binance_spot_symbols()


def is_on_binance(symbol: str) -> bool:
    """Check if a symbol is tradeable on Binance Spot OR Futures USDT."""
    return is_on_binance_spot(symbol) or is_on_binance_futures(symbol)