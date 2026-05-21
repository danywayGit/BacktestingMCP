"""
Download market data for signal strategy deduction.

Downloads OHLCV data from Binance for all symbols and base timeframes needed
to analyze Discord (Meta Signals) and Telegram (DaviddTech) signals.

Usage:
    python download_signal_data.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timezone
from config.settings import TimeFrame
from src.data.downloader import DataDownloader
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---- Symbols and timeframes needed ----

# Discord top symbols (mapped to CCXT format with /USDT)
# BTC and ETH appear without USDT suffix in Discord data
DISCORD_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "AAVE/USDT", "BCH/USDT", "TRX/USDT",
    "ZEC/USDT", "XRP/USDT", "ATOM/USDT", "UNI/USDT", "LINK/USDT",
    "DASH/USDT", "DOT/USDT", "SAND/USDT", "XTZ/USDT", "XLM/USDT",
    "QTUM/USDT", "ETC/USDT", "LTC/USDT", "NEO/USDT", "DOGE/USDT",
    "BAND/USDT", "SUI/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT",
    "DYDX/USDT", "APT/USDT", "INJ/USDT", "ALGO/USDT", "EOS/USDT",
]

# Telegram strategy symbols
TELEGRAM_SYMBOLS = [
    "SOL/USDT",   # Precision Trend Mastery 30m
    "NEAR/USDT",  # NEAR Trendhoo 30mins
    "ADA/USDT",   # ADA SuperF 1h
    "LINK/USDT",  # McGinley Trend 45m
    "DOGE/USDT",  # DOGE Liquidity Spec 45m
    "BTC/USDT",   # BTC Trendhoo 2h + Stiff Zone 15m
    "AGLD/USDT",  # AGLD Trigger Happy 2 45min
]

# Combined unique symbols
ALL_SYMBOLS = sorted(set(DISCORD_SYMBOLS + TELEGRAM_SYMBOLS))

# Date ranges
DISCORD_START = datetime(2024, 6, 1, tzinfo=timezone.utc)
TELEGRAM_START = datetime(2025, 4, 1, tzinfo=timezone.utc)
END_DATE = datetime(2026, 4, 1, tzinfo=timezone.utc)

# Base timeframes to download:
# - 15m: aggregate to 45M (3x15m) for Discord and Telegram 45M strategies
# - 30m: direct use for Telegram 30M strategies
# - 1h: direct use + aggregate to 2H (2x), 3H (3x), 8H (8x)
# - 4h: aggregate to 24H if needed
# - 15m: direct use for Telegram 15M strategy

DOWNLOAD_PLAN = [
    # (timeframe, symbols, start_date, description)
    (TimeFrame.M15, ALL_SYMBOLS, DISCORD_START, "15m data for 45M aggregation + 15M strategies"),
    (TimeFrame.M30, ["SOL/USDT", "NEAR/USDT"], TELEGRAM_START, "30m data for Telegram 30M strategies"),
    (TimeFrame.H1, ALL_SYMBOLS, DISCORD_START, "1h data for 1H/2H/3H/8H"),
]


def main():
    downloader = DataDownloader(exchange_name="binance")

    total_downloads = sum(len(symbols) for _, symbols, _, _ in DOWNLOAD_PLAN)
    completed = 0
    failed = []

    for timeframe, symbols, start_date, description in DOWNLOAD_PLAN:
        print(f"\n{'='*60}")
        print(f"Downloading {timeframe.value} - {description}")
        print(f"{'='*60}")

        for symbol in symbols:
            completed += 1
            print(f"\n[{completed}/{total_downloads}] {symbol} {timeframe.value} "
                  f"({start_date.strftime('%Y-%m-%d')} → {END_DATE.strftime('%Y-%m-%d')})")

            try:
                df = downloader.download_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=END_DATE,
                    force_update=False
                )
                print(f"  ✓ {len(df)} candles")
            except Exception as e:
                error_msg = f"{symbol} {timeframe.value}: {e}"
                logger.error(error_msg)
                failed.append(error_msg)
                print(f"  ✗ FAILED: {e}")

    # Summary
    print(f"\n{'='*60}")
    print(f"DOWNLOAD COMPLETE")
    print(f"  Successful: {completed - len(failed)}/{total_downloads}")
    if failed:
        print(f"  Failed ({len(failed)}):")
        for f in failed:
            print(f"    - {f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
