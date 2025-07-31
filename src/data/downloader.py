"""
Data downloader for cryptocurrency market data using CCXT.
Supports incremental downloads and multiple exchanges.
"""

import ccxt
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any
import time
import logging
from pathlib import Path

from ..data.database import db
from config.settings import settings, TimeFrame, CCXT_TIMEFRAME_MAPPING

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataDownloader:
    """Download cryptocurrency market data from exchanges."""
    
    def __init__(self, exchange_name: str = "binance"):
        """Initialize data downloader."""
        self.exchange_name = exchange_name
        self.exchange = self._init_exchange(exchange_name)
        
    def _init_exchange(self, exchange_name: str) -> ccxt.Exchange:
        """Initialize exchange connection."""
        exchange_class = getattr(ccxt, exchange_name)
        
        # Exchange configuration
        config = {
            'apiKey': None,  # No API key needed for public data
            'secret': None,
            'timeout': 30000,
            'enableRateLimit': True,
            'sandbox': False,
        }
        
        # Add exchange-specific settings
        if exchange_name == 'binance':
            config.update({
                'options': {
                    'defaultType': 'spot',  # spot, future, margin
                }
            })
        
        return exchange_class(config)
    
    def download_data(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        force_update: bool = False
    ) -> pd.DataFrame:
        """
        Download market data for a symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Data timeframe
            start_date: Start date for data
            end_date: End date for data (default: now)
            force_update: Force re-download of existing data
            
        Returns:
            DataFrame with OHLCV data
        """
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        
        # Convert symbol format (BTC/USDT -> BTCUSDT for database)
        db_symbol = symbol.replace('/', '')
        ccxt_timeframe = CCXT_TIMEFRAME_MAPPING[timeframe]
        
        logger.info(f"Downloading {symbol} {timeframe} data from {start_date} to {end_date}")
        
        # Check for existing data if not forcing update
        all_data = pd.DataFrame()
        
        if not force_update:
            # Get existing data from database
            existing_data = db.get_market_data(db_symbol, timeframe.value, start_date, end_date)
            
            if not existing_data.empty:
                logger.info(f"Found {len(existing_data)} existing records")
                all_data = existing_data
                
                # Find missing data ranges
                missing_ranges = db.get_missing_data_ranges(db_symbol, timeframe.value, start_date, end_date)
                
                if not missing_ranges:
                    logger.info("No missing data found")
                    return existing_data
                
                logger.info(f"Found {len(missing_ranges)} missing data ranges")
            else:
                missing_ranges = [(start_date, end_date)]
        else:
            missing_ranges = [(start_date, end_date)]
        
        # Download missing data
        for range_start, range_end in missing_ranges:
            logger.info(f"Downloading data from {range_start} to {range_end}")
            
            new_data = self._download_range(symbol, ccxt_timeframe, range_start, range_end)
            
            if not new_data.empty:
                # Save to database
                saved_count = db.insert_market_data(db_symbol, timeframe.value, new_data)
                logger.info(f"Saved {saved_count} new records to database")
                
                # Combine with existing data
                if all_data.empty:
                    all_data = new_data
                else:
                    all_data = pd.concat([all_data, new_data]).sort_index().drop_duplicates()
        
        # Return final combined data
        return db.get_market_data(db_symbol, timeframe.value, start_date, end_date)
    
    def _download_range(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Download data for a specific date range."""
        all_candles = []
        
        # Convert to milliseconds
        since = int(start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)
        
        # Calculate limit based on timeframe (max candles per request)
        limit = 1000  # Most exchanges support up to 1000-1500 candles per request
        
        while since < end_ms:
            try:
                # Fetch OHLCV data
                candles = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=limit
                )
                
                if not candles:
                    break
                
                all_candles.extend(candles)
                
                # Update since to the last candle timestamp + 1 period
                last_timestamp = candles[-1][0]
                since = last_timestamp + 1
                
                # Rate limiting
                time.sleep(self.exchange.rateLimit / 1000)
                
                logger.debug(f"Downloaded {len(candles)} candles, last timestamp: {datetime.fromtimestamp(last_timestamp/1000)}")
                
            except Exception as e:
                logger.error(f"Error downloading data: {e}")
                time.sleep(5)  # Wait before retry
                continue
        
        if not all_candles:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Convert timestamp to datetime index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        
        # Remove duplicates and sort
        df = df[~df.index.duplicated(keep='last')].sort_index()
        
        # Filter to requested date range
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        return df
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available trading symbols."""
        try:
            markets = self.exchange.load_markets()
            symbols = [symbol for symbol in markets.keys() if '/USDT' in symbol]
            return sorted(symbols)
        except Exception as e:
            logger.error(f"Error fetching symbols: {e}")
            return []
    
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Get information about a trading symbol."""
        try:
            markets = self.exchange.load_markets()
            if symbol in markets:
                return markets[symbol]
            return {}
        except Exception as e:
            logger.error(f"Error fetching symbol info: {e}")
            return {}
    
    def download_multiple_symbols(
        self,
        symbols: List[str],
        timeframe: TimeFrame,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        force_update: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """Download data for multiple symbols."""
        results = {}
        
        for symbol in symbols:
            try:
                logger.info(f"Downloading data for {symbol}")
                data = self.download_data(symbol, timeframe, start_date, end_date, force_update)
                results[symbol] = data
                
                # Brief pause between symbols
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error downloading {symbol}: {e}")
                continue
        
        return results
    
    def update_all_data(self, symbols: Optional[List[str]] = None) -> Dict[str, int]:
        """Update all existing data with latest candles."""
        if symbols is None:
            # Get symbols from database
            symbol_timeframe_pairs = db.get_symbols_and_timeframes()
            symbols_to_update = list(set([pair[0] for pair in symbol_timeframe_pairs]))
        else:
            symbols_to_update = [s.replace('/', '') for s in symbols]
        
        results = {}
        current_time = datetime.now(timezone.utc)
        
        for db_symbol in symbols_to_update:
            try:
                # Convert back to exchange format
                exchange_symbol = f"{db_symbol[:3]}/{db_symbol[3:]}" if len(db_symbol) > 4 else db_symbol
                
                # Get available timeframes for this symbol
                symbol_timeframes = [pair[1] for pair in db.get_symbols_and_timeframes() if pair[0] == db_symbol]
                
                symbol_results = {}
                
                for tf_str in symbol_timeframes:
                    try:
                        timeframe = TimeFrame(tf_str)
                        
                        # Get last available data point
                        start_range, end_range = db.get_available_data_range(db_symbol, tf_str)
                        
                        if end_range:
                            # Download from last data point to now
                            update_start = end_range
                            new_data = self.download_data(
                                exchange_symbol, timeframe, update_start, current_time, force_update=False
                            )
                            symbol_results[tf_str] = len(new_data)
                        
                    except Exception as e:
                        logger.error(f"Error updating {db_symbol} {tf_str}: {e}")
                        continue
                
                results[db_symbol] = symbol_results
                
            except Exception as e:
                logger.error(f"Error updating {db_symbol}: {e}")
                continue
        
        return results


# Convenience functions
def download_crypto_data(
    symbol: str,
    timeframe: TimeFrame = TimeFrame.H1,
    start_date: str = "2022-01-01",
    end_date: Optional[str] = None,
    exchange: str = "binance",
    force_update: bool = False
) -> pd.DataFrame:
    """
    Convenience function to download cryptocurrency data.
    
    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
        timeframe: Data timeframe
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        exchange: Exchange name
        force_update: Force re-download
        
    Returns:
        DataFrame with OHLCV data
    """
    downloader = DataDownloader(exchange)
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc) if end_date else None
    
    return downloader.download_data(symbol, timeframe, start_dt, end_dt, force_update)


def update_all_crypto_data(exchange: str = "binance") -> Dict[str, int]:
    """Update all existing cryptocurrency data."""
    downloader = DataDownloader(exchange)
    return downloader.update_all_data()


# Global downloader instance
downloader = DataDownloader(settings.data.exchange)
