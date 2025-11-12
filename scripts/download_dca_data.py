"""
Download historical data for DCA strategy backtesting

This script downloads daily data for BTC, ETH, BNB, and TRX
to prepare for DCA strategy comparison.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.downloader import downloader
from src.data.database import db
from config.settings import TimeFrame


def download_dca_data():
    """Download data for all DCA strategy symbols."""
    
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'TRX/USDT']
    timeframe = TimeFrame.D1  # Use enum instead of string
    start_date = datetime(2017, 1, 1, tzinfo=timezone.utc)
    end_date = datetime(2025, 10, 31, tzinfo=timezone.utc)
    
    print("📥 Downloading DCA Strategy Data")
    print("=" * 70)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Timeframe: {timeframe.value}")
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("=" * 70)
    
    for symbol in symbols:
        print(f"\n📊 Downloading {symbol}...")
        
        try:
            # Download data (downloader automatically saves to database)
            data = downloader.download_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            if data is not None and len(data) > 0:
                symbol_db = symbol.replace('/', '')  # Convert BTC/USDT to BTCUSDT
                print(f"✅ Downloaded and saved {len(data)} candles for {symbol}")
            else:
                print(f"⚠️  No data retrieved for {symbol}")
                
        except Exception as e:
            print(f"❌ Error downloading {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("✅ Data download complete!")
    print("=" * 70)
    
    # Verify data
    print("\n📋 Verifying downloaded data...")
    for symbol in symbols:
        symbol_db = symbol.replace('/', '')
        try:
            data = db.get_market_data(
                symbol=symbol_db,
                timeframe=timeframe.value,
                start_date=start_date,
                end_date=end_date
            )
            
            if data is not None and len(data) > 0:
                print(f"  {symbol_db}: {len(data)} candles ✓")
            else:
                print(f"  {symbol_db}: No data ✗")
        except Exception as e:
            print(f"  {symbol_db}: Error - {e} ✗")


if __name__ == "__main__":
    download_dca_data()
