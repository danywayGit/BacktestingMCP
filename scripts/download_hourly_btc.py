"""
Download multi-timeframe BTC data from 2017 to 2025 for DCA optimization
Downloads 1h, 4h, and 12h timeframes
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.downloader import downloader
from src.data.database import db
from config.settings import TimeFrame


def download_btc_timeframes():
    """Download BTC data in multiple timeframes for comprehensive backtesting."""
    
    symbol = 'BTC/USDT'
    timeframes = [TimeFrame.H1, TimeFrame.H4, TimeFrame.H12]  # 1h, 4h, 12h
    start_date = datetime(2017, 1, 1, tzinfo=timezone.utc)
    end_date = datetime(2025, 10, 31, tzinfo=timezone.utc)
    
    print("=" * 70)
    print("📥 Downloading Multi-Timeframe BTC Data for DCA Optimization")
    print("=" * 70)
    print(f"Symbol: {symbol}")
    print(f"Timeframes: {', '.join([tf.value for tf in timeframes])}")
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Expected 1h candles: ~{(end_date - start_date).days * 24:,}")
    print("=" * 70)
    print()
    
    results = {}
    
    for timeframe in timeframes:
        try:
            print(f"\n📊 Downloading {symbol} {timeframe.value} data...")
            print("   This may take several minutes...")
            
            # Download data (downloader automatically saves to database)
            data = downloader.download_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            if data is not None and len(data) > 0:
                symbol_db = symbol.replace('/', '')  # Convert BTC/USDT to BTCUSDT
                print(f"✅ Downloaded and saved {len(data):,} candles for {symbol} {timeframe.value}")
                print(f"   Date range: {data.index[0]} to {data.index[-1]}")
                print(f"   Price range: ${data['close'].min():,.2f} - ${data['close'].max():,.2f}")
                results[timeframe.value] = True
            else:
                print(f"⚠️  No data retrieved for {symbol} {timeframe.value}")
                results[timeframe.value] = False
                
        except Exception as e:
            print(f"❌ Error downloading {symbol} {timeframe.value}: {e}")
            import traceback
            traceback.print_exc()
            results[timeframe.value] = False
    
    print()
    print("=" * 70)
    print("✅ Multi-timeframe data download complete!")
    print("=" * 70)
    print()
    
    # Verify data in database
    print("📋 Verifying database storage...")
    symbol_db = 'BTCUSDT'
    all_success = True
    
    for timeframe in timeframes:
        try:
            data = db.get_market_data(
                symbol=symbol_db,
                timeframe=timeframe.value,
                start_date=start_date,
                end_date=end_date
            )
            
            if data is not None and len(data) > 0:
                print(f"✓ {timeframe.value}: {len(data):,} candles ({data.index[0]} to {data.index[-1]})")
            else:
                print(f"✗ {timeframe.value}: No data found")
                all_success = False
        except Exception as e:
            print(f"✗ {timeframe.value}: Error - {e}")
            all_success = False
    
    return all_success


if __name__ == "__main__":
    success = download_btc_timeframes()
    if success:
        print("\n🚀 Ready for optimization with 2017-2025 data!")
    else:
        print("\n⚠️  Some downloads incomplete")
        sys.exit(1)
