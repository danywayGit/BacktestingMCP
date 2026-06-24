#!/usr/bin/env python3
"""
Pre-warm OHLCV data for all pairs currently tracked in edge_signals.
Skips pairs not available on Binance Futures (binanceusdm).
Logs progress to /tmp/prewarm.log
"""
import sys, os, sqlite3, logging

# Load .env
with open('/home/hermes/BacktestingMCP/.env') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            os.environ[k.strip()] = v.strip()

sys.path.insert(0, '/home/hermes/BacktestingMCP')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('/tmp/prewarm.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger('prewarm')

from src.core.backtesting_engine import engine
from config.settings import TimeFrame
from datetime import datetime, timezone, timedelta

conn = sqlite3.connect('/home/hermes/BacktestingMCP/data/crypto.db')
pairs = conn.execute('''
    SELECT DISTINCT pair, timeframe FROM edge_signals
    WHERE status="PENDING" ORDER BY pair
''').fetchall()
conn.close()

log.info(f"Pre-warming {len(pairs)} pairs...")

end = datetime.now(timezone.utc)
start = end - timedelta(days=5)  # 5 days = enough for resolve + TA scanner

ok, skipped, failed = 0, 0, 0
for i, (pair, tf) in enumerate(pairs, 1):
    try:
        data = engine.get_data(pair, TimeFrame(tf), start, end)
        if data.empty:
            log.warning(f"[{i}/{len(pairs)}] {pair} — no data (not on Binance Futures)")
            skipped += 1
        else:
            log.info(f"[{i}/{len(pairs)}] {pair} — {len(data)} rows, close={data['Close'].iloc[-1]:.4f}")
            ok += 1
    except Exception as e:
        log.error(f"[{i}/{len(pairs)}] {pair} — ERROR: {e}")
        failed += 1

log.info(f"\n=== DONE === ok={ok}  skipped={skipped}  failed={failed}")
