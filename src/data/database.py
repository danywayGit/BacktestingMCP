"""
Database module for storing and retrieving cryptocurrency market data.
"""

import sqlite3
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from contextlib import contextmanager

from config.settings import settings, TimeFrame


class CryptoDatabase:
    """SQLite database for cryptocurrency market data."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize database connection."""
        self.db_path = db_path or settings.data.database_path
        self.db_path = Path(self.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables."""
        with self.get_connection() as conn:
            # Market data table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timeframe, timestamp)
                )
            """)
            
            # Backtest results table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    parameters TEXT,  -- JSON string
                    metrics TEXT,     -- JSON string
                    trades TEXT,      -- JSON string
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Strategy parameters table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategy_parameters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    parameter_name TEXT NOT NULL,
                    parameter_value TEXT NOT NULL,
                    parameter_type TEXT NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(strategy_name, parameter_name)
                )
            """)
            
            # Performance metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    backtest_id INTEGER NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (backtest_id) REFERENCES backtest_results (id)
                )
            """)
            
            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timeframe ON market_data(symbol, timeframe)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_backtest_results_strategy ON backtest_results(strategy_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_backtest_results_symbol ON backtest_results(symbol)")
    
    @contextmanager
    def get_connection(self):
        """Get database connection with automatic closing."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def insert_market_data(self, symbol: str, timeframe: str, data: pd.DataFrame) -> int:
        """Insert market data into database."""
        if data.empty:
            return 0
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")
        
        # Convert timestamp to unix timestamp if it's datetime
        if isinstance(data.index[0], pd.Timestamp):
            timestamps = data.index.astype('int64') // 10**9  # Convert to seconds
        else:
            timestamps = data.index
        
        # Prepare data for insertion
        records = []
        for i, (timestamp, row) in enumerate(zip(timestamps, data.itertuples())):
            records.append((
                symbol,
                timeframe,
                int(timestamp),
                float(row.open),
                float(row.high),
                float(row.low),
                float(row.close),
                float(row.volume)
            ))
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT OR REPLACE INTO market_data 
                (symbol, timeframe, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
            
            return cursor.rowcount
    
    def get_market_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Retrieve market data from database."""
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM market_data
            WHERE symbol = ? AND timeframe = ?
        """
        params = [symbol, timeframe]
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(int(start_date.timestamp()))
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(int(end_date.timestamp()))
        
        query += " ORDER BY timestamp"
        
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        
        if df.empty:
            return df
        
        # Convert timestamp back to datetime index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def get_available_data_range(self, symbol: str, timeframe: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Get the available data range for a symbol and timeframe."""
        query = """
            SELECT MIN(timestamp) as start_ts, MAX(timestamp) as end_ts
            FROM market_data
            WHERE symbol = ? AND timeframe = ?
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (symbol, timeframe))
            result = cursor.fetchone()
        
        if result and result[0] and result[1]:
            start_date = datetime.fromtimestamp(result[0], tz=timezone.utc)
            end_date = datetime.fromtimestamp(result[1], tz=timezone.utc)
            return start_date, end_date
        
        return None, None
    
    def get_missing_data_ranges(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Tuple[datetime, datetime]]:
        """Find missing data ranges that need to be downloaded."""
        # Get existing data timestamps
        query = """
            SELECT timestamp FROM market_data
            WHERE symbol = ? AND timeframe = ? 
            AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (
                symbol, timeframe,
                int(start_date.timestamp()),
                int(end_date.timestamp())
            ))
            existing_timestamps = [row[0] for row in cursor.fetchall()]
        
        if not existing_timestamps:
            return [(start_date, end_date)]
        
        # Find gaps in the data
        missing_ranges = []
        
        # Check if data starts after requested start
        first_timestamp = datetime.fromtimestamp(existing_timestamps[0], tz=timezone.utc)
        if start_date < first_timestamp:
            missing_ranges.append((start_date, first_timestamp))
        
        # Check for gaps in the middle
        for i in range(len(existing_timestamps) - 1):
            current_ts = existing_timestamps[i]
            next_ts = existing_timestamps[i + 1]
            
            # Calculate expected next timestamp based on timeframe
            expected_gap = self._get_timeframe_seconds(timeframe)
            if next_ts - current_ts > expected_gap * 1.5:  # Allow some tolerance
                gap_start = datetime.fromtimestamp(current_ts + expected_gap, tz=timezone.utc)
                gap_end = datetime.fromtimestamp(next_ts, tz=timezone.utc)
                missing_ranges.append((gap_start, gap_end))
        
        # Check if data ends before requested end
        last_timestamp = datetime.fromtimestamp(existing_timestamps[-1], tz=timezone.utc)
        if end_date > last_timestamp:
            missing_ranges.append((last_timestamp, end_date))
        
        return missing_ranges
    
    def _get_timeframe_seconds(self, timeframe: str) -> int:
        """Get timeframe duration in seconds."""
        timeframe_seconds = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "4h": 14400,
            "12h": 43200,
            "1d": 86400,
            "1w": 604800,
        }
        return timeframe_seconds.get(timeframe, 3600)
    
    def save_backtest_result(
        self,
        strategy_name: str,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        parameters: Dict[str, Any],
        metrics: Dict[str, float],
        trades: List[Dict[str, Any]]
    ) -> int:
        """Save backtest results to database."""
        import json
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO backtest_results
                (strategy_name, symbol, timeframe, start_date, end_date, parameters, metrics, trades)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                strategy_name, symbol, timeframe, start_date, end_date,
                json.dumps(parameters), json.dumps(metrics), json.dumps(trades)
            ))
            
            backtest_id = cursor.lastrowid
            
            # Save individual metrics for easier querying
            for metric_name, metric_value in metrics.items():
                cursor.execute("""
                    INSERT INTO performance_metrics (backtest_id, metric_name, metric_value)
                    VALUES (?, ?, ?)
                """, (backtest_id, metric_name, metric_value))
            
            return backtest_id
    
    def get_backtest_results(
        self,
        strategy_name: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieve backtest results from database."""
        import json
        
        query = "SELECT * FROM backtest_results WHERE 1=1"
        params = []
        
        if strategy_name:
            query += " AND strategy_name = ?"
            params.append(strategy_name)
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
        
        results = []
        for row in rows:
            result = {
                'id': row[0],
                'strategy_name': row[1],
                'symbol': row[2],
                'timeframe': row[3],
                'start_date': row[4],
                'end_date': row[5],
                'parameters': json.loads(row[6]) if row[6] else {},
                'metrics': json.loads(row[7]) if row[7] else {},
                'trades': json.loads(row[8]) if row[8] else [],
                'created_at': row[9]
            }
            results.append(result)
        
        return results
    
    def get_symbols_and_timeframes(self) -> List[Tuple[str, str]]:
        """Get all available symbol and timeframe combinations."""
        query = """
            SELECT DISTINCT symbol, timeframe FROM market_data
            ORDER BY symbol, timeframe
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            return cursor.fetchall()
    
    def delete_market_data(self, symbol: str, timeframe: str) -> int:
        """Delete market data for a symbol and timeframe."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM market_data WHERE symbol = ? AND timeframe = ?
            """, (symbol, timeframe))
            return cursor.rowcount


# Global database instance
db = CryptoDatabase()
