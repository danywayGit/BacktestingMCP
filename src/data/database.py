"""
Database module for storing and retrieving cryptocurrency market data.
"""

import json
import logging
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

from config.settings import settings, TimeFrame


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy scalar types to native Python types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


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
                    parameters TEXT,
                    metrics TEXT,
                    trades TEXT,
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

            # Scoring config versions table
            # Every change to weights/filters creates a new row here.
            # The active config is flagged with is_active=1 (only one at a time).
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scoring_configs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT NOT NULL UNIQUE,
                    description TEXT,
                    config_json TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 0,
                    activated_at TIMESTAMP,
                    retired_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_scoring_configs_active
                ON scoring_configs(is_active)
            """)

            # Edge scanner signals table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS edge_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    pair TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    composite_score REAL NOT NULL,
                    components TEXT,
                    config_version TEXT NOT NULL DEFAULT 'v1.0',
                    coin_type TEXT,
                    entry_price REAL NOT NULL,
                    entry_time TIMESTAMP NOT NULL,
                    horizon_hours INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    exit_price REAL,
                    forward_return_pct REAL,
                    outcome TEXT,
                    resolved_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Migrate existing edge_signals rows missing config_version / coin_type
            try:
                conn.execute("ALTER TABLE edge_signals ADD COLUMN config_version TEXT NOT NULL DEFAULT 'v1.0'")
            except Exception:
                pass  # column already exists
            try:
                conn.execute("ALTER TABLE edge_signals ADD COLUMN coin_type TEXT")
            except Exception:
                pass  # column already exists

            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timeframe ON market_data(symbol, timeframe)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_backtest_results_strategy ON backtest_results(strategy_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_backtest_results_symbol ON backtest_results(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edge_signals_status ON edge_signals(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edge_signals_symbol ON edge_signals(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edge_signals_config ON edge_signals(config_version)")
    
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
        
        # Convert timestamp index to unix seconds.
        # pandas DatetimeIndex.astype('int64') returns milliseconds since epoch
        # (pandas 3.x behaviour for timezone-aware indexes).
        if isinstance(data.index[0], pd.Timestamp):
            timestamps = data.index.astype('int64') // 10**3  # milliseconds → seconds
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
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO backtest_results
                (strategy_name, symbol, timeframe, start_date, end_date, parameters, metrics, trades)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                strategy_name, symbol, timeframe, start_date, end_date,
                json.dumps(parameters, cls=_NumpyEncoder),
                json.dumps(metrics, cls=_NumpyEncoder),
                json.dumps(trades, cls=_NumpyEncoder),
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

    def insert_edge_signal(
        self,
        symbol: str,
        pair: str,
        timeframe: str,
        direction: str,
        composite_score: float,
        components: Dict[str, Any],
        entry_price: float,
        horizon_hours: int,
        config_version: str = "v1.0",
        coin_type: str = "OTHER",
        target_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> int:
        """Log a composite-scanner signal so its forward outcome can be tracked."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO edge_signals
                (symbol, pair, timeframe, direction, composite_score, components,
                 config_version, coin_type, entry_price, entry_time, horizon_hours,
                 target_price, stop_price)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol, pair, timeframe, direction, composite_score,
                json.dumps(components), config_version, coin_type,
                entry_price, datetime.now(timezone.utc).isoformat(),
                horizon_hours, target_price, stop_price,
            ))
            signal_id = cursor.lastrowid
        logger.info("Logged %s %s signal #%d (score=%+.2f, horizon=%dh)", symbol, direction, signal_id, composite_score, horizon_hours)
        return signal_id

    def get_pending_edge_signals(self) -> List[Dict[str, Any]]:
        """Signals whose tracking horizon has elapsed and are ready to be resolved."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, symbol, pair, timeframe, direction, composite_score,
                       entry_price, entry_time, horizon_hours
                FROM edge_signals
                WHERE status = 'PENDING'
            """)
            columns = [d[0] for d in cursor.description]
            rows = cursor.fetchall()
        return [dict(zip(columns, row)) for row in rows]

    def get_pending_edge_signal(self, symbol: str, direction: str, config_version: str) -> Optional[Dict[str, Any]]:
        """Return a single PENDING signal for this symbol+direction+config, or None."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, symbol, pair, timeframe, direction, composite_score,
                       entry_price, entry_time, horizon_hours, components, config_version
                FROM edge_signals
                WHERE status = 'PENDING' AND symbol = ? AND direction = ? AND config_version = ?
                LIMIT 1
            """, (symbol, direction, config_version))
            row = cursor.fetchone()
            if row is None:
                return None
            columns = [d[0] for d in cursor.description]
        return dict(zip(columns, row))

    def get_edge_signal(self, signal_id: int) -> Optional[Dict[str, Any]]:
        """Fetch a single edge signal by its ID."""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM edge_signals WHERE id = ?", (signal_id,))
            row = cursor.fetchone()
            if row:
                columns = [d[0] for d in cursor.description]
                return dict(zip(columns, row))
        return None

    def update_edge_signal(
        self,
        signal_id: int,
        composite_score: float,
        entry_price: float,
        components: Dict[str, Any],
    ) -> None:
        """Update an existing PENDING signal with latest score/price (keeps entry_time + config_version)."""
        with self.get_connection() as conn:
            conn.execute("""
                UPDATE edge_signals
                SET composite_score = ?, entry_price = ?, components = ?
                WHERE id = ? AND status = 'PENDING'
            """, (composite_score, entry_price, json.dumps(components), signal_id))

    def resolve_edge_signal(
        self,
        signal_id: int,
        exit_price: float,
        forward_return_pct: float,
        outcome: str,
        time_to_resolve_hours: float = 0.0,
    ) -> None:
        """Record the actual forward outcome of a previously logged signal."""
        if time_to_resolve_hours <= 0:
            # Fallback: use the signal's own horizon_hours as the estimated time
            signal = self.get_edge_signal(signal_id)
            if signal and signal.get("entry_time"):
                et = datetime.fromisoformat(signal["entry_time"])
                if et.tzinfo is None:
                    et = et.replace(tzinfo=timezone.utc)
                time_to_resolve_hours = (datetime.now(timezone.utc) - et).total_seconds() / 3600
        with self.get_connection() as conn:
            conn.execute("""
                UPDATE edge_signals
                SET status = 'RESOLVED', exit_price = ?, forward_return_pct = ?,
                    outcome = ?, resolved_at = ?, time_to_resolve_hours = ?
                WHERE id = ?
            """, (exit_price, forward_return_pct, outcome, datetime.now(timezone.utc).isoformat(),
                  round(time_to_resolve_hours, 4), signal_id))

    def get_resolved_edge_signals(self, since: Optional[datetime] = None, resolved_since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Resolved signals, for win-rate / forward-performance reporting.

        Args:
            since: If set, only signals whose entry_time >= since are returned.
            resolved_since: If set, only signals whose resolved_at >= resolved_since are returned.
        """
        query = "SELECT * FROM edge_signals WHERE status = 'RESOLVED'"
        params: List[Any] = []
        if since:
            query += " AND entry_time >= ?"
            params.append(since.isoformat())
        if resolved_since:
            query += " AND resolved_at >= ?"
            params.append(resolved_since.isoformat())
        query += " ORDER BY entry_time"

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            columns = [d[0] for d in cursor.description]
            rows = cursor.fetchall()
        return [dict(zip(columns, row)) for row in rows]

    # ------------------------------------------------------------------
    # Scoring config versioning
    # ------------------------------------------------------------------

    def save_scoring_config(self, config: "ScoringConfig") -> None:  # type: ignore[name-defined]
        """Persist a ScoringConfig to the DB (insert or update description/json)."""
        import json
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO scoring_configs (version, description, config_json, is_active)
                VALUES (?, ?, ?, 0)
                ON CONFLICT(version) DO UPDATE SET
                    description = excluded.description,
                    config_json = excluded.config_json
            """, (config.version, config.description, json.dumps(config.to_dict())))

    def activate_scoring_config(self, version: str) -> None:
        """Set a config as active, retiring the previously active one."""
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            # Retire current active
            conn.execute("""
                UPDATE scoring_configs SET is_active = 0, retired_at = ?
                WHERE is_active = 1
            """, (now,))
            # Activate new one
            conn.execute("""
                UPDATE scoring_configs SET is_active = 1, activated_at = ?, retired_at = NULL
                WHERE version = ?
            """, (now, version))

    def get_active_scoring_config_version(self) -> Optional[str]:
        """Return the version string of the currently active config, or None."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT version FROM scoring_configs WHERE is_active = 1 LIMIT 1"
            ).fetchone()
        return row[0] if row else None

    def list_scoring_configs(self) -> List[Dict[str, Any]]:
        """Return all scoring config versions, newest first."""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT version, description, is_active, activated_at, retired_at, created_at
                FROM scoring_configs ORDER BY created_at DESC
            """)
            columns = [d[0] for d in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]


# Global database instance
db = CryptoDatabase()
