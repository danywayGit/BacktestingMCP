"""
Core backtesting engine using the backtesting.py framework.
"""

from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone
import logging
from dataclasses import dataclass, field

from ..data.database import db
from ..data.downloader import downloader
from ..data.timeframe_converter import convert_data_for_backtesting
from config.settings import settings, TimeFrame, Direction, RiskConfig, TradingConfig

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Backtesting result container."""
    strategy_name: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    parameters: Dict[str, Any]
    stats: Dict[str, float]
    trades: List[Dict[str, Any]]
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'strategy_name': self.strategy_name,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'parameters': self.parameters,
            'stats': self.stats,
            'trades': self.trades,
            'equity_curve': self.equity_curve.to_dict() if self.equity_curve is not None else {},
            'drawdown_curve': self.drawdown_curve.to_dict() if self.drawdown_curve is not None else {}
        }


class BaseStrategy(Strategy):
    """Base strategy class with common functionality."""
    
    # Default parameters that can be optimized
    stop_loss_pct = 2.0
    take_profit_pct = 4.0
    risk_pct = 1.0
    
    # Trading filters
    trading_days = list(range(7))  # All days by default
    trading_hours = None  # All hours by default
    direction = Direction.BOTH
    
    def init(self):
        """Initialize strategy indicators and variables."""
        # Common indicators can be added here
        pass
    
    def next(self):
        """Main strategy logic - override in subclasses."""
        pass
    
    def should_trade(self) -> bool:
        """Check if current time allows trading."""
        current_time = self.data.index[-1]
        
        # Check trading days
        if current_time.weekday() not in self.trading_days:
            return False
        
        # Check trading hours
        if self.trading_hours and current_time.hour not in self.trading_hours:
            return False
        
        return True
    
    def calculate_position_size(self, entry_price: float, stop_loss_price: float) -> float:
        """Calculate position size based on risk management."""
        if stop_loss_price is None or entry_price == stop_loss_price:
            return 0.1  # Default 10% of equity
        
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss_price)
        
        # Calculate account risk amount
        account_value = self.equity
        risk_amount = account_value * (self.risk_pct / 100)
        
        # Calculate position value based on risk
        position_value = risk_amount / (risk_per_share / entry_price)
        
        # Convert to fraction of equity
        position_fraction = position_value / account_value
        
        # Cap at 20% of equity to be conservative and avoid margin issues
        position_fraction = min(position_fraction, 0.20)
        
        # Ensure minimum 1% position if we're trading
        if position_fraction > 0 and position_fraction < 0.01:
            position_fraction = 0.01
        
        return position_fraction
    
    def enter_long_position(self, stop_loss: Optional[float] = None, take_profit: Optional[float] = None):
        """Enter a long position with risk management."""
        if self.direction in [Direction.SHORT]:
            return  # Only short trades allowed
        
        if not self.should_trade():
            return
        
        current_price = self.data.Close[-1]
        
        # Calculate stop loss and take profit
        if stop_loss is None:
            stop_loss = current_price * (1 - self.stop_loss_pct / 100)
        
        if take_profit is None:
            take_profit = current_price * (1 + self.take_profit_pct / 100)
        
        # Calculate position size
        size = self.calculate_position_size(current_price, stop_loss)
        
        if size > 0:
            self.buy(size=size, sl=stop_loss, tp=take_profit)
    
    def enter_short_position(self, stop_loss: Optional[float] = None, take_profit: Optional[float] = None):
        """Enter a short position with risk management."""
        if self.direction in [Direction.LONG]:
            return  # Only long trades allowed
        
        if not self.should_trade():
            return
        
        current_price = self.data.Close[-1]
        
        # Calculate stop loss and take profit for short
        if stop_loss is None:
            stop_loss = current_price * (1 + self.stop_loss_pct / 100)
        
        if take_profit is None:
            take_profit = current_price * (1 - self.take_profit_pct / 100)
        
        # Calculate position size
        size = self.calculate_position_size(current_price, stop_loss)
        
        if size > 0:
            self.sell(size=size, sl=stop_loss, tp=take_profit)


class BacktestingEngine:
    """Main backtesting engine."""
    
    def __init__(self, 
                 risk_config: Optional[RiskConfig] = None,
                 trading_config: Optional[TradingConfig] = None):
        """Initialize backtesting engine."""
        self.risk_config = risk_config or settings.risk
        self.trading_config = trading_config or settings.trading
    
    def get_data(self, 
                 symbol: str, 
                 timeframe: TimeFrame,
                 start_date: datetime,
                 end_date: datetime,
                 source_timeframe: Optional[TimeFrame] = None) -> pd.DataFrame:
        """Get market data for backtesting."""
        # Convert symbol format
        db_symbol = symbol.replace('/', '')
        
        # Try to get data from database first
        data = db.get_market_data(db_symbol, timeframe.value, start_date, end_date)
        
        # If data is missing, download it
        if data.empty:
            logger.info(f"No data found in database, downloading {symbol} {timeframe}")
            exchange_symbol = symbol if '/' in symbol else f"{symbol[:3]}/{symbol[3:]}"
            data = downloader.download_data(exchange_symbol, timeframe, start_date, end_date)
        
        # Convert timeframe if needed
        if source_timeframe and source_timeframe != timeframe:
            source_data = db.get_market_data(db_symbol, source_timeframe.value, start_date, end_date)
            if not source_data.empty:
                data = convert_data_for_backtesting(source_data, source_timeframe, timeframe)
        
        # Ensure required columns and naming for backtesting.py
        if not data.empty:
            # Rename columns to match backtesting.py expectations
            data.columns = [col.capitalize() for col in data.columns]
            
            # Ensure we have OHLCV columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_cols):
                raise ValueError(f"Data must contain columns: {required_cols}")
        
        return data
    
    def run_backtest(self,
                    strategy_class: type,
                    symbol: str,
                    timeframe: TimeFrame,
                    start_date: datetime,
                    end_date: datetime,
                    parameters: Optional[Dict[str, Any]] = None,
                    cash: float = 10000,
                    commission: Optional[float] = None) -> BacktestResult:
        """Run a single backtest."""
        
        # Get market data
        data = self.get_data(symbol, timeframe, start_date, end_date)
        
        if data.empty:
            raise ValueError(f"No data available for {symbol} {timeframe}")
        
        # Set up strategy parameters
        strategy_params = parameters or {}
        
        # Add risk and trading configuration to strategy
        strategy_params.update({
            'stop_loss_pct': self.risk_config.default_stop_loss_pct,
            'take_profit_pct': self.risk_config.default_take_profit_pct,
            'risk_pct': self.risk_config.account_risk_pct,
            'trading_days': self.trading_config.trading_days,
            'trading_hours': self.trading_config.trading_hours,
            'direction': self.trading_config.direction
        })
        
        # Override with provided parameters
        strategy_params.update(parameters or {})
        
        # Create backtest instance
        bt = Backtest(
            data=data,
            strategy=strategy_class,
            cash=cash,
            commission=commission or self.trading_config.commission,
            exclusive_orders=True
        )
        
        try:
            # Run backtest
            result = bt.run(**strategy_params)
            
            # Extract stats
            stats = self._extract_stats(result)
            
            # Extract trades
            trades = self._extract_trades(result)
            
            # Create result object
            backtest_result = BacktestResult(
                strategy_name=strategy_class.__name__,
                symbol=symbol,
                timeframe=timeframe.value,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                parameters=strategy_params,
                stats=stats,
                trades=trades,
                equity_curve=result._equity_curve if hasattr(result, '_equity_curve') else None,
                drawdown_curve=result._drawdown_curve if hasattr(result, '_drawdown_curve') else None
            )
            
            # Save to database
            self._save_result(backtest_result)
            
            return backtest_result
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise
    
    def _extract_stats(self, result) -> Dict[str, float]:
        """Extract statistics from backtest result."""
        stats = {}
        
        # Standard metrics
        stats_mapping = {
            'Start': 'start_date',
            'End': 'end_date', 
            'Duration': 'duration_days',
            'Exposure Time [%]': 'exposure_time_pct',
            'Equity Final [$]': 'final_equity',
            'Equity Peak [$]': 'peak_equity',
            'Return [%]': 'total_return_pct',
            'Buy & Hold Return [%]': 'buy_hold_return_pct',
            'Return (Ann.) [%]': 'annualized_return_pct',
            'Volatility (Ann.) [%]': 'volatility_pct',
            'Sharpe Ratio': 'sharpe_ratio',
            'Sortino Ratio': 'sortino_ratio',
            'Calmar Ratio': 'calmar_ratio',
            'Max. Drawdown [%]': 'max_drawdown_pct',
            'Avg. Drawdown [%]': 'avg_drawdown_pct',
            'Max. Drawdown Duration': 'max_drawdown_duration',
            'Avg. Drawdown Duration': 'avg_drawdown_duration',
            '# Trades': 'num_trades',
            'Win Rate [%]': 'win_rate_pct',
            'Best Trade [%]': 'best_trade_pct',
            'Worst Trade [%]': 'worst_trade_pct',
            'Avg. Trade [%]': 'avg_trade_pct',
            'Max. Trade Duration': 'max_trade_duration',
            'Avg. Trade Duration': 'avg_trade_duration',
            'Profit Factor': 'profit_factor',
            'Expectancy [%]': 'expectancy_pct',
            'SQN': 'sqn'
        }
        
        # Extract available stats
        for original_key, new_key in stats_mapping.items():
            attr_name = original_key.replace(' ', '_').replace('[%]', '').replace('[$]', '').replace('.', '').replace('#', 'num')
            if hasattr(result, attr_name):
                value = getattr(result, attr_name)
                # Handle different data types
                if value is None:
                    stats[new_key] = 0.0
                elif isinstance(value, (int, float)):
                    stats[new_key] = float(value)
                else:
                    # For timestamps or other types, convert to string
                    stats[new_key] = str(value)
        
        return stats
    
    def _extract_trades(self, result) -> List[Dict[str, Any]]:
        """Extract individual trades from backtest result."""
        trades = []
        
        if hasattr(result, '_trades') and result._trades is not None:
            trades_df = result._trades
            
            for _, trade in trades_df.iterrows():
                trade_dict = {
                    'entry_time': trade.get('EntryTime', '').isoformat() if hasattr(trade.get('EntryTime', ''), 'isoformat') else str(trade.get('EntryTime', '')),
                    'exit_time': trade.get('ExitTime', '').isoformat() if hasattr(trade.get('ExitTime', ''), 'isoformat') else str(trade.get('ExitTime', '')),
                    'entry_price': float(trade.get('EntryPrice', 0)),
                    'exit_price': float(trade.get('ExitPrice', 0)),
                    'size': float(trade.get('Size', 0)),
                    'return_pct': float(trade.get('ReturnPct', 0)),
                    'duration': str(trade.get('Duration', '')),
                    'direction': 'long' if trade.get('Size', 0) > 0 else 'short'
                }
                trades.append(trade_dict)
        
        return trades
    
    def _save_result(self, result: BacktestResult):
        """Save backtest result to database."""
        try:
            db.save_backtest_result(
                strategy_name=result.strategy_name,
                symbol=result.symbol,
                timeframe=result.timeframe,
                start_date=result.start_date,
                end_date=result.end_date,
                parameters=result.parameters,
                metrics=result.stats,
                trades=result.trades
            )
        except Exception as e:
            logger.error(f"Error saving backtest result: {e}")
    
    def run_multi_symbol_backtest(self,
                                 strategy_class: type,
                                 symbols: List[str],
                                 timeframe: TimeFrame,
                                 start_date: datetime,
                                 end_date: datetime,
                                 parameters: Optional[Dict[str, Any]] = None,
                                 cash_per_symbol: float = 10000) -> Dict[str, BacktestResult]:
        """Run backtests on multiple symbols."""
        results = {}
        
        for symbol in symbols:
            try:
                logger.info(f"Running backtest for {symbol}")
                result = self.run_backtest(
                    strategy_class=strategy_class,
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    parameters=parameters,
                    cash=cash_per_symbol
                )
                results[symbol] = result
                
            except Exception as e:
                logger.error(f"Error backtesting {symbol}: {e}")
                continue
        
        return results
    
    def run_multi_timeframe_backtest(self,
                                   strategy_class: type,
                                   symbol: str,
                                   timeframes: List[TimeFrame],
                                   start_date: datetime,
                                   end_date: datetime,
                                   parameters: Optional[Dict[str, Any]] = None,
                                   cash: float = 10000) -> Dict[TimeFrame, BacktestResult]:
        """Run backtests on multiple timeframes."""
        results = {}
        
        for timeframe in timeframes:
            try:
                logger.info(f"Running backtest for {symbol} on {timeframe}")
                result = self.run_backtest(
                    strategy_class=strategy_class,
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    parameters=parameters,
                    cash=cash
                )
                results[timeframe] = result
                
            except Exception as e:
                logger.error(f"Error backtesting {symbol} on {timeframe}: {e}")
                continue
        
        return results


# Global engine instance
engine = BacktestingEngine()
