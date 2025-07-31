"""
Risk management module for position sizing and risk controls.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from config.settings import RiskConfig, settings


class PositionSizeMethod(str, Enum):
    """Position sizing methods."""
    FIXED_PERCENT = "fixed_percent"
    FIXED_AMOUNT = "fixed_amount"
    ATR_BASED = "atr_based"
    VOLATILITY_BASED = "volatility_based"
    KELLY_CRITERION = "kelly_criterion"


@dataclass
class RiskMetrics:
    """Risk metrics for a trading period."""
    max_drawdown_pct: float = 0.0
    current_drawdown_pct: float = 0.0
    daily_var_95: float = 0.0  # 95% Value at Risk
    daily_var_99: float = 0.0  # 99% Value at Risk
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    volatility_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0


@dataclass
class PositionInfo:
    """Information about a trading position."""
    symbol: str
    size: float
    entry_price: float
    current_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    direction: str = "long"  # "long" or "short"
    entry_time: Optional[datetime] = None
    
    @property
    def market_value(self) -> float:
        """Current market value of the position."""
        return abs(self.size) * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L of the position."""
        if self.direction == "long":
            return self.size * (self.current_price - self.entry_price)
        else:
            return self.size * (self.entry_price - self.current_price)
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L as percentage."""
        if self.entry_price == 0:
            return 0.0
        
        if self.direction == "long":
            return (self.current_price - self.entry_price) / self.entry_price * 100
        else:
            return (self.entry_price - self.current_price) / self.entry_price * 100


class RiskManager:
    """Risk management system for trading."""
    
    def __init__(self, risk_config: Optional[RiskConfig] = None):
        """Initialize risk manager."""
        self.config = risk_config or settings.risk
        self.positions: Dict[str, PositionInfo] = {}
        self.trade_history: List[Dict] = []
        self.daily_pnl: List[Tuple[datetime, float]] = []
        self.account_value_history: List[Tuple[datetime, float]] = []
        
        # Risk state tracking
        self.daily_loss_pct = 0.0
        self.weekly_loss_pct = 0.0
        self.monthly_loss_pct = 0.0
        self.last_reset_day = datetime.now().date()
        self.last_reset_week = datetime.now().isocalendar()[:2]
        self.last_reset_month = (datetime.now().year, datetime.now().month)
    
    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_price: float,
        account_value: float,
        method: PositionSizeMethod = PositionSizeMethod.FIXED_PERCENT,
        **kwargs
    ) -> float:
        """
        Calculate position size based on risk management rules.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss_price: Stop loss price
            account_value: Current account value
            method: Position sizing method
            **kwargs: Additional parameters for specific methods
            
        Returns:
            Position size (number of shares/units)
        """
        if entry_price <= 0 or account_value <= 0:
            return 0.0
        
        if method == PositionSizeMethod.FIXED_PERCENT:
            return self._fixed_percent_sizing(entry_price, stop_loss_price, account_value)
        
        elif method == PositionSizeMethod.FIXED_AMOUNT:
            fixed_amount = kwargs.get('fixed_amount', 1000)
            return fixed_amount / entry_price
        
        elif method == PositionSizeMethod.ATR_BASED:
            atr = kwargs.get('atr', 0.02 * entry_price)  # Default 2% ATR
            return self._atr_based_sizing(entry_price, atr, account_value)
        
        elif method == PositionSizeMethod.VOLATILITY_BASED:
            volatility = kwargs.get('volatility', 0.02)  # Default 2% daily volatility
            return self._volatility_based_sizing(entry_price, volatility, account_value)
        
        elif method == PositionSizeMethod.KELLY_CRITERION:
            win_rate = kwargs.get('win_rate', 0.5)
            avg_win = kwargs.get('avg_win', 0.02)
            avg_loss = kwargs.get('avg_loss', 0.01)
            return self._kelly_criterion_sizing(entry_price, account_value, win_rate, avg_win, avg_loss)
        
        else:
            return self._fixed_percent_sizing(entry_price, stop_loss_price, account_value)
    
    def _fixed_percent_sizing(self, entry_price: float, stop_loss_price: float, account_value: float) -> float:
        """Fixed percentage risk position sizing."""
        if stop_loss_price is None or entry_price == stop_loss_price:
            return 0.0
        
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss_price)
        
        # Calculate account risk amount
        risk_amount = account_value * (self.config.account_risk_pct / 100)
        
        # Calculate position size
        position_size = risk_amount / risk_per_share
        
        # Apply maximum position limits
        max_position_value = account_value * 0.95  # Leave 5% cash
        max_shares = max_position_value / entry_price
        
        return min(position_size, max_shares)
    
    def _atr_based_sizing(self, entry_price: float, atr: float, account_value: float) -> float:
        """ATR-based position sizing."""
        if atr <= 0:
            return 0.0
        
        # Use ATR as the risk measure
        risk_amount = account_value * (self.config.account_risk_pct / 100)
        position_size = risk_amount / atr
        
        # Apply limits
        max_position_value = account_value * 0.95
        max_shares = max_position_value / entry_price
        
        return min(position_size, max_shares)
    
    def _volatility_based_sizing(self, entry_price: float, volatility: float, account_value: float) -> float:
        """Volatility-based position sizing."""
        if volatility <= 0:
            return 0.0
        
        # Scale position size inversely with volatility
        base_risk_amount = account_value * (self.config.account_risk_pct / 100)
        volatility_adjustment = min(0.02 / volatility, 2.0)  # Cap at 2x adjustment
        
        adjusted_risk_amount = base_risk_amount * volatility_adjustment
        risk_per_share = entry_price * volatility
        
        position_size = adjusted_risk_amount / risk_per_share
        
        # Apply limits
        max_position_value = account_value * 0.95
        max_shares = max_position_value / entry_price
        
        return min(position_size, max_shares)
    
    def _kelly_criterion_sizing(
        self, 
        entry_price: float, 
        account_value: float, 
        win_rate: float, 
        avg_win: float, 
        avg_loss: float
    ) -> float:
        """Kelly Criterion position sizing."""
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0
        
        # Kelly percentage
        kelly_pct = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_loss
        
        # Cap Kelly at maximum risk percentage and apply safety factor
        kelly_pct = min(kelly_pct, self.config.account_risk_pct / 100)
        kelly_pct *= 0.25  # Quarter Kelly for safety
        
        # Calculate position size
        position_value = account_value * kelly_pct
        position_size = position_value / entry_price
        
        return position_size
    
    def can_open_position(
        self, 
        symbol: str, 
        position_size: float, 
        entry_price: float,
        account_value: float
    ) -> Tuple[bool, str]:
        """
        Check if a new position can be opened based on risk limits.
        
        Returns:
            Tuple of (can_open, reason)
        """
        # Check daily loss limit
        if self.daily_loss_pct >= self.config.max_daily_loss_pct:
            return False, f"Daily loss limit reached: {self.daily_loss_pct:.2f}%"
        
        # Check weekly loss limit
        if self.weekly_loss_pct >= self.config.max_weekly_loss_pct:
            return False, f"Weekly loss limit reached: {self.weekly_loss_pct:.2f}%"
        
        # Check monthly loss limit
        if self.monthly_loss_pct >= self.config.max_monthly_loss_pct:
            return False, f"Monthly loss limit reached: {self.monthly_loss_pct:.2f}%"
        
        # Check maximum positions limit
        if len(self.positions) >= self.config.max_positions:
            return False, f"Maximum positions limit reached: {len(self.positions)}/{self.config.max_positions}"
        
        # Check position size reasonableness
        position_value = position_size * entry_price
        max_position_value = account_value * 0.9  # Max 90% of account
        
        if position_value > max_position_value:
            return False, f"Position size too large: ${position_value:,.2f} > ${max_position_value:,.2f}"
        
        # Check correlation if multiple positions
        if len(self.positions) > 1:
            correlation_check = self._check_correlation_limits(symbol)
            if not correlation_check[0]:
                return False, correlation_check[1]
        
        return True, "Position approved"
    
    def _check_correlation_limits(self, new_symbol: str) -> Tuple[bool, str]:
        """Check correlation limits with existing positions."""
        # This is a simplified correlation check
        # In practice, you would calculate actual correlations using historical data
        
        existing_symbols = list(self.positions.keys())
        
        # Simple rule: limit number of crypto positions
        crypto_symbols = [s for s in existing_symbols if 'USDT' in s or 'USD' in s]
        if 'USDT' in new_symbol or 'USD' in new_symbol:
            if len(crypto_symbols) >= 3:  # Max 3 crypto positions
                return False, "Maximum crypto correlation limit reached"
        
        return True, "Correlation check passed"
    
    def add_position(
        self, 
        symbol: str, 
        size: float, 
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        direction: str = "long"
    ):
        """Add a new position to tracking."""
        position = PositionInfo(
            symbol=symbol,
            size=size,
            entry_price=entry_price,
            current_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            direction=direction,
            entry_time=datetime.now()
        )
        
        self.positions[symbol] = position
    
    def remove_position(self, symbol: str, exit_price: float) -> Optional[Dict]:
        """Remove a position and record the trade."""
        if symbol not in self.positions:
            return None
        
        position = self.positions.pop(symbol)
        
        # Calculate trade P&L
        if position.direction == "long":
            pnl = position.size * (exit_price - position.entry_price)
        else:
            pnl = position.size * (position.entry_price - exit_price)
        
        pnl_pct = (exit_price - position.entry_price) / position.entry_price * 100
        if position.direction == "short":
            pnl_pct = -pnl_pct
        
        # Record trade
        trade = {
            'symbol': symbol,
            'direction': position.direction,
            'size': position.size,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'entry_time': position.entry_time,
            'exit_time': datetime.now(),
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'duration': datetime.now() - position.entry_time if position.entry_time else timedelta(0)
        }
        
        self.trade_history.append(trade)
        return trade
    
    def update_positions(self, price_data: Dict[str, float]):
        """Update current prices for all positions."""
        for symbol, position in self.positions.items():
            if symbol in price_data:
                position.current_price = price_data[symbol]
    
    def update_account_value(self, account_value: float):
        """Update account value and calculate daily/weekly/monthly P&L."""
        current_time = datetime.now()
        current_date = current_time.date()
        current_week = current_time.isocalendar()[:2]
        current_month = (current_time.year, current_time.month)
        
        # Add to history
        self.account_value_history.append((current_time, account_value))
        
        # Calculate period returns if we have previous values
        if len(self.account_value_history) > 1:
            # Daily return
            if current_date != self.last_reset_day:
                # Calculate daily P&L
                day_start_value = self._get_value_at_period_start('day')
                if day_start_value:
                    self.daily_pnl.append((current_time, account_value - day_start_value))
                    self.daily_loss_pct = max(0, (day_start_value - account_value) / day_start_value * 100)
                
                self.last_reset_day = current_date
            
            # Weekly return
            if current_week != self.last_reset_week:
                week_start_value = self._get_value_at_period_start('week')
                if week_start_value:
                    self.weekly_loss_pct = max(0, (week_start_value - account_value) / week_start_value * 100)
                
                self.last_reset_week = current_week
            
            # Monthly return
            if current_month != self.last_reset_month:
                month_start_value = self._get_value_at_period_start('month')
                if month_start_value:
                    self.monthly_loss_pct = max(0, (month_start_value - account_value) / month_start_value * 100)
                
                self.last_reset_month = current_month
    
    def _get_value_at_period_start(self, period: str) -> Optional[float]:
        """Get account value at the start of the specified period."""
        if not self.account_value_history:
            return None
        
        current_time = datetime.now()
        
        if period == 'day':
            start_time = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == 'week':
            days_since_monday = current_time.weekday()
            start_time = current_time - timedelta(days=days_since_monday)
            start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == 'month':
            start_time = current_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            return None
        
        # Find closest value to start time
        for timestamp, value in reversed(self.account_value_history):
            if timestamp <= start_time:
                return value
        
        return self.account_value_history[0][1] if self.account_value_history else None
    
    def calculate_portfolio_risk_metrics(self) -> RiskMetrics:
        """Calculate comprehensive risk metrics for the portfolio."""
        if not self.account_value_history:
            return RiskMetrics()
        
        # Extract returns
        values = [value for _, value in self.account_value_history]
        if len(values) < 2:
            return RiskMetrics()
        
        returns = np.array([(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))])
        
        if len(returns) == 0:
            return RiskMetrics()
        
        # Calculate metrics
        metrics = RiskMetrics()
        
        # Drawdown calculation
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak * 100
        metrics.max_drawdown_pct = np.max(drawdown)
        metrics.current_drawdown_pct = drawdown[-1]
        
        # VaR calculation
        if len(returns) > 1:
            metrics.daily_var_95 = np.percentile(returns, 5) * 100
            metrics.daily_var_99 = np.percentile(returns, 1) * 100
        
        # Volatility
        metrics.volatility_pct = np.std(returns) * np.sqrt(252) * 100  # Annualized
        
        # Sharpe ratio (assuming 0% risk-free rate)
        if np.std(returns) > 0:
            metrics.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        
        # Sortino ratio
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0 and np.std(negative_returns) > 0:
            metrics.sortino_ratio = np.mean(returns) / np.std(negative_returns) * np.sqrt(252)
        
        # Calmar ratio
        if metrics.max_drawdown_pct > 0:
            annual_return = (values[-1] / values[0]) ** (252 / len(values)) - 1
            metrics.calmar_ratio = annual_return * 100 / metrics.max_drawdown_pct
        
        # Trade-based metrics
        if self.trade_history:
            winning_trades = [t for t in self.trade_history if t['pnl'] > 0]
            losing_trades = [t for t in self.trade_history if t['pnl'] < 0]
            
            metrics.win_rate = len(winning_trades) / len(self.trade_history) * 100
            
            if winning_trades and losing_trades:
                avg_win = np.mean([t['pnl'] for t in winning_trades])
                avg_loss = abs(np.mean([t['pnl'] for t in losing_trades]))
                
                if avg_loss > 0:
                    metrics.profit_factor = avg_win / avg_loss
        
        return metrics
    
    def get_position_summary(self) -> Dict[str, any]:
        """Get summary of current positions."""
        if not self.positions:
            return {
                'total_positions': 0,
                'total_market_value': 0.0,
                'total_unrealized_pnl': 0.0,
                'positions': []
            }
        
        positions_data = []
        total_market_value = 0.0
        total_unrealized_pnl = 0.0
        
        for symbol, position in self.positions.items():
            position_data = {
                'symbol': symbol,
                'direction': position.direction,
                'size': position.size,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'market_value': position.market_value,
                'unrealized_pnl': position.unrealized_pnl,
                'unrealized_pnl_pct': position.unrealized_pnl_pct,
                'stop_loss': position.stop_loss,
                'take_profit': position.take_profit
            }
            
            positions_data.append(position_data)
            total_market_value += position.market_value
            total_unrealized_pnl += position.unrealized_pnl
        
        return {
            'total_positions': len(self.positions),
            'total_market_value': total_market_value,
            'total_unrealized_pnl': total_unrealized_pnl,
            'positions': positions_data
        }


# Global risk manager instance
risk_manager = RiskManager()
