"""
Complete Fractional DCA Backtesting System

All-in-one file implementing true fractional position sizing for DCA strategies:
- Monthly DCA Strategy (systematic buying on dips)
- Signal-Based DCA Strategy (10-point scoring system)
- Full analytics (win rate, Sharpe ratio, max drawdown, profit factor)
- JSON export for detailed analysis

No dependencies on backtesting.py framework - pure custom implementation.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json

from src.data.database import db


# =====================================================================
# DATA CLASSES FOR TRADE TRACKING
# =====================================================================

@dataclass
class Trade:
    """Record of a single trade execution."""
    date: datetime
    type: str  # 'BUY' or 'SELL'
    price: float
    amount_usd: float
    coins: float
    fee_usd: float = 0.0


class Position:
    """Tracks fractional cryptocurrency position with full precision."""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.coins = 0.0  # Total coins held (8 decimal places)
        self.total_invested = 0.0  # Total USD invested (cost basis)
        self.realized_pnl = 0.0  # Realized profit/loss from sells
        self.trades: List[Trade] = []
    
    def add_buy(self, date: datetime, price: float, amount_usd: float, commission: float = 0.001):
        """
        Buy fractional coins with USD.
        
        Args:
            date: Trade execution date
            price: Price per coin in USD
            amount_usd: Total USD to spend (including fees)
            commission: Fee rate (default 0.1%)
        
        Returns:
            Number of coins purchased
        """
        fee_usd = amount_usd * commission
        net_amount = amount_usd - fee_usd
        coins = net_amount / price
        
        self.coins += coins
        self.total_invested += amount_usd
        
        trade = Trade(
            date=date,
            type='BUY',
            price=price,
            amount_usd=amount_usd,
            coins=coins,
            fee_usd=fee_usd
        )
        self.trades.append(trade)
        
        return coins
    
    def add_sell(self, date: datetime, price: float, coins_to_sell: float, commission: float = 0.001):
        """
        Sell fractional coins for USD.
        
        Args:
            date: Trade execution date
            price: Price per coin in USD
            coins_to_sell: Number of coins to sell
            commission: Fee rate (default 0.1%)
        
        Returns:
            Net USD received after fees
        """
        if coins_to_sell > self.coins:
            coins_to_sell = self.coins
        
        gross_amount = coins_to_sell * price
        fee_usd = gross_amount * commission
        net_amount = gross_amount - fee_usd
        
        # Calculate cost basis for sold coins
        avg_cost = self.total_invested / self.coins if self.coins > 0 else 0
        cost_basis = coins_to_sell * avg_cost
        pnl = net_amount - cost_basis
        
        self.coins -= coins_to_sell
        self.total_invested -= cost_basis
        self.realized_pnl += pnl
        
        trade = Trade(
            date=date,
            type='SELL',
            price=price,
            amount_usd=gross_amount,
            coins=coins_to_sell,
            fee_usd=fee_usd
        )
        self.trades.append(trade)
        
        return net_amount
    
    def current_value(self, price: float) -> float:
        """Current market value of position."""
        return self.coins * price
    
    def unrealized_pnl(self, price: float) -> float:
        """Unrealized profit/loss at current price."""
        return self.current_value(price) - self.total_invested


# =====================================================================
# STRATEGY 1: MONTHLY DCA
# =====================================================================

class FractionalDCAStrategy:
    """
    Monthly DCA Strategy with true fractional position sizing.
    
    Buys on dips (2 red days or RSI < 30) using monthly budget.
    Sells 30% when 15% above 50 EMA.
    """
    
    def __init__(self, symbol: str, initial_cash: float, monthly_budget: float, 
                 red_days_trigger: int = 2, commission: float = 0.001):
        self.symbol = symbol
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.monthly_budget = monthly_budget
        self.red_days_trigger = red_days_trigger
        self.commission = commission
        
        self.position = Position(symbol=symbol)
        self.monthly_budget_used = 0.0
        self.red_day_count = 0
        self.last_buy_date = None
        self.last_sell_date = None
        self.monthly_buy_count = 0
        self.last_rebalance_month = None
    
    def reset_monthly_counters(self, current_date: datetime):
        """Add monthly budget and reset counters at start of new month."""
        if self.last_rebalance_month != current_date.month:
            self.cash += self.monthly_budget
            self.monthly_budget_used = 0.0
            self.monthly_buy_count = 0
            self.last_rebalance_month = current_date.month
    
    def check_buy_condition(self, data: pd.DataFrame, idx: int, current_date: datetime) -> Tuple[bool, float]:
        """
        Check if should buy and how much.
        
        Buy triggers:
        - 2+ consecutive red days OR
        - RSI < 30 (oversold)
        
        Limit: Max 5 buys per month
        """
        if idx < 50:  # Need data for indicators
            return False, 0.0
        
        current_row = data.iloc[idx]
        
        # Track red days
        if current_row['Close'] < current_row['Open']:
            self.red_day_count += 1
        else:
            self.red_day_count = 0
        
        # Calculate RSI (14-period)
        close_prices = data['Close'].iloc[max(0, idx-14):idx+1].values
        if len(close_prices) < 15:
            return False, 0.0
        
        delta = np.diff(close_prices)
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            rsi = 100
        else:
            rsi = 100 - (100 / (1 + (avg_gain / avg_loss)))
        
        # Check buy conditions
        has_red_days = self.red_day_count >= self.red_days_trigger
        is_oversold = rsi < 30
        
        if (has_red_days or is_oversold) and self.monthly_buy_count < 5:
            remaining_monthly = self.monthly_budget - self.monthly_budget_used
            available_cash = min(remaining_monthly, self.cash)
            
            if available_cash >= 10:  # Minimum $10 buy
                return True, min(remaining_monthly, available_cash)
        
        return False, 0.0
    
    def check_sell_condition(self, data: pd.DataFrame, idx: int, current_date: datetime) -> Tuple[bool, float]:
        """
        Check if should sell and how much.
        
        Sell trigger:
        - Price > 15% above 50 EMA
        - Minimum 7 days between sells
        
        Sell size: 30% of position
        """
        if self.position.coins == 0 or idx < 50:
            return False, 0.0
        
        # Minimum time between sells
        if self.last_sell_date and (current_date - self.last_sell_date).days < 7:
            return False, 0.0
        
        # Calculate 50 EMA
        close_prices = data['Close'].iloc[max(0, idx-50):idx+1].values
        ema_50 = close_prices[0]
        alpha = 2 / 51
        for price in close_prices[1:]:
            ema_50 = alpha * price + (1 - alpha) * ema_50
        
        current_price = data.iloc[idx]['Close']
        
        # Sell if 15% above 50 EMA
        if current_price > ema_50 * 1.15:
            return True, self.position.coins * 0.30
        
        return False, 0.0
    
    def run_backtest(self, data: pd.DataFrame) -> Dict:
        """Run complete backtest on historical data."""
        
        results = {
            'symbol': self.symbol,
            'start_date': data.index[0],
            'end_date': data.index[-1],
            'initial_cash': self.initial_cash,
        }
        
        for idx in range(len(data)):
            current_date = data.index[idx]
            
            # Monthly budget management
            self.reset_monthly_counters(current_date)
            
            # Check buy condition
            should_buy, buy_amount = self.check_buy_condition(data, idx, current_date)
            if should_buy:
                current_price = data.iloc[idx]['Close']
                self.position.add_buy(current_date, current_price, buy_amount, self.commission)
                self.cash -= buy_amount
                self.monthly_budget_used += buy_amount
                self.monthly_buy_count += 1
                self.last_buy_date = current_date
                self.red_day_count = 0  # Reset after buy
            
            # Check sell condition
            should_sell, coins_to_sell = self.check_sell_condition(data, idx, current_date)
            if should_sell:
                current_price = data.iloc[idx]['Close']
                cash_received = self.position.add_sell(current_date, current_price, coins_to_sell, self.commission)
                self.cash += cash_received
                self.last_sell_date = current_date
        
        # Final results
        final_price = data.iloc[-1]['Close']
        final_position_value = self.position.current_value(final_price)
        final_equity = self.cash + final_position_value
        
        results.update({
            'final_cash': self.cash,
            'final_position_value': final_position_value,
            'final_equity': final_equity,
            'total_return': ((final_equity - self.initial_cash) / self.initial_cash) * 100,
            'coins_held': self.position.coins,
            'realized_pnl': self.position.realized_pnl,
            'unrealized_pnl': self.position.unrealized_pnl(final_price),
            'total_trades': len(self.position.trades),
            'buy_trades': len([t for t in self.position.trades if t.type == 'BUY']),
            'sell_trades': len([t for t in self.position.trades if t.type == 'SELL']),
            'trades': self.position.trades
        })
        
        return results


# =====================================================================
# STRATEGY 2: SIGNAL-BASED DCA
# =====================================================================

class FractionalSignalDCAStrategy(FractionalDCAStrategy):
    """
    Signal-Based DCA with 10-point scoring system.
    
    Uses EMA-200 distance and RSI to score opportunities.
    Dynamic position sizing based on signal strength.
    """
    
    def __init__(self, symbol: str, initial_cash: float, monthly_contribution: float, commission: float = 0.001):
        super().__init__(symbol, initial_cash, monthly_contribution, 2, commission)
        
        self.monthly_contribution = monthly_contribution
        self.current_month = None
        self.last_buy_price = 0.0
        
        # Cash pools (FIXED - Do not optimize)
        self.cash_active = 0.0  # 70% for regular signals
        self.cash_reserve = 0.0  # 30% for extreme signals only
        self.active_cash_pct = 70.0  # FIXED
        self.reserve_cash_pct = 30.0  # FIXED
        
        # Signal thresholds (OPTIMIZABLE)
        self.min_signal_score = 6
        self.extreme_signal_score = 10
        self.strong_signal_threshold = 8  # Score threshold for strong_signal_multiplier
        
        # Indicator parameters (OPTIMIZABLE)
        self.rsi_period = 14
        self.ema_period = 200
        
        # EMA distance thresholds for scoring (OPTIMIZABLE)
        self.ema_distance_extreme = -18.0  # 3 points
        self.ema_distance_strong = -12.0   # 2 points
        self.ema_distance_moderate = -8.0  # 1 point
        
        # RSI oversold thresholds for scoring (OPTIMIZABLE)
        self.rsi_oversold_extreme = 35.0  # 2 points
        self.rsi_oversold_moderate = 45.0 # 1 point
        
        # Position sizing rules (FIXED)
        self.base_size_thresholds = {
            (600, 1200): 10,
            (1200, 2400): 15,
            (2400, 3600): 20,
            (3600, float('inf')): 25
        }
        self.price_drop_bonus_10 = 15.0  # +15% size on 10% price drop
        self.price_drop_bonus_20 = 25.0  # +25% size on 20% price drop
        self.strong_signal_multiplier = 1.3  # 1.3x for score 8-9
        self.extreme_signal_multiplier = 1.5  # 1.5x for score 10
    
    def reset_monthly_counters(self, current_date: datetime):
        """Add monthly contribution split 70/30 between active/reserve."""
        if self.current_month != (current_date.year, current_date.month):
            self.current_month = (current_date.year, current_date.month)
            
            active_portion = self.monthly_contribution * (self.active_cash_pct / 100)
            reserve_portion = self.monthly_contribution * (self.reserve_cash_pct / 100)
            
            self.cash_active += active_portion
            self.cash_reserve += reserve_portion
            self.cash += self.monthly_contribution
            
            self.monthly_budget_used = 0.0
            self.monthly_buy_count = 0
    
    def _calculate_signal_score(self, data: pd.DataFrame, idx: int) -> int:
        """
        Calculate 10-point signal score.
        
        Scoring:
        - Distance from EMA: 0-3 points (based on ema_distance thresholds)
        - RSI oversold: 0-2 points (based on rsi_oversold thresholds)
        - Total: 0-10 points
        """
        if idx < self.ema_period:
            return 0
        
        score = 0
        current_price = data.iloc[idx]['Close']
        
        # EMA distance scoring (using configurable ema_period)
        ema = data['Close'].iloc[max(0, idx-self.ema_period):idx+1].ewm(span=self.ema_period, adjust=False).mean().iloc[-1]
        
        # RSI calculation (using configurable rsi_period)
        closes = data['Close'].iloc[max(0, idx-(self.rsi_period+1)):idx+1]
        if len(closes) >= self.rsi_period + 1:
            deltas = closes.diff()
            gain = deltas.where(deltas > 0, 0).rolling(window=self.rsi_period).mean()
            loss = -deltas.where(deltas < 0, 0).rolling(window=self.rsi_period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi_value = rsi.iloc[-1]
        else:
            rsi_value = 50
        
        # Score from EMA distance (using configurable thresholds)
        if ema > 0:
            distance_pct = ((current_price - ema) / ema) * 100
            if distance_pct <= self.ema_distance_extreme:
                score += 3
            elif distance_pct <= self.ema_distance_strong:
                score += 2
            elif distance_pct <= self.ema_distance_moderate:
                score += 1
        
        # Score from RSI (using configurable thresholds)
        if rsi_value < self.rsi_oversold_extreme:
            score += 2
        elif rsi_value < self.rsi_oversold_moderate:
            score += 1
        
        return score
    
    def check_buy_condition(self, data: pd.DataFrame, idx: int, current_date: datetime) -> Tuple[bool, float]:
        """
        Check buy signal based on scoring system.
        
        Requirements:
        - Signal score >= min_signal_score
        - Price below EMA (configurable period)
        - 2 red days OR 5% single-day drop
        """
        if idx < self.ema_period:
            return False, 0.0
        
        current_price = data.iloc[idx]['Close']
        signal_score = self._calculate_signal_score(data, idx)
        
        # Minimum score required
        if signal_score < self.min_signal_score:
            return False, 0.0
        
        # Must be below EMA (using configurable ema_period)
        ema = data['Close'].iloc[max(0, idx-self.ema_period):idx+1].ewm(span=self.ema_period, adjust=False).mean().iloc[-1]
        if current_price > ema:
            return False, 0.0
        
        # Check red day conditions
        red_days = 0
        if idx >= 1 and data['Close'].iloc[idx] < data['Close'].iloc[idx-1]:
            red_days = 1
        if idx >= 2 and data['Close'].iloc[idx-1] < data['Close'].iloc[idx-2]:
            red_days += 1
        
        single_day_drop = 0.0
        if idx >= 1:
            single_day_drop = ((current_price - data['Close'].iloc[idx-1]) / data['Close'].iloc[idx-1]) * 100
        
        if not ((red_days >= 2) or (red_days >= 1 and single_day_drop <= -5.0)):
            return False, 0.0
        
        # Determine available cash
        is_extreme = signal_score >= self.extreme_signal_score
        available_cash = (self.cash_active + self.cash_reserve) if is_extreme else self.cash_active
        
        if available_cash < 50:
            return False, 0.0
        
        # Calculate position size
        total_contributed = self.monthly_contribution * max(1, self.monthly_buy_count + 1)
        
        # Base divisor from accumulated capital
        base_divisor = 20
        for (min_cash, max_cash), divisor in self.base_size_thresholds.items():
            if min_cash <= total_contributed < max_cash:
                base_divisor = divisor
                break
        
        base_size = available_cash / base_divisor
        size_multiplier = 1.0
        
        # Price drop bonus
        if self.last_buy_price > 0:
            price_drop_pct = ((current_price - self.last_buy_price) / self.last_buy_price) * 100
            if price_drop_pct <= -20:
                size_multiplier += (self.price_drop_bonus_20 / 100)
            elif price_drop_pct <= -10:
                size_multiplier += (self.price_drop_bonus_10 / 100)
        
        # Signal strength multiplier (using configurable strong_signal_threshold)
        if signal_score >= self.extreme_signal_score:
            size_multiplier *= self.extreme_signal_multiplier
        elif signal_score >= self.strong_signal_threshold:
            size_multiplier *= self.strong_signal_multiplier
        
        buy_amount = min(base_size * size_multiplier, available_cash)
        
        # Deduct from appropriate cash pool
        if buy_amount > 0:
            if is_extreme:
                if buy_amount <= self.cash_active:
                    self.cash_active -= buy_amount
                else:
                    remaining = buy_amount - self.cash_active
                    self.cash_active = 0
                    self.cash_reserve -= remaining
            else:
                self.cash_active -= buy_amount
            
            self.last_buy_price = current_price
            return True, buy_amount
        
        return False, 0.0


# =====================================================================
# ANALYTICS
# =====================================================================

def calculate_analytics(results: Dict, data: pd.DataFrame) -> Dict:
    """Calculate win rate, Sharpe ratio, max drawdown, profit factor."""
    
    analytics = {}
    trades = results['trades']
    
    if len(trades) == 0:
        return {
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0
        }
    
    # Win rate calculation
    sell_trades = [t for t in trades if t.type == 'SELL']
    profitable_sells = 0
    total_wins = 0.0
    total_losses = 0.0
    
    buy_prices = []
    for trade in trades:
        if trade.type == 'BUY':
            buy_prices.append(trade.price)
        elif trade.type == 'SELL' and len(buy_prices) > 0:
            avg_buy_price = np.mean(buy_prices)
            profit_pct = ((trade.price - avg_buy_price) / avg_buy_price) * 100
            if profit_pct > 0:
                profitable_sells += 1
                total_wins += profit_pct
            else:
                total_losses += abs(profit_pct)
    
    analytics['win_rate'] = (profitable_sells / len(sell_trades) * 100) if len(sell_trades) > 0 else 0.0
    analytics['avg_win'] = (total_wins / profitable_sells) if profitable_sells > 0 else 0.0
    analytics['avg_loss'] = (total_losses / (len(sell_trades) - profitable_sells)) if (len(sell_trades) - profitable_sells) > 0 else 0.0
    analytics['profit_factor'] = (total_wins / total_losses) if total_losses > 0 else 0.0
    
    # Sharpe ratio and drawdown
    if len(data) > 1:
        equity_curve = []
        for idx in range(len(data)):
            price = data.iloc[idx]['Close']
            position_value = results['coins_held'] * price
            equity = results['final_cash'] + position_value
            equity_curve.append(equity)
        
        returns = np.diff(equity_curve) / equity_curve[:-1]
        analytics['sharpe_ratio'] = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0.0
        
        cumulative = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - cumulative) / cumulative
        analytics['max_drawdown'] = abs(np.min(drawdowns)) * 100
    else:
        analytics['sharpe_ratio'] = 0.0
        analytics['max_drawdown'] = 0.0
    
    return analytics


# =====================================================================
# MAIN BACKTEST RUNNER
# =====================================================================

def run_comparison_backtest(
    strategy_type: str = 'both',
    initial_capital: float = 10000.0,
    monthly_contribution: float = 600.0,
    start_date_str: str = '2017-01-01',
    end_date_str: str = '2025-10-31'
) -> Dict:
    """
    Run fractional DCA backtest comparison.
    
    Args:
        strategy_type: 'monthly', 'signal', or 'both'
        initial_capital: Starting capital
        monthly_contribution: Monthly DCA amount
        start_date_str: Start date (YYYY-MM-DD)
        end_date_str: End date (YYYY-MM-DD)
    
    Returns:
        Dictionary with results for all strategies and assets
    """
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'TRXUSDT']
    allocation_ratios = {
        'BTCUSDT': 0.50,
        'ETHUSDT': 0.25,
        'BNBUSDT': 0.15,
        'TRXUSDT': 0.10
    }
    
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    
    # Determine which strategies to run
    strategies_to_test = []
    if strategy_type in ['monthly', 'both']:
        strategies_to_test.append(('Monthly DCA', FractionalDCAStrategy))
    if strategy_type in ['signal', 'both']:
        strategies_to_test.append(('Signal-Based DCA', FractionalSignalDCAStrategy))
    
    all_results = {}
    
    for strategy_name, StrategyClass in strategies_to_test:
        print("\n" + "=" * 80)
        print(f"FRACTIONAL BACKTEST - {strategy_name}")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  Initial Capital: ${initial_capital:,.2f}")
        print(f"  Monthly Contribution: ${monthly_contribution:,.2f}")
        print(f"  Period: {start_date.date()} to {end_date.date()}")
        print(f"  Allocation: BTC 50%, ETH 25%, BNB 15%, TRX 10%")
        
        strategy_results = {}
        
        for symbol in symbols:
            print(f"\n{'-' * 80}")
            print(f"Backtesting {symbol}...")
            
            # Load data
            data = db.get_market_data(symbol, '1d', start_date, end_date)
            if data is None or len(data) == 0:
                print(f"  ERROR: No data for {symbol}")
                continue
            
            # Prepare data
            data = data.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            if 'timestamp' in data.columns:
                data.set_index('timestamp', inplace=True)
            
            # Allocate capital
            allocated_cash = initial_capital * allocation_ratios[symbol]
            monthly_budget = monthly_contribution * allocation_ratios[symbol]
            
            print(f"  Loaded {len(data)} candles")
            print(f"  Allocated: ${allocated_cash:,.2f} + ${monthly_budget:,.2f}/month")
            
            # Initialize strategy
            if StrategyClass == FractionalDCAStrategy:
                strategy = StrategyClass(
                    symbol=symbol,
                    initial_cash=allocated_cash,
                    monthly_budget=monthly_budget,
                    red_days_trigger=2,
                    commission=0.001
                )
            else:  # Signal-Based
                strategy = StrategyClass(
                    symbol=symbol,
                    initial_cash=allocated_cash,
                    monthly_contribution=monthly_budget,
                    commission=0.001
                )
            
            # Run backtest
            results = strategy.run_backtest(data)
            analytics = calculate_analytics(results, data)
            results['analytics'] = analytics
            strategy_results[symbol] = results
            
            # Print results
            print(f"\n  Performance:")
            print(f"    Final Equity: ${results['final_equity']:,.2f}")
            print(f"    Return: {results['total_return']:+.2f}%")
            print(f"    Trades: {results['total_trades']} ({results['buy_trades']} buys, {results['sell_trades']} sells)")
            print(f"\n  Analytics:")
            print(f"    Win Rate: {analytics['win_rate']:.1f}%")
            print(f"    Sharpe Ratio: {analytics['sharpe_ratio']:.2f}")
            print(f"    Max Drawdown: {analytics['max_drawdown']:.2f}%")
        
        # Portfolio summary
        print("\n" + "=" * 80)
        print("PORTFOLIO SUMMARY")
        print("=" * 80)
        
        total_initial = sum([initial_capital * allocation_ratios[s] for s in symbols if s in strategy_results])
        total_final = sum([strategy_results[s]['final_equity'] for s in strategy_results])
        total_return = ((total_final - total_initial) / total_initial) * 100
        total_trades = sum([strategy_results[s]['total_trades'] for s in strategy_results])
        
        avg_win_rate = np.mean([strategy_results[s]['analytics']['win_rate'] for s in strategy_results])
        avg_sharpe = np.mean([strategy_results[s]['analytics']['sharpe_ratio'] for s in strategy_results])
        max_dd = np.max([strategy_results[s]['analytics']['max_drawdown'] for s in strategy_results])
        
        print(f"\nPortfolio Performance:")
        print(f"  Starting Value: ${total_initial:,.2f}")
        print(f"  Ending Value: ${total_final:,.2f}")
        print(f"  Total Return: {total_return:+.2f}%")
        print(f"  Total Trades: {total_trades}")
        print(f"\nPortfolio Analytics:")
        print(f"  Avg Win Rate: {avg_win_rate:.1f}%")
        print(f"  Avg Sharpe Ratio: {avg_sharpe:.2f}")
        print(f"  Max Drawdown: {max_dd:.2f}%")
        
        print(f"\nPer-Asset Performance:")
        for symbol in symbols:
            if symbol in strategy_results:
                r = strategy_results[symbol]
                a = r['analytics']
                print(f"  {symbol}: ${r['final_equity']:,.2f} ({r['total_return']:+.2f}%) - {r['total_trades']} trades - Win Rate: {a['win_rate']:.1f}%")
        
        all_results[strategy_name] = {
            'portfolio': {
                'initial_value': total_initial,
                'final_value': total_final,
                'total_return': total_return,
                'total_trades': total_trades,
                'avg_win_rate': avg_win_rate,
                'avg_sharpe': avg_sharpe,
                'max_drawdown': max_dd
            },
            'assets': strategy_results
        }
    
    return all_results


def save_results(results: Dict, filename: str = None):
    """Save backtest results to JSON file."""
    
    results_dir = Path(__file__).parent / 'backtest_results'
    results_dir.mkdir(exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'fractional_dca_comparison_{timestamp}.json'
    
    filepath = results_dir / filename
    
    # Convert to JSON-serializable format
    json_results = {}
    for strategy_name, strategy_data in results.items():
        json_results[strategy_name] = {
            'portfolio': strategy_data['portfolio'],
            'assets': {}
        }
        
        for symbol, asset_data in strategy_data['assets'].items():
            # Convert trades
            trades_list = []
            for trade in asset_data['trades']:
                trades_list.append({
                    'date': trade.date.isoformat(),
                    'type': trade.type,
                    'price': float(trade.price),
                    'amount_usd': float(trade.amount_usd),
                    'coins': float(trade.coins),
                    'fee_usd': float(trade.fee_usd)
                })
            
            json_results[strategy_name]['assets'][symbol] = {
                'symbol': asset_data['symbol'],
                'start_date': asset_data['start_date'].isoformat(),
                'end_date': asset_data['end_date'].isoformat(),
                'initial_cash': float(asset_data['initial_cash']),
                'final_cash': float(asset_data['final_cash']),
                'final_position_value': float(asset_data['final_position_value']),
                'final_equity': float(asset_data['final_equity']),
                'total_return': float(asset_data['total_return']),
                'coins_held': float(asset_data['coins_held']),
                'realized_pnl': float(asset_data['realized_pnl']),
                'unrealized_pnl': float(asset_data['unrealized_pnl']),
                'total_trades': asset_data['total_trades'],
                'buy_trades': asset_data['buy_trades'],
                'sell_trades': asset_data['sell_trades'],
                'analytics': asset_data['analytics'],
                'trades': trades_list
            }
    
    with open(filepath, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n{'=' * 80}")
    print(f"Results saved to: {filepath}")
    print(f"{'=' * 80}\n")
    
    return filepath


if __name__ == "__main__":
    # Run both strategies
    results = run_comparison_backtest(strategy_type='both')
    
    # Save to JSON
    save_results(results)
