"""
DCA Strategy Comparison Runner

This script runs and compares both DCA strategies across multiple cryptocurrencies:
- BTC (Bitcoin)
- ETH (Ethereum)
- BNB (Binance Coin)
- TRX (Tron)

It generates comprehensive comparison reports and visualizations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from backtesting import Backtest
import json
from pathlib import Path

from src.strategies.dca_strategies import DCAMonthlyStrategy, DCASignalStrategy
from src.data.database import db
from config.settings import TimeFrame


class DCAStrategyComparison:
    """
    Compare two DCA strategies across multiple assets.
    """
    
    def __init__(
        self,
        symbols: List[str] = None,
        timeframe: str = '1d',
        start_date: str = '2023-01-01',
        end_date: str = '2024-12-31',
        initial_cash: float = 10000.0,
        monthly_contribution: float = 600.0
    ):
        """
        Initialize the comparison runner.
        
        Args:
            symbols: List of crypto symbols (default: BTC, ETH, BNB, TRX)
            timeframe: Data timeframe
            start_date: Backtest start date
            end_date: Backtest end date
            initial_cash: Starting capital
            monthly_contribution: Monthly DCA contribution
        """
        self.symbols = symbols or ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'TRXUSDT']
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.monthly_contribution = monthly_contribution
        
        # Results storage
        self.results = {
            'monthly': {},
            'signal': {}
        }
        
        self.allocation_ratios = {
            'BTCUSDT': 0.50,
            'ETHUSDT': 0.25,
            'BNBUSDT': 0.15,
            'TRXUSDT': 0.10
        }
    
    def load_data(self, symbol: str) -> pd.DataFrame:
        """
        Load historical data for a symbol.
        
        Args:
            symbol: Symbol to load
        
        Returns:
            DataFrame with OHLCV data
        """
        print(f"📊 Loading data for {symbol}...")
        
        # Try to load from database
        try:
            data = db.get_ohlcv_data(
                symbol=symbol,
                timeframe=self.timeframe,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            if data is not None and len(data) > 0:
                # Prepare data for backtesting.py
                data = data.rename(columns={
                    'timestamp': 'Date',
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })
                
                if 'Date' in data.columns:
                    data['Date'] = pd.to_datetime(data['Date'])
                    data = data.set_index('Date')
                
                print(f"✅ Loaded {len(data)} candles for {symbol}")
                return data
            else:
                print(f"⚠️  No data found for {symbol}")
                return None
                
        except Exception as e:
            print(f"❌ Error loading data for {symbol}: {e}")
            return None
    
    def run_single_backtest(
        self,
        strategy_class,
        symbol: str,
        data: pd.DataFrame
    ) -> Dict:
        """
        Run backtest for a single symbol with a strategy.
        
        Args:
            strategy_class: Strategy class to use
            symbol: Symbol being tested
            data: OHLCV data
        
        Returns:
            Dictionary with backtest results
        """
        try:
            # Calculate cash allocation for this symbol
            symbol_allocation = self.allocation_ratios.get(symbol, 0.25)
            symbol_cash = self.initial_cash * symbol_allocation
            
            # Set strategy parameters
            strategy_params = {
                'monthly_budget': self.monthly_contribution * symbol_allocation
            }
            
            # Create backtest
            bt = Backtest(
                data,
                strategy_class,
                cash=symbol_cash,
                commission=0.001,  # 0.1% commission
                exclusive_orders=True
            )
            
            # Run backtest
            print(f"  🔄 Running {strategy_class.__name__} on {symbol}...")
            stats = bt.run(**strategy_params)
            
            # Extract key metrics
            result = {
                'symbol': symbol,
                'strategy': strategy_class.__name__,
                'start_value': symbol_cash,
                'end_value': stats['Equity Final [$]'],
                'return_pct': stats['Return [%]'],
                'buy_hold_return': stats['Buy & Hold Return [%]'],
                'num_trades': stats['# Trades'],
                'win_rate': stats['Win Rate [%]'],
                'profit_factor': stats.get('Profit Factor', 0),
                'max_drawdown': stats['Max. Drawdown [%]'],
                'sharpe_ratio': stats.get('Sharpe Ratio', 0),
                'sortino_ratio': stats.get('Sortino Ratio', 0),
                'avg_trade': stats.get('Avg. Trade [%]', 0),
                'trades': stats._trades.to_dict('records') if hasattr(stats, '_trades') else [],
                'equity_curve': stats._equity_curve['Equity'].to_dict() if hasattr(stats, '_equity_curve') else {}
            }
            
            print(f"  ✅ {symbol}: Return {result['return_pct']:.2f}%, Trades: {result['num_trades']}")
            
            return result
            
        except Exception as e:
            print(f"  ❌ Error running backtest for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_strategy_comparison(self):
        """
        Run both strategies across all symbols and compare.
        """
        print("🚀 Starting DCA Strategy Comparison")
        print("=" * 70)
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Initial Cash: ${self.initial_cash:,.2f}")
        print(f"Monthly Contribution: ${self.monthly_contribution:,.2f}")
        print("=" * 70)
        
        # Run Strategy 1: Monthly DCA
        print("\n📈 Strategy 1: Monthly DCA with Rebalancing")
        print("-" * 70)
        
        for symbol in self.symbols:
            data = self.load_data(symbol)
            if data is not None and len(data) > 200:  # Need enough data for 200 EMA
                result = self.run_single_backtest(DCAMonthlyStrategy, symbol, data)
                if result:
                    self.results['monthly'][symbol] = result
        
        # Run Strategy 2: Signal-Based DCA
        print("\n📊 Strategy 2: Signal-Based DCA")
        print("-" * 70)
        
        for symbol in self.symbols:
            data = self.load_data(symbol)
            if data is not None and len(data) > 200:
                result = self.run_single_backtest(DCASignalStrategy, symbol, data)
                if result:
                    self.results['signal'][symbol] = result
        
        # Generate comparison report
        self.generate_comparison_report()
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report."""
        print("\n" + "=" * 70)
        print("📋 STRATEGY COMPARISON REPORT")
        print("=" * 70)
        
        # Calculate portfolio totals
        monthly_totals = self._calculate_portfolio_totals('monthly')
        signal_totals = self._calculate_portfolio_totals('signal')
        
        # Overall Portfolio Performance
        print("\n🎯 OVERALL PORTFOLIO PERFORMANCE")
        print("-" * 70)
        
        print("\nStrategy 1: Monthly DCA")
        print(f"  Total Start Value:  ${monthly_totals['start_value']:,.2f}")
        print(f"  Total End Value:    ${monthly_totals['end_value']:,.2f}")
        print(f"  Total Return:       {monthly_totals['return_pct']:.2f}%")
        print(f"  Total Trades:       {monthly_totals['num_trades']}")
        print(f"  Avg Win Rate:       {monthly_totals['avg_win_rate']:.2f}%")
        print(f"  Max Drawdown:       {monthly_totals['max_drawdown']:.2f}%")
        
        print("\nStrategy 2: Signal-Based DCA")
        print(f"  Total Start Value:  ${signal_totals['start_value']:,.2f}")
        print(f"  Total End Value:    ${signal_totals['end_value']:,.2f}")
        print(f"  Total Return:       {signal_totals['return_pct']:.2f}%")
        print(f"  Total Trades:       {signal_totals['num_trades']}")
        print(f"  Avg Win Rate:       {signal_totals['avg_win_rate']:.2f}%")
        print(f"  Max Drawdown:       {signal_totals['max_drawdown']:.2f}%")
        
        # Winner
        print("\n🏆 WINNER")
        print("-" * 70)
        if monthly_totals['return_pct'] > signal_totals['return_pct']:
            diff = monthly_totals['return_pct'] - signal_totals['return_pct']
            print(f"Monthly DCA Strategy wins by {diff:.2f}% return")
        else:
            diff = signal_totals['return_pct'] - monthly_totals['return_pct']
            print(f"Signal-Based DCA Strategy wins by {diff:.2f}% return")
        
        # Per-Symbol Breakdown
        print("\n📊 PER-SYMBOL PERFORMANCE")
        print("-" * 70)
        
        for symbol in self.symbols:
            print(f"\n{symbol}:")
            
            if symbol in self.results['monthly']:
                m = self.results['monthly'][symbol]
                print(f"  Monthly DCA:  Return: {m['return_pct']:7.2f}% | Trades: {m['num_trades']:3d} | Win Rate: {m['win_rate']:5.1f}%")
            else:
                print(f"  Monthly DCA:  No data")
            
            if symbol in self.results['signal']:
                s = self.results['signal'][symbol]
                print(f"  Signal DCA:   Return: {s['return_pct']:7.2f}% | Trades: {s['num_trades']:3d} | Win Rate: {s['win_rate']:5.1f}%")
            else:
                print(f"  Signal DCA:   No data")
        
        # Trading Activity Comparison
        print("\n📈 TRADING ACTIVITY ANALYSIS")
        print("-" * 70)
        
        print("\nMonthly DCA Strategy:")
        for symbol in self.symbols:
            if symbol in self.results['monthly']:
                r = self.results['monthly'][symbol]
                print(f"  {symbol}: {r['num_trades']} trades, avg {r['avg_trade']:.2f}% per trade")
        
        print("\nSignal-Based DCA Strategy:")
        for symbol in self.symbols:
            if symbol in self.results['signal']:
                r = self.results['signal'][symbol]
                print(f"  {symbol}: {r['num_trades']} trades, avg {r['avg_trade']:.2f}% per trade")
        
        # Risk Metrics
        print("\n⚠️  RISK METRICS")
        print("-" * 70)
        
        print("\nMaximum Drawdown by Symbol:")
        for symbol in self.symbols:
            print(f"  {symbol}:")
            if symbol in self.results['monthly']:
                print(f"    Monthly DCA:  {self.results['monthly'][symbol]['max_drawdown']:6.2f}%")
            if symbol in self.results['signal']:
                print(f"    Signal DCA:   {self.results['signal'][symbol]['max_drawdown']:6.2f}%")
        
        # Key Insights
        print("\n💡 KEY INSIGHTS")
        print("-" * 70)
        self._generate_insights(monthly_totals, signal_totals)
        
        # Save results to file
        self._save_results()
    
    def _calculate_portfolio_totals(self, strategy_key: str) -> Dict:
        """Calculate portfolio-level totals for a strategy."""
        results = self.results[strategy_key]
        
        if not results:
            return {
                'start_value': 0,
                'end_value': 0,
                'return_pct': 0,
                'num_trades': 0,
                'avg_win_rate': 0,
                'max_drawdown': 0
            }
        
        total_start = sum(r['start_value'] for r in results.values())
        total_end = sum(r['end_value'] for r in results.values())
        total_return = ((total_end - total_start) / total_start * 100) if total_start > 0 else 0
        total_trades = sum(r['num_trades'] for r in results.values())
        avg_win_rate = np.mean([r['win_rate'] for r in results.values()])
        max_drawdown = np.mean([r['max_drawdown'] for r in results.values()])  # Average of max drawdowns
        
        return {
            'start_value': total_start,
            'end_value': total_end,
            'return_pct': total_return,
            'num_trades': total_trades,
            'avg_win_rate': avg_win_rate,
            'max_drawdown': max_drawdown
        }
    
    def _generate_insights(self, monthly_totals: Dict, signal_totals: Dict):
        """Generate key insights from the comparison."""
        insights = []
        
        # Return comparison
        if monthly_totals['return_pct'] > signal_totals['return_pct']:
            insights.append(f"✓ Monthly DCA generated higher returns ({monthly_totals['return_pct']:.2f}% vs {signal_totals['return_pct']:.2f}%)")
        else:
            insights.append(f"✓ Signal-Based DCA generated higher returns ({signal_totals['return_pct']:.2f}% vs {monthly_totals['return_pct']:.2f}%)")
        
        # Trading frequency
        if monthly_totals['num_trades'] > signal_totals['num_trades']:
            insights.append(f"✓ Monthly DCA traded more frequently ({monthly_totals['num_trades']} vs {signal_totals['num_trades']} trades)")
        else:
            insights.append(f"✓ Signal-Based DCA traded more selectively ({signal_totals['num_trades']} vs {monthly_totals['num_trades']} trades)")
        
        # Win rate
        if monthly_totals['avg_win_rate'] > signal_totals['avg_win_rate']:
            insights.append(f"✓ Monthly DCA had better win rate ({monthly_totals['avg_win_rate']:.1f}% vs {signal_totals['avg_win_rate']:.1f}%)")
        else:
            insights.append(f"✓ Signal-Based DCA had better win rate ({signal_totals['avg_win_rate']:.1f}% vs {monthly_totals['avg_win_rate']:.1f}%)")
        
        # Risk (drawdown)
        if monthly_totals['max_drawdown'] < signal_totals['max_drawdown']:
            insights.append(f"✓ Monthly DCA had lower drawdown risk ({monthly_totals['max_drawdown']:.2f}% vs {signal_totals['max_drawdown']:.2f}%)")
        else:
            insights.append(f"✓ Signal-Based DCA had lower drawdown risk ({signal_totals['max_drawdown']:.2f}% vs {monthly_totals['max_drawdown']:.2f}%)")
        
        # Capital efficiency
        monthly_capital_efficiency = monthly_totals['return_pct'] / max(monthly_totals['num_trades'], 1)
        signal_capital_efficiency = signal_totals['return_pct'] / max(signal_totals['num_trades'], 1)
        
        if signal_capital_efficiency > monthly_capital_efficiency:
            insights.append(f"✓ Signal-Based DCA was more capital efficient ({signal_capital_efficiency:.2f}% per trade vs {monthly_capital_efficiency:.2f}%)")
        else:
            insights.append(f"✓ Monthly DCA was more capital efficient ({monthly_capital_efficiency:.2f}% per trade vs {signal_capital_efficiency:.2f}%)")
        
        for insight in insights:
            print(f"  {insight}")
    
    def _save_results(self):
        """Save results to JSON file."""
        output_dir = Path('backtest_results')
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = output_dir / f'dca_comparison_{timestamp}.json'
        
        # Prepare data for JSON serialization
        save_data = {
            'config': {
                'symbols': self.symbols,
                'timeframe': self.timeframe,
                'start_date': self.start_date,
                'end_date': self.end_date,
                'initial_cash': self.initial_cash,
                'monthly_contribution': self.monthly_contribution
            },
            'results': {
                'monthly': {k: {key: val for key, val in v.items() if key not in ['trades', 'equity_curve']} 
                           for k, v in self.results['monthly'].items()},
                'signal': {k: {key: val for key, val in v.items() if key not in ['trades', 'equity_curve']} 
                          for k, v in self.results['signal'].items()}
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")


def main():
    """Main entry point for DCA strategy comparison."""
    print("🎯 DCA Strategy Comparison Tool")
    print("=" * 70)
    
    # Configuration
    comparison = DCAStrategyComparison(
        symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'TRXUSDT'],
        timeframe='1d',
        start_date='2023-01-01',
        end_date='2024-10-31',
        initial_cash=10000.0,
        monthly_contribution=600.0
    )
    
    # Run comparison
    comparison.run_strategy_comparison()
    
    print("\n" + "=" * 70)
    print("✅ Comparison complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
