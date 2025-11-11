"""
Comprehensive comparison of:
1. Scaling approach (1000x) with backtesting.py framework
2. Fractional approach with custom engine

Shows the differences in methodology and results.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import json
from datetime import datetime

def load_scaling_results():
    """Load the most recent scaling approach results."""
    results_dir = Path(__file__).parent / 'backtest_results'
    
    # Find the most recent dca_comparison file
    scaling_files = list(results_dir.glob('dca_comparison_*.json'))
    if not scaling_files:
        return None
    
    latest_file = max(scaling_files, key=lambda p: p.stat().st_mtime)
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    # Transform to expected format
    results = {}
    
    # Process monthly strategy
    if 'results' in data and 'monthly' in data['results']:
        monthly_data = data['results']['monthly']
        total_start = sum([v['start_value'] for v in monthly_data.values()])
        total_end = sum([v['end_value'] for v in monthly_data.values()])
        total_return = ((total_end - total_start) / total_start) * 100
        total_trades = sum([v['num_trades'] for v in monthly_data.values()])
        
        results['Monthly DCA'] = {
            'portfolio_summary': {
                'total_start': total_start / 1000,  # Scale back from 1000x
                'total_end': total_end / 1000,
                'total_return': total_return,
                'total_trades': total_trades
            },
            'results': [
                {
                    'symbol': symbol,
                    'metrics': {
                        'final_equity': data['end_value'] / 1000,
                        'return_pct': data['return_pct'],
                        'num_trades': data['num_trades'],
                        'win_rate': data['win_rate']
                    }
                }
                for symbol, data in monthly_data.items()
            ]
        }
    
    # Process signal strategy
    if 'results' in data and 'signal' in data['results']:
        signal_data = data['results']['signal']
        total_start = sum([v['start_value'] for v in signal_data.values()])
        total_end = sum([v['end_value'] for v in signal_data.values()])
        total_return = ((total_end - total_start) / total_start) * 100
        total_trades = sum([v['num_trades'] for v in signal_data.values()])
        
        results['Signal-Based DCA'] = {
            'portfolio_summary': {
                'total_start': total_start / 1000,
                'total_end': total_end / 1000,
                'total_return': total_return,
                'total_trades': total_trades
            },
            'results': [
                {
                    'symbol': symbol,
                    'metrics': {
                        'final_equity': data['end_value'] / 1000,
                        'return_pct': data['return_pct'],
                        'num_trades': data['num_trades'],
                        'win_rate': data['win_rate']
                    }
                }
                for symbol, data in signal_data.items()
            ]
        }
    
    return results

def load_fractional_results():
    """Load the most recent fractional approach results."""
    results_dir = Path(__file__).parent / 'backtest_results'
    
    # Find the most recent fractional_dca_comparison file
    fractional_files = list(results_dir.glob('fractional_dca_comparison_*.json'))
    if not fractional_files:
        return None
    
    latest_file = max(fractional_files, key=lambda p: p.stat().st_mtime)
    
    with open(latest_file, 'r') as f:
        return json.load(f)

def print_comparison():
    """Print detailed comparison of both approaches."""
    
    scaling_results = load_scaling_results()
    fractional_results = load_fractional_results()
    
    if not scaling_results or not fractional_results:
        print("ERROR: Could not load results files")
        return
    
    print("=" * 100)
    print("DCA STRATEGY COMPARISON - SCALING vs FRACTIONAL APPROACH")
    print("=" * 100)
    
    # Overview of methodologies
    print("\n" + "=" * 100)
    print("METHODOLOGY COMPARISON")
    print("=" * 100)
    
    print("\n1. SCALING APPROACH (backtesting.py framework)")
    print("   " + "-" * 80)
    print("   • Multiplies capital by 1000x to work around fractional limitations")
    print("   • Initial: $10,000 → $10,000,000 (scaled)")
    print("   • Monthly: $600 → $600,000 (scaled)")
    print("   • Results divided by 1000 to get actual values")
    print("   • Uses backtesting.py framework (size = fraction of equity)")
    print("   • Pros: Leverages existing framework, stable")
    print("   • Cons: Unrealistic position sizes during backtest")
    
    print("\n2. FRACTIONAL APPROACH (custom engine)")
    print("   " + "-" * 80)
    print("   • Direct fractional share tracking: coins = amount_usd / price")
    print("   • Example: $300 at BTC $74,500 = 0.00402685 BTC exactly")
    print("   • No scaling needed - uses actual capital amounts")
    print("   • Custom Position class with add_buy/add_sell methods")
    print("   • Pros: Realistic position sizing, true fractional support")
    print("   • Cons: Custom implementation, more complex")
    
    # Compare Monthly DCA strategies
    print("\n" + "=" * 100)
    print("MONTHLY DCA STRATEGY COMPARISON")
    print("=" * 100)
    
    scaling_monthly = scaling_results.get('Monthly DCA', {})
    fractional_monthly = fractional_results.get('Monthly DCA', {})
    
    if scaling_monthly and fractional_monthly:
        print("\n┌─────────────────────────────────────────────────────────────────────────────┐")
        print("│                           SCALING APPROACH                                  │")
        print("└─────────────────────────────────────────────────────────────────────────────┘")
        
        s_portfolio = scaling_monthly.get('portfolio_summary', {})
        print(f"\n  Portfolio Performance:")
        print(f"    Initial Value:      ${s_portfolio.get('total_start', 0):,.2f}")
        print(f"    Final Value:        ${s_portfolio.get('total_end', 0):,.2f}")
        print(f"    Total Return:       {s_portfolio.get('total_return', 0):+.2f}%")
        print(f"    Total Trades:       {s_portfolio.get('total_trades', 0)}")
        
        print(f"\n  Per-Asset Performance:")
        for symbol_data in scaling_monthly.get('results', []):
            symbol = symbol_data.get('symbol', 'UNKNOWN')
            metrics = symbol_data.get('metrics', {})
            print(f"    {symbol:8s}: ${metrics.get('final_equity', 0):>10,.2f} "
                  f"({metrics.get('return_pct', 0):>+7.2f}%) - "
                  f"{metrics.get('num_trades', 0):>4d} trades - "
                  f"Win Rate: {metrics.get('win_rate', 0):>5.1f}%")
        
        print("\n┌─────────────────────────────────────────────────────────────────────────────┐")
        print("│                         FRACTIONAL APPROACH                                 │")
        print("└─────────────────────────────────────────────────────────────────────────────┘")
        
        f_portfolio = fractional_monthly.get('portfolio', {})
        print(f"\n  Portfolio Performance:")
        print(f"    Initial Value:      ${f_portfolio.get('initial_value', 0):,.2f}")
        print(f"    Final Value:        ${f_portfolio.get('final_value', 0):,.2f}")
        print(f"    Total Return:       {f_portfolio.get('total_return', 0):+.2f}%")
        print(f"    Total Trades:       {f_portfolio.get('total_trades', 0)}")
        print(f"    Avg Win Rate:       {f_portfolio.get('avg_win_rate', 0):.1f}%")
        print(f"    Avg Sharpe Ratio:   {f_portfolio.get('avg_sharpe', 0):.2f}")
        print(f"    Max Drawdown:       {f_portfolio.get('max_drawdown', 0):.2f}%")
        
        print(f"\n  Per-Asset Performance:")
        for symbol, asset_data in fractional_monthly.get('assets', {}).items():
            analytics = asset_data.get('analytics', {})
            print(f"    {symbol:8s}: ${asset_data.get('final_equity', 0):>10,.2f} "
                  f"({asset_data.get('total_return', 0):>+7.2f}%) - "
                  f"{asset_data.get('total_trades', 0):>4d} trades - "
                  f"Win Rate: {analytics.get('win_rate', 0):>5.1f}% - "
                  f"Sharpe: {analytics.get('sharpe_ratio', 0):>5.2f}")
        
        # Key differences
        print("\n  ┌───────────────────────────────────────────────────────────────────────────┐")
        print("  │                         KEY DIFFERENCES                                   │")
        print("  └───────────────────────────────────────────────────────────────────────────┘")
        
        scaling_return = s_portfolio.get('total_return', 0)
        fractional_return = f_portfolio.get('total_return', 0)
        return_diff = fractional_return - scaling_return
        
        scaling_final = s_portfolio.get('total_end', 0)
        fractional_final = f_portfolio.get('final_value', 0)
        final_diff = fractional_final - scaling_final
        
        print(f"\n    Return Difference:  {return_diff:+.2f}% (Fractional vs Scaling)")
        print(f"    Value Difference:   ${final_diff:+,.2f}")
        print(f"    Winner:             {'Fractional' if fractional_return > scaling_return else 'Scaling'} "
              f"by {abs(return_diff):.2f}%")
        
        print(f"\n    Why the difference?")
        print(f"    • Fractional: {f_portfolio.get('total_trades', 0)} trades with true position sizing")
        print(f"    • Scaling: {s_portfolio.get('total_trades', 0)} trades with scaled positions")
        print(f"    • Fractional has better analytics: {f_portfolio.get('avg_win_rate', 0):.1f}% win rate, "
              f"{f_portfolio.get('max_drawdown', 0):.2f}% max DD")
    
    # Compare Signal-Based DCA strategies
    print("\n" + "=" * 100)
    print("SIGNAL-BASED DCA STRATEGY COMPARISON")
    print("=" * 100)
    
    scaling_signal = scaling_results.get('Signal-Based DCA', {})
    fractional_signal = fractional_results.get('Signal-Based DCA', {})
    
    if scaling_signal and fractional_signal:
        print("\n┌─────────────────────────────────────────────────────────────────────────────┐")
        print("│                           SCALING APPROACH                                  │")
        print("└─────────────────────────────────────────────────────────────────────────────┘")
        
        s_portfolio = scaling_signal.get('portfolio_summary', {})
        print(f"\n  Portfolio Performance:")
        print(f"    Initial Value:      ${s_portfolio.get('total_start', 0):,.2f}")
        print(f"    Final Value:        ${s_portfolio.get('total_end', 0):,.2f}")
        print(f"    Total Return:       {s_portfolio.get('total_return', 0):+.2f}%")
        print(f"    Total Trades:       {s_portfolio.get('total_trades', 0)}")
        
        print(f"\n  Per-Asset Performance:")
        for symbol_data in scaling_signal.get('results', []):
            symbol = symbol_data.get('symbol', 'UNKNOWN')
            metrics = symbol_data.get('metrics', {})
            print(f"    {symbol:8s}: ${metrics.get('final_equity', 0):>10,.2f} "
                  f"({metrics.get('return_pct', 0):>+7.2f}%) - "
                  f"{metrics.get('num_trades', 0):>4d} trades - "
                  f"Win Rate: {metrics.get('win_rate', 0):>5.1f}%")
        
        print("\n┌─────────────────────────────────────────────────────────────────────────────┐")
        print("│                         FRACTIONAL APPROACH                                 │")
        print("└─────────────────────────────────────────────────────────────────────────────┘")
        
        f_portfolio = fractional_signal.get('portfolio', {})
        print(f"\n  Portfolio Performance:")
        print(f"    Initial Value:      ${f_portfolio.get('initial_value', 0):,.2f}")
        print(f"    Final Value:        ${f_portfolio.get('final_value', 0):,.2f}")
        print(f"    Total Return:       {f_portfolio.get('total_return', 0):+.2f}%")
        print(f"    Total Trades:       {f_portfolio.get('total_trades', 0)}")
        print(f"    Avg Win Rate:       {f_portfolio.get('avg_win_rate', 0):.1f}%")
        print(f"    Avg Sharpe Ratio:   {f_portfolio.get('avg_sharpe', 0):.2f}")
        print(f"    Max Drawdown:       {f_portfolio.get('max_drawdown', 0):.2f}%")
        
        print(f"\n  Per-Asset Performance:")
        for symbol, asset_data in fractional_signal.get('assets', {}).items():
            analytics = asset_data.get('analytics', {})
            print(f"    {symbol:8s}: ${asset_data.get('final_equity', 0):>10,.2f} "
                  f"({asset_data.get('total_return', 0):>+7.2f}%) - "
                  f"{asset_data.get('total_trades', 0):>4d} trades - "
                  f"Win Rate: {analytics.get('win_rate', 0):>5.1f}% - "
                  f"Sharpe: {analytics.get('sharpe_ratio', 0):>5.2f}")
        
        # Key differences
        print("\n  ┌───────────────────────────────────────────────────────────────────────────┐")
        print("  │                         KEY DIFFERENCES                                   │")
        print("  └───────────────────────────────────────────────────────────────────────────┘")
        
        scaling_return = s_portfolio.get('total_return', 0)
        fractional_return = f_portfolio.get('total_return', 0)
        return_diff = fractional_return - scaling_return
        
        scaling_final = s_portfolio.get('total_end', 0)
        fractional_final = f_portfolio.get('final_value', 0)
        final_diff = fractional_final - scaling_final
        
        print(f"\n    Return Difference:  {return_diff:+.2f}% (Fractional vs Scaling)")
        print(f"    Value Difference:   ${final_diff:+,.2f}")
        print(f"    Winner:             {'Fractional' if fractional_return > scaling_return else 'Scaling'} "
              f"by {abs(return_diff):.2f}%")
        
        print(f"\n    Why the difference?")
        print(f"    • Fractional: {f_portfolio.get('total_trades', 0)} trades with true position sizing")
        print(f"    • Scaling: {s_portfolio.get('total_trades', 0)} trades with scaled positions")
        print(f"    • Fractional has exceptional win rate: {f_portfolio.get('avg_win_rate', 0):.1f}%")
        print(f"    • Fractional has minimal drawdown: {f_portfolio.get('max_drawdown', 0):.2f}%")
    
    # Overall strategy comparison
    print("\n" + "=" * 100)
    print("STRATEGY WINNER: MONTHLY DCA vs SIGNAL-BASED DCA")
    print("=" * 100)
    
    if fractional_monthly and fractional_signal:
        monthly_portfolio = fractional_monthly.get('portfolio', {})
        signal_portfolio = fractional_signal.get('portfolio', {})
        
        print("\nUsing Fractional Approach (more realistic):\n")
        
        print("┌────────────────────────────────────────────────────────────────────┐")
        print("│                         MONTHLY DCA                                │")
        print("└────────────────────────────────────────────────────────────────────┘")
        print(f"  Final Value:        ${monthly_portfolio.get('final_value', 0):,.2f}")
        print(f"  Total Return:       {monthly_portfolio.get('total_return', 0):+.2f}%")
        print(f"  Total Trades:       {monthly_portfolio.get('total_trades', 0)}")
        print(f"  Avg Win Rate:       {monthly_portfolio.get('avg_win_rate', 0):.1f}%")
        print(f"  Max Drawdown:       {monthly_portfolio.get('max_drawdown', 0):.2f}%")
        print(f"  Avg Sharpe:         {monthly_portfolio.get('avg_sharpe', 0):.2f}")
        
        print("\n┌────────────────────────────────────────────────────────────────────┐")
        print("│                      SIGNAL-BASED DCA                              │")
        print("└────────────────────────────────────────────────────────────────────┘")
        print(f"  Final Value:        ${signal_portfolio.get('final_value', 0):,.2f}")
        print(f"  Total Return:       {signal_portfolio.get('total_return', 0):+.2f}%")
        print(f"  Total Trades:       {signal_portfolio.get('total_trades', 0)}")
        print(f"  Avg Win Rate:       {signal_portfolio.get('avg_win_rate', 0):.1f}%")
        print(f"  Max Drawdown:       {signal_portfolio.get('max_drawdown', 0):.2f}%")
        print(f"  Avg Sharpe:         {signal_portfolio.get('avg_sharpe', 0):.2f}")
        
        monthly_return = monthly_portfolio.get('total_return', 0)
        signal_return = signal_portfolio.get('total_return', 0)
        monthly_final = monthly_portfolio.get('final_value', 0)
        signal_final = signal_portfolio.get('final_value', 0)
        monthly_dd = monthly_portfolio.get('max_drawdown', 0)
        signal_dd = signal_portfolio.get('max_drawdown', 0)
        monthly_wr = monthly_portfolio.get('avg_win_rate', 0)
        signal_wr = signal_portfolio.get('avg_win_rate', 0)
        
        print("\n" + "=" * 100)
        print("FINAL VERDICT")
        print("=" * 100)
        
        if monthly_return > signal_return:
            winner = "MONTHLY DCA"
            return_advantage = monthly_return - signal_return
            value_advantage = monthly_final - signal_final
        else:
            winner = "SIGNAL-BASED DCA"
            return_advantage = signal_return - monthly_return
            value_advantage = signal_final - monthly_final
        
        print(f"\n🏆 ABSOLUTE RETURN WINNER: {winner}")
        print(f"   • Return advantage: +{return_advantage:.2f}%")
        print(f"   • Value advantage: ${value_advantage:+,.2f}")
        
        # Risk-adjusted analysis
        monthly_risk_adj = monthly_return / max(monthly_dd, 1)
        signal_risk_adj = signal_return / max(signal_dd, 1)
        
        risk_winner = "MONTHLY DCA" if monthly_risk_adj > signal_risk_adj else "SIGNAL-BASED DCA"
        
        print(f"\n🛡️  RISK-ADJUSTED WINNER: {risk_winner}")
        print(f"   • Monthly DCA: {monthly_return:.2f}% return / {monthly_dd:.2f}% DD = {monthly_risk_adj:.2f}")
        print(f"   • Signal DCA: {signal_return:.2f}% return / {signal_dd:.2f}% DD = {signal_risk_adj:.2f}")
        
        print(f"\n📊 TRADE EFFICIENCY:")
        print(f"   • Monthly DCA: {monthly_portfolio.get('total_trades', 0)} trades for {monthly_return:.2f}% return")
        print(f"   • Signal DCA: {signal_portfolio.get('total_trades', 0)} trades for {signal_return:.2f}% return")
        print(f"   • Winner: {'SIGNAL-BASED' if signal_portfolio.get('total_trades', 0) < monthly_portfolio.get('total_trades', 0) else 'MONTHLY'} (fewer trades)")
        
        print(f"\n🎯 WIN RATE COMPARISON:")
        print(f"   • Monthly DCA: {monthly_wr:.1f}%")
        print(f"   • Signal DCA: {signal_wr:.1f}%")
        print(f"   • Advantage: {abs(signal_wr - monthly_wr):.1f}% to {'SIGNAL' if signal_wr > monthly_wr else 'MONTHLY'}")
        
        print("\n" + "=" * 100)
        print("RECOMMENDATION")
        print("=" * 100)
        
        if winner == "MONTHLY DCA":
            print("\n✅ Choose MONTHLY DCA if you want:")
            print("   • Maximum absolute returns")
            print("   • More frequent trading opportunities")
            print("   • Simpler, more predictable strategy")
            
            print("\n⚠️  But consider SIGNAL-BASED DCA if you prioritize:")
            print("   • Lower risk (much smaller drawdowns)")
            print(f"   • Higher win rate ({signal_wr:.1f}% vs {monthly_wr:.1f}%)")
            print("   • Trading only on strong signals")
        else:
            print("\n✅ Choose SIGNAL-BASED DCA if you want:")
            print("   • Maximum absolute returns")
            print("   • Extremely low drawdowns")
            print(f"   • Exceptional win rate ({signal_wr:.1f}%)")
            print("   • Risk-adjusted performance")
            
            print("\n⚠️  But consider MONTHLY DCA if you prefer:")
            print("   • More frequent trading")
            print("   • Simpler strategy logic")
            print("   • Consistent monthly buying")
    
    print("\n" + "=" * 100)
    print("END OF COMPARISON")
    print("=" * 100)


if __name__ == "__main__":
    print_comparison()
