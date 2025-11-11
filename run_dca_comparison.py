"""
DCA Strategy Comparison Runner

Simplified script to run either:
1. Fractional approach (realistic, true fractional shares)
2. Scaling approach (1000x capital workaround)
3. Both approaches comparison

Usage:
    python run_dca_comparison.py fractional  # Run fractional only
    python run_dca_comparison.py scaling     # Run scaling only
    python run_dca_comparison.py compare     # Compare both approaches
"""

import sys

def main():
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = 'fractional'  # Default to fractional
    
    print("\n" + "=" * 80)
    print("DCA STRATEGY COMPARISON RUNNER")
    print("=" * 80)
    
    if mode == 'fractional':
        print("\nRunning FRACTIONAL approach (realistic position sizing)...")
        print("=" * 80)
        from fractional_dca_comparison import run_comparison_backtest, save_results
        results = run_comparison_backtest(strategy_type='both')
        save_results(results)
        
    elif mode == 'scaling':
        print("\nRunning SCALING approach (1000x capital workaround)...")
        print("=" * 80)
        from compare_dca_strategies import main as run_scaling
        run_scaling()
        
    elif mode == 'compare':
        print("\nRunning BOTH approaches for comparison...")
        print("=" * 80)
        
        # Run scaling first
        print("\n" + "=" * 80)
        print("STEP 1/2: Running Scaling Approach")
        print("=" * 80)
        from compare_dca_strategies import main as run_scaling
        run_scaling()
        
        # Run fractional
        print("\n" + "=" * 80)
        print("STEP 2/2: Running Fractional Approach")
        print("=" * 80)
        from fractional_dca_comparison import run_comparison_backtest, save_results
        results = run_comparison_backtest(strategy_type='both')
        save_results(results)
        
        # Now compare
        print("\n" + "=" * 80)
        print("COMPARISON: Scaling vs Fractional")
        print("=" * 80)
        from compare_both_approaches import print_comparison
        print_comparison()
        
    else:
        print(f"\nUnknown mode: {mode}")
        print("\nUsage:")
        print("  python run_dca_comparison.py fractional  # Fractional approach (recommended)")
        print("  python run_dca_comparison.py scaling     # Scaling approach")
        print("  python run_dca_comparison.py compare     # Compare both")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("✅ COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
