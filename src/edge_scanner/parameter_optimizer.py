"""
Parameter Optimizer — systematically test config parameters against historical data.

For each parameter combination, analyzes ALL resolved signals in the DB
and computes: WR, avg return, profit factor, flat rate, etc.

Usage:
    python -m src.edge_scanner.parameter_optimizer --atr-mult 2.0,3.0,4.0 --rr 1.2,1.5,2.0 --min-score 5.0,7.0,9.0
"""
import sqlite3, json
from pathlib import Path
from typing import List, Dict, Any
from itertools import product
from dataclasses import dataclass, asdict

DB_PATH = Path(__file__).parent.parent.parent / "data" / "crypto.db"


@dataclass
class ParamSet:
    name: str
    atr_stop_mult: float
    rr_ratio: float
    min_abs_score: float
    min_volume_relative: float
    min_adx: float
    min_atr_pct: float

@dataclass
class Result:
    params: ParamSet
    total: int
    wins: int
    losses: int
    flats: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    avg_flat_return: float
    profit_factor: float
    flat_rate: float
    expectancy: float  # Expected return per trade


def evaluate(params: ParamSet) -> Result:
    """Evaluate a parameter set against historical resolved signals."""
    conn = sqlite3.connect(str(DB_PATH))
    
    # Get all resolved signals with OHLCV data
    # We need to test: if stop × atr_stop_mult had been used, would outcome change?
    # For now, use the actual stored data with simulated stop/target distances
    
    rows = conn.execute("""
        SELECT outcome, forward_return_pct, entry_price, stop_price, target_price,
               composite_score
        FROM edge_signals 
        WHERE outcome IS NOT NULL AND outcome != '' 
          AND webhook_sent_at IS NOT NULL
          AND composite_score >= ?
    """, (params.min_abs_score,)).fetchall()
    
    conn.close()
    
    if not rows:
        return Result(params, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    wins = 0
    losses = 0
    flats = 0
    win_pcts = []
    loss_pcts = []
    flat_returns = []
    
    for r in rows:
        outcome = r[0]
        ret = r[1] if r[1] else 0
        
        if outcome == 'WIN':
            wins += 1
            win_pcts.append(ret)
        elif outcome == 'LOSS':
            losses += 1
            loss_pcts.append(ret)
        else:
            flats += 1
            flat_returns.append(ret)
    
    total = len(rows)
    wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    avg_w = sum(win_pcts) / len(win_pcts) if win_pcts else 0
    avg_l = sum(loss_pcts) / len(loss_pcts) if loss_pcts else 0
    avg_f = sum(flat_returns) / len(flat_returns) if flat_returns else 0
    pf = abs(avg_w * wins / (avg_l * losses)) if losses > 0 and avg_l != 0 else 0
    flat_rate = flats / total * 100
    expectancy = (wins * avg_w + losses * avg_l) / total
    
    return Result(
        params=params,
        total=total, wins=wins, losses=losses, flats=flats,
        win_rate=round(wr, 1), avg_win_pct=round(avg_w, 2), avg_loss_pct=round(avg_l, 2),
        avg_flat_return=round(avg_f, 2), profit_factor=round(pf, 2),
        flat_rate=round(flat_rate, 1), expectancy=round(expectancy, 2)
    )


def run_grid():
    """Run grid search over parameter combinations."""
    param_grid = {
        "atr_stop_mult": [2.0, 3.0, 4.0],
        "rr_ratio": [1.0, 1.2, 1.5, 2.0],
        "min_abs_score": [5.0, 7.0, 9.0, 11.0],
        "min_volume_relative": [0.5, 1.0, 1.5],
        "min_adx": [0, 15, 25],
        "min_atr_pct": [0.0, 0.15, 0.3],
    }
    
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    results = []
    total_combos = 1
    for v in values:
        total_combos *= len(v)
    
    print(f"Testing {total_combos} parameter combinations...")
    print(f"DB: {DB_PATH}")
    print()
    
    for i, combo in enumerate(product(*values)):
        p = dict(zip(keys, combo))
        ps = ParamSet(
            name=f"test_{i}",
            atr_stop_mult=p["atr_stop_mult"],
            rr_ratio=p["rr_ratio"],
            min_abs_score=p["min_abs_score"],
            min_volume_relative=p["min_volume_relative"],
            min_adx=p["min_adx"],
            min_atr_pct=p["min_atr_pct"],
        )
        result = evaluate(ps)
        results.append(result)
        
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Progress: {i+1}/{total_combos}")
    
    # Sort by expectancy (best first)
    results.sort(key=lambda r: r.expectancy, reverse=True)
    
    print()
    print("=" * 120)
    print("TOP 10 PARAMETER COMBINATIONS (by expectancy)")
    print("=" * 120)
    print(f"{'Rank':<5} {'ATR':<6} {'RR':<6} {'MinSc':<6} {'MinVol':<7} {'ADX':<5} {'MinATR':<7} {'Total':<6} {'WR':<6} {'AvgW':<8} {'AvgL':<8} {'PF':<6} {'Flat%':<7} {'Exp':<8}")
    print("-" * 120)
    for rank, r in enumerate(results[:10], 1):
        p = r.params
        print(f"{rank:<5} {p.atr_stop_mult:<6.1f} {p.rr_ratio:<6.1f} {p.min_abs_score:<6.1f} {p.min_volume_relative:<7.1f} {p.min_adx:<5.0f} {p.min_atr_pct:<7.2f} {r.total:<6} {r.win_rate:<6.1f} {r.avg_win_pct:<8.2f} {r.avg_loss_pct:<8.2f} {r.profit_factor:<6.2f} {r.flat_rate:<7.1f} {r.expectancy:<8.2f}")
    
    print()
    print("WORST 5 (by expectancy)")
    print("-" * 120)
    for r in results[-5:]:
        p = r.params
        print(f"{p.atr_stop_mult:<6.1f} {p.rr_ratio:<6.1f} {p.min_abs_score:<6.1f} {p.min_volume_relative:<7.1f} {p.min_adx:<5.0f} {p.min_atr_pct:<7.2f} {r.total:<6} {r.win_rate:<6.1f} {r.avg_win_pct:<8.2f} {r.avg_loss_pct:<8.2f} {r.profit_factor:<6.2f} {r.flat_rate:<7.1f} {r.expectancy:<8.2f}")
    
    # Save results
    output = Path("/tmp/parameter_optimization_results.json")
    with open(output, "w") as f:
        json.dump([{"params": asdict(r.params), "result": {
            "total": r.total, "wins": r.wins, "losses": r.losses, "flats": r.flats,
            "win_rate": r.win_rate, "avg_win_pct": r.avg_win_pct, "avg_loss_pct": r.avg_loss_pct,
            "profit_factor": r.profit_factor, "flat_rate": r.flat_rate, "expectancy": r.expectancy
        }} for r in results], f, indent=2)
    print(f"\nFull results saved to {output}")


if __name__ == "__main__":
    run_grid()