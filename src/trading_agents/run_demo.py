"""Standalone demo for TradingAgents.

Creates the trades table, runs one iteration of process_signals(), and
prints the results.

Usage:
    cd ~/BacktestingMCP
    venv/bin/python -m src.trading_agents.run_demo
"""

import logging
import sys
from typing import Any, Dict

from .agent import TradingAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("run_demo")


def main() -> None:
    print("=" * 60)
    print("  TradingAgents — Standalone Demo")
    print("=" * 60)

    # Instantiate the agent (ensures trades table exists)
    agent = TradingAgent(db_path="data/crypto.db", min_score=7.0)
    print(f"\n✓ TradingAgent initialised (db={agent.db_path}, min_score={agent.min_score})")

    # Run one processing iteration
    print("\n--- Fetching & processing pending signals ---")
    summary: Dict[str, Any] = agent.process_signals()

    print(f"\n  Total signals fetched : {summary['total']}")
    print(f"  Validated             : {summary['validated']}")
    print(f"  Executed              : {summary['executed']}")
    print(f"  Failed                : {summary['failed']}")

    if summary.get("failures"):
        print("\n  Failures:")
        for f in summary["failures"]:
            print(f"    - signal #{f.get('signal_id')}: {f.get('reason')}")

    if summary.get("results"):
        print("\n  Executed trades:")
        for r in summary["results"]:
            print(f"    - trade #{r['trade_id']} @ {r['entry_price']:.2f}")

    # Show open positions
    print("\n--- Open positions ---")
    open_positions = agent.get_open_positions()
    if open_positions:
        for p in open_positions:
            print(f"  #{p['id']:>4}  {p['symbol']:<10}  {p['direction']:<5}  "
                  f"entry={p['entry_price']:>10.2f}  qty={p['quantity']:>8.4f}  "
                  f"at {p['entry_time']}")
    else:
        print("  (no open positions)")

    # Show aggregate stats
    print("\n--- Trade stats ---")
    stats = agent.tracker.get_trade_stats()
    print(f"  Total trades : {stats['total_trades']}")
    print(f"  Win rate     : {stats['win_rate']:.2f}%")
    print(f"  Avg PnL      : {stats['avg_pnl']:.2f}%")
    print(f"  Total PnL    : {stats['total_pnl']:.2f}%")

    print("\n" + "=" * 60)
    print("  Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()