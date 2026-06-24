"""trading-agent CLI — Click-based interface for the TradingAgent.

Usage examples:

    python -m src.trading_agents.cli process
    python -m src.trading_agents.cli positions
    python -m src.trading_agents.cli close 1 67500.0
    python -m src.trading_agents.cli stats
    python -m src.trading_agents.cli monitor
"""

import logging
import time
import sys
from typing import Any, Dict

import click

from .agent import TradingAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared options decorator
# ---------------------------------------------------------------------------

def _shared_opts(f):
    """Add --db, --min-score, and --verbose options to a Click command."""
    f = click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")(f)
    f = click.option("--min-score", default=7.0, type=float, help="Minimum |composite_score| threshold", show_default=True)(f)
    f = click.option("--db", default="data/crypto.db", help="Path to the SQLite database", show_default=True)(f)
    return f


def _make_agent(db: str, min_score: float, verbose: bool) -> TradingAgent:
    """Build a TradingAgent and optionally enable verbose logging."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG, stream=sys.stderr, force=True)
    else:
        logging.basicConfig(level=logging.INFO, stream=sys.stderr, force=True)
    return TradingAgent(db_path=db, min_score=min_score)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
def cli():
    """TradingAgents — autonomous signal processing & trade execution."""


# ---------------------------------------------------------------------------
# process
# ---------------------------------------------------------------------------


@cli.command()
@_shared_opts
def process(db: str, min_score: float, verbose: bool):
    """Run one iteration of fetch -> validate -> execute on pending signals."""
    agent = _make_agent(db, min_score, verbose)
    summary: Dict[str, Any] = agent.process_signals()
    click.echo("=== Process Signals Summary ===")
    click.echo(f"  Total signals fetched : {summary['total']}")
    click.echo(f"  Validated             : {summary['validated']}")
    click.echo(f"  Executed              : {summary['executed']}")
    click.echo(f"  Failed                : {summary['failed']}")
    if summary.get("failures"):
        click.echo("\n  Failures:")
        for f in summary["failures"]:
            click.echo(f"    - signal #{f.get('signal_id')}: {f.get('reason')}")


# ---------------------------------------------------------------------------
# positions
# ---------------------------------------------------------------------------


@cli.command()
@_shared_opts
def positions(db: str, min_score: float, verbose: bool):
    """List all currently open positions."""
    agent = _make_agent(db, min_score, verbose)
    open_positions = agent.get_open_positions()
    if not open_positions:
        click.echo("No open positions.")
        return
    click.echo(f"{'ID':>4}  {'Symbol':<10}  {'Dir':<5}  {'Entry':>10}  {'Qty':>8}  {'Time'}")
    click.echo("-" * 65)
    for p in open_positions:
        click.echo(
            f"{p['id']:>4}  {p['symbol']:<10}  {p['direction']:<5}  "
            f"{p['entry_price']:>10.2f}  {p['quantity']:>8.4f}  {p['entry_time']}"
        )


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("trade_id", type=int)
@click.argument("exit_price", type=float)
@_shared_opts
def close(trade_id: int, exit_price: float, db: str, min_score: float, verbose: bool):
    """Close an open position by TRADE_ID at the given EXIT_PRICE."""
    agent = _make_agent(db, min_score, verbose)
    result = agent.close_position(trade_id, exit_price)
    if result.get("success"):
        click.echo(f"Closed trade #{trade_id} @ {exit_price:.2f} — PnL: {result['pnl_pct']:.2f}%")
    else:
        click.echo(f"Failed to close trade #{trade_id} (not found or already closed).")


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


@cli.command()
@_shared_opts
def stats(db: str, min_score: float, verbose: bool):
    """Show aggregate trade performance statistics."""
    agent = _make_agent(db, min_score, verbose)
    s = agent.tracker.get_trade_stats()
    click.echo("=== Trade Statistics ===")
    click.echo(f"  Total trades  : {s['total_trades']}")
    click.echo(f"  Wins          : {s['win_count']}")
    click.echo(f"  Losses        : {s['loss_count']}")
    click.echo(f"  Win rate      : {s['win_rate']:.2f}%")
    click.echo(f"  Avg PnL       : {s['avg_pnl']:.2f}%")
    click.echo(f"  Total PnL     : {s['total_pnl']:.2f}%")
    click.echo(f"  Best trade    : {s['best_trade']:.2f}%")
    click.echo(f"  Worst trade   : {s['worst_trade']:.2f}%")


# ---------------------------------------------------------------------------
# monitor
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--interval", default=60, type=int, help="Seconds between iterations", show_default=True)
@click.option("--iterations", default=5, type=int, help="Number of iterations (0 = infinite)", show_default=True)
@_shared_opts
def monitor(interval: int, iterations: int, db: str, min_score: float, verbose: bool):
    """Run a monitoring loop that checks for new signals every N seconds."""
    agent = _make_agent(db, min_score, verbose)
    iteration = 0
    try:
        while True:
            iteration += 1
            click.echo(f"\n--- Monitor iteration #{iteration} @ {time.strftime('%H:%M:%S')} ---")
            summary = agent.process_signals()
            click.echo(f"  Fetched={summary['total']}  Validated={summary['validated']}  Executed={summary['executed']}  Failed={summary['failed']}")
            if iterations > 0 and iteration >= iterations:
                click.echo("Monitor finished (reached iteration limit).")
                break
            click.echo(f"Sleeping {interval}s...")
            time.sleep(interval)
    except KeyboardInterrupt:
        click.echo("\nMonitor stopped by user.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    cli()  # type: ignore[misc]


if __name__ == "__main__":
    main()