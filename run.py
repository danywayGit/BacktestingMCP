#!/usr/bin/env python3
"""
BacktestingMCP — Interactive Launcher

Run this script to access all features through a simple menu.
If the virtual environment is not active it will re-launch itself
inside the venv automatically.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

ROOT = Path(__file__).parent


# ---------------------------------------------------------------------------
# Venv helpers
# ---------------------------------------------------------------------------

def _in_venv() -> bool:
    return hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )


def _venv_python() -> str:
    if platform.system() == "Windows":
        return str(ROOT / "venv" / "Scripts" / "python.exe")
    return str(ROOT / "venv" / "bin" / "python")


def _ensure_venv():
    """Re-launch inside the venv if we're not already there."""
    if _in_venv():
        return  # already good
    venv_py = _venv_python()
    if not Path(venv_py).exists():
        print("Virtual environment not found.")
        print("Run:  python setup_venv.py   to create it, then try again.")
        sys.exit(1)
    # Re-exec this script with the venv interpreter
    os.execv(venv_py, [venv_py] + sys.argv)


# ---------------------------------------------------------------------------
# CLI wrapper
# ---------------------------------------------------------------------------

def _subprocess_env() -> dict:
    """Return env with PYTHONPATH=project root so all scripts find src/."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + existing if existing else "")
    return env


def _cli(*args):
    """Run a src.cli.main command and wait for it to finish."""
    subprocess.run([sys.executable, "-m", "src.cli.main"] + list(args), cwd=ROOT)


def _run_example(script: str):
    """Run an examples/ script from the project root."""
    subprocess.run(
        [sys.executable, str(ROOT / "examples" / script)],
        cwd=ROOT,
        env=_subprocess_env(),
    )


def _run_tool(script: str):
    """Run a tools/ script from the project root."""
    subprocess.run(
        [sys.executable, str(ROOT / "tools" / script)],
        cwd=ROOT,
        env=_subprocess_env(),
    )


# ---------------------------------------------------------------------------
# Menu actions
# ---------------------------------------------------------------------------

def menu_download():
    print("\n--- Download Market Data ---")
    symbol    = input("Symbol  [BTC/USDT]: ").strip() or "BTC/USDT"
    timeframe = input("Timeframe [1h]: ").strip() or "1h"
    start     = input("Start date [2024-01-01]: ").strip() or "2024-01-01"
    end       = input("End date   [2024-12-31]: ").strip() or "2024-12-31"
    _cli("data", "download",
         "--symbol", symbol,
         "--timeframe", timeframe,
         "--start", start,
         "--end", end)


def menu_list_data():
    print("\n--- Available Data ---")
    _cli("data", "list-data")


def menu_run_backtest():
    print("\n--- Run a Backtest ---")
    print("Available strategies:")
    try:
        sys.path.insert(0, str(ROOT))
        from src.strategies.templates import list_available_strategies
        strats = list_available_strategies()
    except Exception:
        strats = []
    for s in strats:
        print(f"  • {s}")
    if not strats:
        print("  (could not load strategy list — run option 3 to see all)")
    strategy  = input("\nStrategy [emacrossrsistrategy]: ").strip() or "emacrossrsistrategy"
    symbol    = input("Symbol   [BTCUSDT]: ").strip() or "BTCUSDT"
    timeframe = input("Timeframe [4h]: ").strip() or "4h"
    start     = input("Start date [2020-01-01]: ").strip() or "2020-01-01"
    end       = input("End date   [2025-12-31]: ").strip() or "2025-12-31"
    cash      = input("Starting cash [1000000]: ").strip() or "1000000"

    # Optional strategy parameters
    params = _get_strategy_params(strategy)
    params_json = None
    if params:
        print("\nStrategy parameters (press Enter to use defaults):")
        raw_json = input("  Paste JSON  (or press Enter to enter values one by one): ").strip()
        if raw_json:
            params_json = raw_json
        else:
            chosen = {}
            print(f"  {'Parameter':<26} {'Default':>10}  Value")
            print("  " + "-" * 50)
            for name, default in params.items():
                val = input(f"  {name:<26} {str(default):>10}  → ").strip()
                if val:
                    try:
                        chosen[name] = float(val) if "." in val else int(val)
                    except ValueError:
                        chosen[name] = val
            if chosen:
                import json as _json
                params_json = _json.dumps(chosen)
                print(f"  Using: {params_json}")

    print("\nTrading direction:")
    print("  1  both (long + short)")
    print("  2  long only")
    print("  3  short only")
    dir_input = input("Direction [1]: ").strip() or "1"
    direction = {"1": "both", "2": "long", "3": "short"}.get(dir_input, "both")
    print(f"  → {direction}")

    save_input = input("\nSave results to database? [y/N]: ").strip().lower()
    save = save_input == "y"

    args = ["backtest", "run",
            "--strategy", strategy,
            "--symbol", symbol,
            "--timeframe", timeframe,
            "--start", start,
            "--end", end,
            "--cash", cash,
            "--direction", direction]
    if params_json:
        args += ["--parameters", params_json]
    if save:
        args += ["--save"]
    _cli(*args)


def menu_list_strategies():
    print("\n--- Available Strategies ---")
    _cli("strategy", "list-strategies")


def _get_strategy_params(strategy_name: str) -> dict:
    """Return {param_name: default_value} for numeric class-level params."""
    try:
        sys.path.insert(0, str(ROOT))
        from src.strategies.templates import STRATEGY_REGISTRY
        import backtesting
        cls = STRATEGY_REGISTRY.get(strategy_name.lower())
        if cls is None:
            return {}
        bt_attrs = set(dir(backtesting.Strategy))
        params = {}
        for klass in reversed(cls.__mro__):
            if klass is object:
                continue
            for name, val in vars(klass).items():
                if name.startswith('_') or name in bt_attrs or callable(val):
                    continue
                if isinstance(val, (int, float)):
                    params[name] = val
        return params
    except Exception:
        return {}


def _count_combinations(param_grid: dict) -> int:
    n = 1
    for v in param_grid.values():
        n *= len(v)
    return n


def _to_ccxt_symbol(symbol: str) -> str:
    """Convert BTCUSDT → BTC/USDT for CCXT / the download command."""
    if '/' in symbol:
        return symbol
    for quote in ('USDT', 'BUSD', 'USDC', 'BTC', 'ETH', 'BNB'):
        if symbol.endswith(quote):
            return symbol[:-len(quote)] + '/' + quote
    return symbol[:3] + '/' + symbol[3:]  # fallback


def _check_and_ensure_data(symbol: str, timeframe: str, start: str, end: str) -> bool:
    """
    Check whether the DB contains data for symbol/timeframe over the requested
    period.  Inform the user and optionally trigger a download.
    Returns False only if the user explicitly chooses to abort.
    """
    try:
        sys.path.insert(0, str(ROOT))
        from src.data.database import db
        from datetime import datetime, timezone, timedelta
    except Exception as e:
        print(f"  (could not check database: {e})")
        return True

    db_symbol = symbol.replace('/', '')
    try:
        req_start = datetime.strptime(start, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        req_end   = datetime.strptime(end,   '%Y-%m-%d').replace(tzinfo=timezone.utc)
    except ValueError:
        return True  # bad date — let the CLI report it

    print()
    print(f"  Checking database for {db_symbol} {timeframe} ...")
    try:
        db_start, db_end = db.get_available_data_range(db_symbol, timeframe)
    except Exception as e:
        print(f"  (database check failed: {e})")
        return True

    TOLERANCE = timedelta(days=7)

    if db_start is None:
        print(f"  ✗  No data found for {db_symbol} {timeframe} in the database.")
    else:
        start_ok = db_start <= req_start + TOLERANCE
        end_ok   = db_end   >= req_end   - TOLERANCE
        if start_ok and end_ok:
            print(f"  ✓  Data OK: {db_start.strftime('%Y-%m-%d')} → {db_end.strftime('%Y-%m-%d')}")
            return True
        parts = []
        if not start_ok:
            days = (req_start - db_start).days
            parts.append(f"starts {abs(days)}d late ({db_start.strftime('%Y-%m-%d')})")
        if not end_ok:
            days = (req_end - db_end).days
            parts.append(f"ends {days}d early ({db_end.strftime('%Y-%m-%d')})")
        print(f"  ⚠  Partial data — {', '.join(parts)}.")
        print(f"     Requested : {start} → {end}")

    print()
    choice = input("  [d] Download missing data   [c] Continue anyway   [q] Quit: ").strip().lower()
    if choice == 'q':
        print("  Cancelled.")
        return False
    if choice == 'd':
        ccxt_sym = _to_ccxt_symbol(symbol)
        print(f"\n  Downloading {ccxt_sym} {timeframe}  {start} → {end} ...\n")
        _cli("data", "download",
             "--symbol", ccxt_sym,
             "--timeframe", timeframe,
             "--start", start,
             "--end", end)
        print()
    # 'c' or anything else → continue without downloading
    return True


def menu_optimize_strategy():
    print("\n--- Optimize a Strategy ---")
    print("Builds the parameter grid interactively and calls: backtest optimize")
    print()

    print("Available strategies:")
    try:
        sys.path.insert(0, str(ROOT))
        from src.strategies.templates import list_available_strategies
        strats = list_available_strategies()
    except Exception:
        strats = []
    for s in strats:
        print(f"  \u2022 {s}")
    if not strats:
        print("  (could not load strategy list)")

    strategy   = input("\nStrategy  [emacrossrsistrategy]: ").strip() or "emacrossrsistrategy"
    symbol     = input("Symbol    [BTCUSDT]: ").strip() or "BTCUSDT"
    timeframe  = input("Timeframe [4h]: ").strip() or "4h"
    start      = input("Start date [2020-01-01]: ").strip() or "2020-01-01"
    end        = input("End date   [2026-01-01]: ").strip() or "2026-01-01"
    if not _check_and_ensure_data(symbol, timeframe, start, end):
        return
    cash       = input("Starting cash [1000000]: ").strip() or "1000000"
    commission = input("Commission    [0.001]: ").strip() or "0.001"
    top_n      = input("Show top N results [10]: ").strip() or "10"

    objectives = [
        "sqn", "sharpe_ratio", "sortino_ratio", "calmar_ratio",
        "profit_factor", "total_return_pct", "win_rate_pct", "max_drawdown_pct",
    ]
    print("\nOptimization objective:")
    for i, obj in enumerate(objectives, 1):
        print(f"  {i}  {obj}")
    obj_input = input("Objective [1 = sqn]: ").strip() or "1"
    try:
        objective = objectives[int(obj_input) - 1]
    except (ValueError, IndexError):
        objective = obj_input if obj_input in objectives else "sqn"
    print(f"  \u2192 {objective}")

    # Parameter grid builder
    params = _get_strategy_params(strategy)
    param_grid = {}

    if params:
        print(f"\nParameters for '{strategy}':")
        print("  Press Enter to skip a param, or enter comma-separated values to test.")
        print("  Example:  8,13,21   or   0.5,1.0,1.5")
        print()
        print(f"  {'Parameter':<26} {'Default':>10}  Values to test")
        print("  " + "-" * 62)
        for name, default in params.items():
            val_input = input(f"  {name:<26} {str(default):>10}  \u2192 ").strip()
            if val_input:
                try:
                    values = [float(v.strip()) if '.' in v else int(v.strip())
                              for v in val_input.split(',')]
                    param_grid[name] = values
                except ValueError:
                    print(f"    (skipped \u2014 invalid values for '{name}')")
    else:
        print("\nCould not introspect strategy parameters.")
        raw = input('Enter --param-grid as JSON (e.g. {"ema_fast":[8,13,21]}): ').strip()
        if raw:
            try:
                import json as _json
                param_grid = _json.loads(raw)
            except Exception:
                print("  (invalid JSON \u2014 skipped)")

    if not param_grid:
        print("\n  No parameters selected \u2014 nothing to optimize.")
        return

    import json
    grid_json = json.dumps(param_grid)
    n_combos = _count_combinations(param_grid)
    print(f"\n  Param grid  : {grid_json}")
    print(f"  Objective   : {objective}")
    print(f"\n  Combinations: {n_combos:,}")

    max_tries_arg = None
    if n_combos > 50_000:
        print(f"\n  WARNING: {n_combos:,} combinations is very large and will take a long time.")
        print("  Tip: Use random sampling to test a manageable subset first.")
        print("  Recommended: 500\u20132,000 for a quick run, 5,000\u201310,000 for a thorough run.")
        sample = input(f"  Sample how many? (Enter = all {n_combos:,}, or type e.g. 2000): ").strip()
        if sample.isdigit() and int(sample) > 0:
            max_tries_arg = int(sample)
            print(f"  \u2192 Will randomly sample {max_tries_arg:,} combinations.")
    elif n_combos > 5_000:
        print(f"  Note: {n_combos:,} combinations may take several minutes.")
        sample = input("  Sample a subset? (Enter = all, or type e.g. 2000): ").strip()
        if sample.isdigit() and int(sample) > 0:
            max_tries_arg = int(sample)
            print(f"  \u2192 Will randomly sample {max_tries_arg:,} combinations.")

    confirm = input("\nRun optimization? [Y/n]: ").strip().lower()
    if confirm == 'n':
        print("  Cancelled.")
        return

    args = ["backtest", "optimize",
            "--strategy", strategy,
            "--symbol", symbol,
            "--timeframe", timeframe,
            "--start", start,
            "--end", end,
            "--objective", objective,
            "--param-grid", grid_json,
            "--cash", cash,
            "--commission", commission,
            "--top-n", top_n]
    if max_tries_arg:
        args += ["--max-tries", str(max_tries_arg)]
    _cli(*args)


def menu_walk_forward():
    print("\n--- Check for Overfitting (Walk-Forward) ---")
    print("Splits the period into train/test, runs both, and compares results.")
    print("Tip: paste the best parameters found by option 5 (optimization).")
    print()

    print("Available strategies:")
    try:
        sys.path.insert(0, str(ROOT))
        from src.strategies.templates import list_available_strategies
        strats = list_available_strategies()
    except Exception:
        strats = []
    for s in strats:
        print(f"  \u2022 {s}")

    strategy    = input("\nStrategy  [emacrossrsistrategy]: ").strip() or "emacrossrsistrategy"
    symbol      = input("Symbol    [BTCUSDT]: ").strip() or "BTCUSDT"
    timeframe   = input("Timeframe [4h]: ").strip() or "4h"
    start       = input("Start date [2020-01-01]: ").strip() or "2020-01-01"
    end         = input("End date   [2025-01-01]: ").strip() or "2025-01-01"
    if not _check_and_ensure_data(symbol, timeframe, start, end):
        return
    train_ratio = input("Train ratio [0.7]  (0.7 = 70% train / 30% test): ").strip() or "0.7"
    cash        = input("Starting cash [1000000]: ").strip() or "1000000"
    commission  = input("Commission    [0.001]: ").strip() or "0.001"

    print("\nStrategy parameters from optimization (press Enter to use defaults):")
    print("  Tip: copy the JSON line printed at the end of option 5, OR enter values below.")
    print()
    raw_json = input("  Paste JSON (or press Enter to enter params one by one): ").strip()

    params_json = None
    if raw_json:
        params_json = raw_json
    else:
        params = _get_strategy_params(strategy)
        param_grid = {}
        if params:
            print(f"\n  {'Parameter':<26} {'Default':>10}  Value to use")
            print("  " + "-" * 56)
            for name, default in params.items():
                val_input = input(f"  {name:<26} {str(default):>10}  → ").strip()
                if val_input:
                    try:
                        param_grid[name] = float(val_input) if '.' in val_input else int(val_input)
                    except ValueError:
                        param_grid[name] = val_input
        if param_grid:
            import json as _json
            params_json = _json.dumps(param_grid)
            print(f"\n  Using: {params_json}")

    args = ["backtest", "walk-forward",
            "--strategy", strategy,
            "--symbol", symbol,
            "--timeframe", timeframe,
            "--start", start,
            "--end", end,
            "--train-ratio", train_ratio,
            "--cash", cash,
            "--commission", commission]

    if params_json:
        args += ["--parameters", params_json]

    _cli(*args)


def menu_ai_strategy():
    print("\n--- Generate AI Strategy ---")
    name        = input("Strategy name  [MyStrategy]: ").strip() or "MyStrategy"
    description = input("Describe the strategy: ").strip()
    if not description:
        description = "Buy when RSI drops below 30 and price is above the 200-day MA. Sell when RSI exceeds 70."
    provider    = input("Provider [openai/anthropic/ollama] [openai]: ").strip() or "openai"
    _cli("strategy", "create",
         "--name", name,
         "--description", description,
         "--provider", provider,
         "--register")


def menu_mcp_server():
    print("\n--- Start MCP Server ---")
    print("Starting MCP server on localhost:8000  (Ctrl+C to stop)")
    subprocess.run([sys.executable, "-m", "src.mcp.server"])


def menu_check_gpu():
    print()
    _run_tool("check_gpu.py")


def menu_check_db():
    print()
    _run_tool("check_db.py")


def menu_tutorial():
    print()
    _run_example("00_tutorial.py")


# ---------------------------------------------------------------------------
# Main menu loop
# ---------------------------------------------------------------------------

MENU = [
    ("1", "Download market data",                 menu_download),
    ("2", "List available data",                  menu_list_data),
    ("3", "List strategies",                      menu_list_strategies),
    ("4", "Run a backtest",                       menu_run_backtest),
    ("5", "Optimize a strategy",                  menu_optimize_strategy),
    ("6", "Check for overfitting (walk-forward)", menu_walk_forward),
    ("7", "Generate AI strategy",                 menu_ai_strategy),
    ("8", "Start MCP server",                     menu_mcp_server),
    ("9", "Check GPU status",                     menu_check_gpu),
    ("0", "Inspect database",                     menu_check_db),
    ("t", "Run tutorial walkthrough",             menu_tutorial),
    ("q", "Quit",                                 None),
]


def print_menu():
    print("\n" + "=" * 50)
    print("  BacktestingMCP")
    print("=" * 50)
    for key, label, _ in MENU:
        print(f"  {key}  {label}")
    print("=" * 50)


def main():
    _ensure_venv()
    while True:
        print_menu()
        choice = input("  Choose: ").strip().lower()
        for key, _, action in MENU:
            if choice == key:
                if action is None:
                    print("Goodbye.")
                    sys.exit(0)
                action()
                break
        else:
            print(f"  Unknown option '{choice}' — try again.")


if __name__ == "__main__":
    main()

