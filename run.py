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
    print("Available strategies (run option 3 to list all):")
    strats = [
        "moving_average_crossover", "rsi_mean_reversion",
        "ema_rsi_combination", "bollinger_band_bounce",
        "atr_breakout", "momentum_oscillator",
    ]
    for s in strats:
        print(f"  • {s}")
    strategy  = input("\nStrategy [moving_average_crossover]: ").strip() or "moving_average_crossover"
    symbol    = input("Symbol   [BTCUSDT]: ").strip() or "BTCUSDT"
    timeframe = input("Timeframe [1h]: ").strip() or "1h"
    start     = input("Start date [2024-01-01]: ").strip() or "2024-01-01"
    end       = input("End date   [2024-12-31]: ").strip() or "2024-12-31"
    cash      = input("Starting cash [10000]: ").strip() or "10000"
    _cli("backtest", "run",
         "--strategy", strategy,
         "--symbol", symbol,
         "--timeframe", timeframe,
         "--start", start,
         "--end", end,
         "--cash", cash)


def menu_list_strategies():
    print("\n--- Available Strategies ---")
    _cli("strategy", "list-strategies")


def menu_gpu_optimization():
    print("\n--- GPU Optimization ---")
    print("1 - DCA optimization  (02_gpu_optimization.py, ~1145 tests/sec)")
    print("2 - EMA Crossover     (03_ema_crossover.py)")
    choice = input("Which? [1]: ").strip() or "1"
    if choice == "2":
        _run_example("03_ema_crossover.py")
    else:
        _run_example("02_gpu_optimization.py")


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
    ("1", "Download market data",         menu_download),
    ("2", "List available data",           menu_list_data),
    ("3", "List strategies",               menu_list_strategies),
    ("4", "Run a backtest",                menu_run_backtest),
    ("5", "GPU optimization",              menu_gpu_optimization),
    ("6", "Generate AI strategy",          menu_ai_strategy),
    ("7", "Start MCP server",              menu_mcp_server),
    ("8", "Check GPU status",              menu_check_gpu),
    ("9", "Inspect database",              menu_check_db),
    ("t", "Run tutorial walkthrough",      menu_tutorial),
    ("q", "Quit",                          None),
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

