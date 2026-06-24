"""TradingAgents — Autonomous trading execution layer.

Pulls scored signals from the edge scanner, validates them against current
market conditions, executes simulated trades, and tracks open positions.
"""

from .agent import TradingAgent
from .position_tracker import PositionTracker
from .signal_validator import SignalValidator

__all__ = [
    "TradingAgent",
    "PositionTracker",
    "SignalValidator",
]