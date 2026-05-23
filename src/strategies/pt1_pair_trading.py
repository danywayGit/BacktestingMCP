"""
PT1 — BTC/ETH Pair Trading (Z-score Mean Reversion)
=====================================================
Timeframe:  1H
Direction:  Long and Short (BTC as single-leg proxy)

Strategy overview:
  Compute a rolling Z-score of the BTC/ETH price ratio.
  - When Z < -z_entry  → BTC cheap vs ETH → BUY BTC (long BTC / short ETH leg)
  - When Z > +z_entry  → BTC expensive     → SELL BTC (short BTC / long ETH leg)

  Because backtesting.py is single-position, only the BTC leg is simulated.
  Sizing uses notional = equity × allocation_pct / 100 (one-leg half-allocation).

Correlation filter:
  Entry blocked if rolling(lookback) Pearson correlation between BTC and ETH < min_correlation.

Exit conditions (checked each bar while in position):
  1. abs(z) < z_exit          → mean reversion complete
  2. Long:  z >  z_max        → Z diverged further against us (stop)
     Short: z < -z_max        → Z diverged further against us (stop)
  3. Combined drawdown >  max_dd_pct from entry equity
  4. bars_held >= max_hold_bars
"""

import numpy as np
import pandas as pd

from ..core.backtesting_engine import BaseStrategy
from ..data.database import db

# ---------------------------------------------------------------------------
# talib / ta dual-fallback (not used directly but kept for convention)
# ---------------------------------------------------------------------------
try:
    import talib  # noqa: F401
except ImportError:
    try:
        import ta  # noqa: F401
    except ImportError:
        pass


class PT1PairTradingStrategy(BaseStrategy):
    """
    PT1 — BTC/ETH Pair Trading via Z-score of price ratio.

    Runs on BTCUSDT 1H; ETH close is loaded from DB and aligned to BTC index.
    """

    # === Parameters ===
    lookback         = 50
    z_entry          = 2.0
    z_exit           = 0.5
    z_max            = 3.0
    allocation_pct   = 10.0
    max_dd_pct       = 5.0
    min_correlation  = 0.7
    min_spread_vol   = 0.03
    max_hold_bars    = 200
    eth_symbol       = 'ETHUSDT'
    risk_pct         = 1.0          # kept for BaseStrategy compatibility

    # ------------------------------------------------------------------
    def init(self):
        # Load ETH close from DB and align to BTC index
        start_dt = self.data.index[0].to_pydatetime()
        end_dt   = self.data.index[-1].to_pydatetime()

        eth_data = db.get_market_data(self.eth_symbol, '1h', start_dt, end_dt)

        if eth_data is None or eth_data.empty:
            eth_close_arr = np.full(len(self.data.Close), np.nan)
        else:
            eth_close_col = (
                eth_data['close'] if 'close' in eth_data.columns else eth_data['Close']
            )
            eth_close_aligned = eth_close_col.reindex(self.data.index).ffill().bfill()
            eth_close_arr = eth_close_aligned.values.astype(float)

        # Register as indicator so backtesting.py slices it with the data window
        self.eth_close = self.I(lambda x: x, eth_close_arr, name='ETH_Close')

        # Per-trade state
        self._bars_held      = 0
        self._entry_is_long  = None   # True = long BTC, False = short BTC
        self._entry_equity   = None
        self._bars_since_exit = 9999

    # ------------------------------------------------------------------
    def _compute_zscore(self):
        """
        Compute z-score from last `lookback` bars of BTC and ETH.

        Returns (z, corr, ratio_std) or (nan, nan, nan) if insufficient data.
        """
        n = self.lookback

        btc_raw = self.data.Close[-n:]
        eth_raw = self.eth_close[-n:]

        btc_arr = np.array(btc_raw, dtype=float)
        eth_arr = np.array(eth_raw, dtype=float)

        if len(btc_arr) < n or len(eth_arr) < n:
            return np.nan, np.nan, np.nan

        if np.any(~np.isfinite(eth_arr)) or np.any(~np.isfinite(btc_arr)):
            return np.nan, np.nan, np.nan

        # Ratio series
        denom = np.where(eth_arr > 0, eth_arr, np.nan)
        ratio_arr = btc_arr / denom

        if np.any(~np.isfinite(ratio_arr)):
            return np.nan, np.nan, np.nan

        ratio_mean = np.nanmean(ratio_arr)
        ratio_std  = np.nanstd(ratio_arr)

        if ratio_std <= self.min_spread_vol:
            return np.nan, np.nan, ratio_std

        z = (ratio_arr[-1] - ratio_mean) / ratio_std

        # Rolling correlation
        corr = np.corrcoef(btc_arr, eth_arr)[0, 1]

        return z, corr, ratio_std

    # ------------------------------------------------------------------
    def next(self):
        if not self.should_trade():
            return

        close = float(self.data.Close[-1])
        if not np.isfinite(close) or close <= 0:
            return

        z, corr, ratio_std = self._compute_zscore()

        # ── In-position management ────────────────────────────────────
        if self.position:
            self._bars_held += 1
            is_long = self._entry_is_long

            exit_reason = None

            # 1. Mean reversion complete
            if np.isfinite(z) and abs(z) < self.z_exit:
                exit_reason = "mean_reversion"

            # 2. Z diverged further against us
            if exit_reason is None and np.isfinite(z):
                if is_long  and z >  self.z_max:
                    exit_reason = "z_diverged_long"
                elif not is_long and z < -self.z_max:
                    exit_reason = "z_diverged_short"

            # 3. Drawdown stop
            if exit_reason is None and self._entry_equity is not None and self._entry_equity > 0:
                dd = (self.equity - self._entry_equity) / self._entry_equity
                if dd < -(self.max_dd_pct / 100.0):
                    exit_reason = "drawdown_stop"

            # 4. Time exit
            if exit_reason is None and self._bars_held >= self.max_hold_bars:
                exit_reason = "max_hold"

            if exit_reason:
                self.position.close()
                self._bars_held       = 0
                self._entry_is_long   = None
                self._entry_equity    = None
                self._bars_since_exit = 0
            return

        # ── No position ───────────────────────────────────────────────
        self._bars_since_exit += 1

        # Need valid z-score and correlation
        if not np.isfinite(z) or not np.isfinite(corr):
            return

        # Correlation filter
        if corr < self.min_correlation:
            return

        # Spread vol already checked inside _compute_zscore (returns nan if below threshold)

        # Position sizing — one leg, half allocation
        notional = self.equity * (self.allocation_pct / 100.0)
        qty = notional / close
        if qty <= 0:
            return

        # ── Long BTC (Z too low → BTC cheap) ─────────────────────────
        if z < -self.z_entry:
            self.buy(size=qty)
            self._entry_is_long  = True
            self._entry_equity   = self.equity
            self._bars_held      = 0

        # ── Short BTC (Z too high → BTC expensive) ───────────────────
        elif z > self.z_entry:
            self.sell(size=qty)
            self._entry_is_long  = False
            self._entry_equity   = self.equity
            self._bars_held      = 0
