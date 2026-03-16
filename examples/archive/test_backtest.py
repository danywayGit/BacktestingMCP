#!/usr/bin/env python3
"""
Backtest EMA Crossover RSI strategy with GPU-optimized parameters.
Compares backtesting.py results with GPU optimizer results.
"""

from backtesting import Backtest, Strategy
import pandas as pd
import sqlite3
import ta


def load_data(symbol, timeframe='4h'):
    """Load OHLCV data from database."""
    conn = sqlite3.connect('data/crypto.db')
    start_ts = int(pd.Timestamp('2021-01-01').timestamp())
    end_ts = int(pd.Timestamp('2025-12-01').timestamp())
    query = '''SELECT timestamp, open, high, low, close, volume 
               FROM market_data WHERE symbol = ? AND timeframe = ? 
               AND timestamp >= ? AND timestamp <= ? ORDER BY timestamp'''
    df = pd.read_sql_query(query, conn, params=(symbol, timeframe, start_ts, end_ts))
    conn.close()
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('datetime', inplace=True)
    df.columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]


class EMAcrossRSI_Simple(Strategy):
    """
    Simple EMA Crossover with RSI - matches GPU optimizer logic exactly.
    Only uses SL/TP for exits (no signal-based exits).
    """
    
    ema_fast = 13
    ema_slow = 55
    ema_12h = 21
    rsi_period = 21
    rsi_threshold = 50
    atr_period = 21
    atr_mult = 1.0
    rr_ratio = 4.0
    
    def init(self):
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        
        self.ema_f = self.I(lambda: ta.trend.ema_indicator(close, self.ema_fast).ffill().bfill().values)
        self.ema_s = self.I(lambda: ta.trend.ema_indicator(close, self.ema_slow).ffill().bfill().values)
        self.ema_12 = self.I(lambda: ta.trend.ema_indicator(close, self.ema_12h * 3).ffill().bfill().values)
        self.rsi = self.I(lambda: ta.momentum.rsi(close, self.rsi_period).fillna(50).values)
        self.atr = self.I(lambda: ta.volatility.average_true_range(high, low, close, self.atr_period).fillna(0).values)
    
    def next(self):
        if len(self.data) < 200:
            return
        
        # Entry conditions (matching GPU optimizer exactly)
        cross = self.ema_f[-2] < self.ema_s[-2] and self.ema_f[-1] > self.ema_s[-1]
        rsi_ok = self.rsi[-1] > self.rsi_threshold
        trend = self.data.Close[-1] > self.ema_12[-1]
        
        if not self.position and cross and rsi_ok and trend and self.atr[-1] > 0:
            price = self.data.Close[-1]
            sl = price - self.atr_mult * self.atr[-1]
            risk = price - sl
            tp = price + self.rr_ratio * risk
            
            # 1% risk position sizing
            risk_amt = self.equity * 0.01
            size = min(risk_amt / risk * price / self.equity, 0.99)
            
            self.buy(size=size, sl=sl, tp=tp)


def run_all():
    """Run backtests for all 4 pairs with GPU-optimized parameters."""
    
    configs = {
        'BTCUSDT': {'ema_fast': 8, 'ema_slow': 21, 'ema_12h': 13, 'rsi_period': 14, 
                    'rsi_threshold': 40, 'atr_period': 21, 'atr_mult': 1.0, 'rr_ratio': 4.0},
        'ETHUSDT': {'ema_fast': 8, 'ema_slow': 21, 'ema_12h': 21, 'rsi_period': 14, 
                    'rsi_threshold': 55, 'atr_period': 21, 'atr_mult': 1.0, 'rr_ratio': 3.0},
        'SOLUSDT': {'ema_fast': 13, 'ema_slow': 55, 'ema_12h': 21, 'rsi_period': 21, 
                    'rsi_threshold': 50, 'atr_period': 21, 'atr_mult': 1.0, 'rr_ratio': 4.0},
        'BNBUSDT': {'ema_fast': 8, 'ema_slow': 89, 'ema_12h': 34, 'rsi_period': 7, 
                    'rsi_threshold': 55, 'atr_period': 14, 'atr_mult': 1.5, 'rr_ratio': 3.5},
    }
    
    gpu_results = {
        'BTCUSDT': {'return': 41.21, 'dd': 8.0, 'wr': 43.5, 'trades': 209},
        'ETHUSDT': {'return': 46.05, 'dd': 7.8, 'wr': 48.1, 'trades': 154},
        'SOLUSDT': {'return': 80.59, 'dd': 5.1, 'wr': 54.1, 'trades': 98},
        'BNBUSDT': {'return': 47.55, 'dd': 6.8, 'wr': 55.0, 'trades': 100},
    }
    
    print("=" * 80)
    print("BACKTEST RESULTS vs GPU OPTIMIZER")
    print("=" * 80)
    print(f"{'Symbol':<10} {'BT Return':>12} {'GPU Return':>12} {'BT DD':>10} {'GPU DD':>10} {'BT Trades':>10} {'GPU Trades':>10}")
    print("-" * 80)
    
    for symbol, params in configs.items():
        # Create strategy class with params
        class TestStrat(EMAcrossRSI_Simple):
            ema_fast = params['ema_fast']
            ema_slow = params['ema_slow']
            ema_12h = params['ema_12h']
            rsi_period = params['rsi_period']
            rsi_threshold = params['rsi_threshold']
            atr_period = params['atr_period']
            atr_mult = params['atr_mult']
            rr_ratio = params['rr_ratio']
        
        df = load_data(symbol)
        bt = Backtest(df, TestStrat, cash=10_000_000, commission=0.001)
        stats = bt.run()
        
        gpu = gpu_results[symbol]
        bt_ret = stats['Return [%]']
        bt_dd = abs(stats['Max. Drawdown [%]'])
        bt_trades = stats['# Trades']
        
        print(f"{symbol:<10} {bt_ret:>11.2f}% {gpu['return']:>11.2f}% {bt_dd:>9.2f}% {gpu['dd']:>9.2f}% {bt_trades:>10} {gpu['trades']:>10}")
    
    print("=" * 80)


if __name__ == '__main__':
    run_all()
