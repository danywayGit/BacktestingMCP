# ----------------------------------------------- #
# Plugin Name           : BacktestingMCP           #
# Author Name           : danywayGIT              #
# File Name             : realized_config_stats.py#
# ----------------------------------------------- #

"""Realized-config-stats loader.

Computes REALIZED per-configuration performance (win rate, count, total PnL)
from the bot's closed trades and stores it in crypto.db `realized_config_stats`.
The edge-scanner tier-sizer (`webhook_bridge._compute_tier_multiplier`) reads
this table so it sizes positions off what the bot ACTUALLY closed, not the
scanner's theoretical forward-return.

This is a BRIDGE-SIDE/LOCAL module. It needs a dump of the bot's closed trades
(see the backtest-vs-execution-comparison skill) joined to edge_signals configs.
The join script `join_bot.py` writes `/home/hermes/bot_joined.json`.

Input: `/home/hermes/bot_joined.json` (array of trades, each with `_cfg`, IsOpen,
OpenDate, ProfitLoss, RiskRewardRatio).
Output: upsert rows into `realized_config_stats` in `data/crypto.db`.
"""

import sqlite3, json, os
from datetime import datetime, timezone

DB_PATH = "/home/hermes/BacktestingMCP/data/crypto.db"
JOINED_PATH = '/home/hermes/bot_joined.json'
RECENCY_DAYS = 30          # "recent real results" sizing window
MIN_TRADES = 10            # trust floor before the sizer boosts/cuts a config


def _utc_now():
    return datetime.now(timezone.utc)


def _load(joined_path=JOINED_PATH):
    with open(joined_path) as f:
        return json.load(f)


def compute(joined=None, recency_days=RECENCY_DAYS):
    """Return {config_version: {wr, n, total_pl, avg_pl, status}} for recent closed trades."""
    joined = joined if joined is not None else _load()
    now = _utc_now()
    from collections import defaultdict
    D = defaultdict(lambda: defaultdict(float))  # cfg -> {'w':0,'l':0,'pl':0.0}
    n_cfg = defaultdict(int)
    for t in joined:
        cfg = t.get('_cfg')
        if cfg in (None, 'None', 'EXIT'): continue
        if str(t.get('TradeID', '')).isdigit() is False: continue
        if str(t.get('IsOpen')) != '0': continue            # closed only
        odate = str(t.get('OpenDate', ''))
        if not odate: continue
        try:
            odt = datetime.fromisoformat(odate.replace(' ', 'T'))
            if odt.tzinfo is None: odt = odt.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if (now - odt.astimezone(timezone.utc)).total_seconds() > recency_days * 86400:
            continue
        try:
            pl = float(t.get('ProfitLoss') or 0)
        except Exception:
            continue
        D[cfg]['w' if pl > 0 else 'l'] += 1
        D[cfg]['pl'] += pl
        n_cfg[cfg] += 1
    out = {}
    for cfg, s in D.items():
        w = int(s['w']); l = int(s['l']); n = w + l
        if n == 0: continue
        out[cfg] = {
            'n': n,
            'wr': w / n,
            'w': w,
            'l': l,
            'total_pl': round(s['pl'], 2),
            'avg_pl': round(s['pl'] / n, 2),
            'status': 'ok',
        }
    return out


def upsert(stats, db_path=DB_PATH):
    """Replace ALL rows for the recency window — it's a full recompute, so delete+insert."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS realized_config_stats (
        config_version TEXT PRIMARY KEY,
        n INTEGER, w INTEGER, l INTEGER,
        wr REAL, total_pl REAL, avg_pl REAL,
        computed_at TEXT
    )""")
    cur.execute("DELETE FROM realized_config_stats")
    nowstr = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    for cfg, s in stats.items():
        cur.execute("""INSERT OR REPLACE INTO realized_config_stats
            (config_version, n, w, l, wr, total_pl, avg_pl, computed_at)
            VALUES (?,?,?,?,?,?,?,?)""",
            (cfg, s['n'], s['w'], s['l'], s['wr'], s['total_pl'],
             s['avg_pl'], nowstr))
    conn.commit()
    conn.close()
    return len(stats)


def load(db_path=DB_PATH):
    conn = sqlite3.connect(db_path); conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM realized_config_stats").fetchall()
    conn.close()
    return {r['config_version']: dict(r) for r in rows}


def main():
    stats = compute()
    upsert(stats)
    print(f"upserted {len(stats)} config versions into realized_config_stats ({RECENCY_DAYS}-day window)")
    for c, s in sorted(stats.items(), key=lambda kv: -kv[1]['avg_pl']):
        print(f"  {c:>6} n={s['n']:>3} WR={s['wr']*100:5.1f}% total=${s['total_pl']:<9.2f} avg=${s['avg_pl']:+.2f}")


if __name__ == '__main__':
    main()