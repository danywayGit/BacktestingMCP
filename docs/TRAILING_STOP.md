# Trailing-Stop System — Architecture & Lifecycle (Sep 2026)

## Purpose
A trailing stop-loss engine that protects open positions across **all 4 exchanges**
(Binance, Bybit, Velotrade/DXtrade, Bitfunded) by ratcheting each position's
stop-loss forward as price moves in our favor, locking profit past break-even.

## Architecture (2-part — brains on Hermes, hands on the bot)
```
HERMES (this box)                      EXECUTION BOT (109.123.229.200)
────────────────────                  ─────────────────────────────────
trailing_stop.py (cron, every 3 min)   handler.py  Action=MoveStopLoss
  reads open positions via dashboard   holds ALL exchange API secrets
  API (/api/bybit_risk, /api/positions)
  computes R, decides when to trail    receives MoveStopLoss webhook,
  POSTs MoveStopLoss webhook to bot    moves SL on exchange via adapter
  -------------------------------------------------------------------
  DECISION (reads, computes)           EXECUTION (holds secrets, moves SL)
  NO secrets on Hermes.                NO decisions on bot.
```
**Why:** exchange API secrets live only on the bot. Hermes never touches them.
No cross-account hedge risk, no duplicated secrets.

## The trail rule (R-anchored, per exchange, user-confirmed)
- `R = (price − entry) / (entry − SL₀)` for LONG (inverted for SHORT).
- **Until R ≥ +1.0**: SL stays at original (full risk, no trail).
- **At R ≥ +1.0 → prompt R ≥ +1.0**: move SL to at least **+0.5R** past entry (lock a little profit just past break-even).
- **Then trail**: keep SL **0.5R behind the peak price**, only ever **forward** (never backward).
- Parameters in `config` of `trailing_stop.py`:
  `BE_TRIGGER_R = 1.0`, `LOCK_R = 0.5`, `TRAIL_BUFFER_R = 0.5`.

## Per-exchange scan targets
| Exchange | AccountType | Position source | Username / user_id |
|---|---|---|---|
| Bybit | Demo | `/api/bybit_risk` | Danyway_HyroTrader / 43 |
| Binance | TestNet | `/api/positions` | Danyway / 1 |
| Velotrade | Standard | `/api/positions` | Danyway_Velotrade / 45 |
| Bitfunded | Standard | `/api/bitfunded_risk` | Danyway_Bitfunded / 44 |

## Files
| Path | Role |
|---|---|
| `~/.hermes/scripts/trailing_stop.py` | Engine (decision + archive) |
| `~/.hermes/scripts/trailing_stop.sh` | Cron wrapper (rotating log) |
| `~/.hermes/scripts/trailing_stats.py` | Read-only stats query helper |
| `~/.hermes/scripts/.trailing_stop_state.json` | Live/transient tracking state |
| `~/.hermes/logs/trailing_stop.log` | Engine log |
| `/opt/Trading-WebHook-Bot/handler.py` (`MoveStopLoss`) | Bot-side SL execution |

## Cron
- **Trailing engine:** every **3 min** on Hermes (`trailing-stop`, `*/3 * * * *`).
- Runs `trailing_stop.sh`, quiet unless an SL move occurs.

## Bot-side `MoveStopLoss` safety (already built)
In `handler.py`:
- Looks up the open position; if position no longer exists → logs `NO_POSITION`, **touches nothing** (graceful when a trade closes between engine cycles).
- **Forward-only guard:** rejects any move that moves the SL backward (`REJECTED_BACKWARD`).
- **Fail-safe order of operations:** places the **new** SL first, verifies it landed, **then** cancels the old one — a position is never left SL-less if the new placement fails (`PLACEMENT_FAILED`).
- Per-exchange working-stop handling (Binance vs DXtrade/Bitfunded).

## Position lifecycle & close handling (robust, not guessing)
1. While a position is open, the engine tracks its **fingerprint**
   `exchange::account_type::symbol_full` (+ best available TradeID).
2. Each 3-min cycle it re-fetches **only open positions** from the bot API.
3. A tracked fingerprint missing from the open set is treated as **candidate close**.
   A **1-cycle grace** is applied (must be absent 2 consecutive runs) so a transient
   API gap never falsely archives a still-open position.
4. On confirmed close, the engine archives **trail metadata only** to
   `trailed_trades` in Hermes `crypto.db`, and removes it from live tracking.
   A re-opened symbol therefore always starts fresh — never reuses a dead trade's peak/entry/SL.
5. **PnL is NOT stored on Hermes.** It is always read **live** from the bot's
   authoritative `trades.db` ledger (`Trades WHERE IsOpen=0`, via `/api/trades`)
   by joining `TradeID`. Single source of truth — stats cannot go stale or be guessed.

## Stats DB (`trailed_trades` in `~/BacktestingMCP/data/crypto.db`)
```sql
CREATE TABLE trailed_trades (
  TradeID        TEXT PRIMARY KEY,   -- join key to bot ledger
  Exchange       TEXT,
  Symbol         TEXT,
  Side           TEXT,
  OriginalSL     REAL,               -- SL at trail start
  FinalSL        REAL,               -- SL at close (last trailed level)
  Entry          REAL,
  TrailStartAt   TEXT,
  TrailEndAt     TEXT,
  created_at     TEXT
);
```
**No PnL column** — joined live from the bot ledger (option A, user decision).

### Query stats
```bash
# Per-exchange wins/losses/win-rate/total PnL (joins bot ledger live)
python3 ~/.hermes/scripts/trailing_stats.py --per-exchange --days 30

# Full detail
python3 ~/.hermes/scripts/trailing_stats.py --days 30
```

### Direct SQL (win rate, total PnL)
```python
import sqlite3
# Hermes-side metadata
c = sqlite3.connect('/home/hermes/BacktestingMCP/data/crypto.db')
# Bot-side authoritative PnL via /api/trades (TradeID -> ProfitLoss), join in code.
```

## Scope boundaries
- **Trail-only ledger:** records only trades the engine saw open & trailed.
  A position opened and closed entirely between two 3-min cycles (fast scalp,
  no SL move ever sent) may never be seen → not recorded. General open/close
  history is owned by the bot's existing **reconcile/sync** crons, not this engine.
- PnL source of truth remains the bot ledger; Hermes never writes PnL.

## Rebuild / off-machine note
All components are files described above. Recreate: (1) `trailed_trades` table
(via `_ensure_table`, auto-run by the engine), (2) the 3-min cron pointing at
`trailing_stop.py`, (3) keep bot-side `MoveStopLoss` in `handler.py`.