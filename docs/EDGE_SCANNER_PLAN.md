# Crypto Trading Edge Scanner — Roadmap & Status

Living plan for the composite edge-identification system. Read this first
when resuming work in a new session.

## Goal (and an honest constraint)

Build a system that scans crypto spot/futures markets, scores candidates
across multiple signal sources, and tells you which ones have a *measured*
edge — not a system that guarantees winning more than losing every day.
No legitimate system can promise that; markets are adversarial and any
"always wins daily" claim is either overfit on history or false. What's
achievable and what this is built for: positive expectancy validated by
real forward outcomes (win-rate, avg return, Sharpe-style stats), tracked
per symbol/hour/direction so you can see what's actually working rather
than guessing.

## Architecture

```
 altFINS screener (2000+ symbols)  ─┐
 altFINS signal feed (VIP-channel   ├─► composite score per symbol ─► log signal ─► track forward outcome ─► win-rate report
   equivalent, bullish/bearish)    ─┤        (Phase 1)                  (Phase 2)         (Phase 2)              (Phase 2)
 BacktestingMCP TA scanners        ─┘
   (price/volume breakouts, real OHLCV)

 [not built yet]
 On-chain data (new provider)       ──► additional scoring input        (Phase 3)
 TradingAgents (multi-agent debate) ──► decision synthesis on shortlist (Phase 4)
 Hermes Agent + self-evolution      ──► refines weights/prompts from
                                         real tracked outcomes           (Phase 5)
```

### Why this shape

- **altFINS has no on-chain data** (confirmed by inspecting altfinsMCP — it
  only references a separate whale-tracker product, not a queryable
  metric). Phase 3 needs a different provider.
- **TradingAgents** (github.com/TauricResearch/TradingAgents, Apache-2.0,
  LangGraph) already implements the analyst/debate/risk-manager pattern
  we'd otherwise build from scratch. Supports local Ollama models, so the
  decision layer can run on the RTX 4090 with zero per-call cost.
- **Hermes Agent** (NousResearch, MIT) — `hermes-agent` builds skills from
  experience; `hermes-agent-self-evolution` uses DSPy+GEPA to read
  execution traces and propose targeted improvements. This is the
  practical "self-learning" path the user wants, and it needs *real
  outcomes* to learn from — which is exactly what Phase 2's tracking loop
  produces. Don't build Phase 5 before Phase 2 has weeks of resolved
  signals; there's nothing to learn from otherwise.
- **No bespoke RL.** Classic RL needs a labeled training dataset and a
  GPU training pipeline before it's useful. TradingAgents + Hermes give a
  more practical path to "self-improving" with far less infrastructure.

### Where to run (cost-conscious)

- **Phases 1-3 (scanning/data):** periodic REST/MCP polling every 5-15 min,
  no GPU needed. Runs fine on a NAS as a cron job / lightweight container,
  24/7, near-zero cost. Do **not** build a websocket/tick-level feed for
  this — these are 1h-4h swing signals, polling is sufficient and far
  cheaper.
- **Phase 4 (TradingAgents LLM calls):** only run on the shortlist that
  already passed Phase 1+2 thresholds (typically ~10-20 of 2000 symbols
  per cycle), not on the whole universe — this is what keeps LLM cost
  near zero. Prefer local Ollama on the RTX 4090 if it's online; otherwise
  cheap/batched cloud calls on just the shortlist.
- **Backtesting/optimization (existing BacktestingMCP GPU work):** RTX
  4090, on-demand, not running continuously.

## Status

### Phase 1 — Composite scanner (BUILT)

New code, all in this repo (`BacktestingMCP`):

- `src/integrations/altfins_client.py` — sync wrapper around the altFINS
  MCP server (`https://mcp.altfins.com/mcp`). Three calls:
  `get_screener_data()` (discovers candidates via SHORT_TERM_TREND
  UP/DOWN filters across altFINS' full universe), `get_signal_feed()`
  (the API equivalent of altFINS' VIP Telegram signal channel —
  see note below), `get_recent_news_counts()` (mention counts only, no
  sentiment yet). Requires `ALTFINS_API_KEY` env var (`.env` supported via
  `python-dotenv`, already a dependency).
- `src/edge_scanner/composite.py` — `run_composite_scan()` discovers
  candidates from altFINS, cross-checks each against the signal feed, then
  confirms with BacktestingMCP's own `evaluate_scan()` breakout scanners
  on real OHLCV (via `engine.get_data`, so it doesn't just trust a third
  party). Produces a `CandidateScore` per symbol: composite score,
  LONG/SHORT/None direction, and a `components` breakdown for
  auditability. If `ALTFINS_API_KEY` isn't set, falls back to the static
  `CRYPTO_PAIRS` universe with TA-only scoring.
- `src/data/database.py` — new `edge_signals` table + `insert_edge_signal`,
  `get_pending_edge_signals`, `resolve_edge_signal`,
  `get_resolved_edge_signals` methods, following the existing
  `CryptoDatabase` pattern.

### Phase 2 — Forward-validation loop (BUILT)

- `src/edge_scanner/store.py` — `log_signals()` persists actionable
  signals with entry price; `resolve_due_signals()` checks signals whose
  `horizon_hours` has elapsed, computes the directional forward return,
  and labels WIN/LOSS/FLAT (moves under 0.3% count as noise, not a
  result); `performance_report()` aggregates win-rate/avg-return by
  symbol, hour-of-day, or direction — the same idea BackTestingSignals
  uses for Telegram/Discord call validation, applied to this scanner's
  own output.

### CLI (BUILT)

```
python -m src.cli.main edge scan   --timeframe 1h --lookback-days 30 --per-side 20 --horizon-hours 24
python -m src.cli.main edge track
python -m src.cli.main edge report --group-by symbol   # or hour, direction
```

Intended cron pattern: run `edge scan` every 15-60 min, `edge track` every
hour (resolves anything past its horizon), check `edge report` weekly to
see what's actually working before trusting any of it with real capital.

### Verified so far

This sandbox's network egress goes through a TLS-intercepting proxy that
breaks live calls to Binance/altFINS (`SSLCertVerificationError:
self-signed certificate`), so the full pipeline could not be exercised
against live data here. What *was* verified directly:
- `composite.score_symbol()` end-to-end with synthetic OHLCV + mocked
  screener/signal-feed input → correct score, correct LONG/SHORT
  threshold behavior.
- `evaluate_scan()` integration triggers correctly on real breakout-shaped
  data (confirmed `unusual_volume_breakout` and `new_local_high_breakout`
  fire as expected).
- `store.log_signals()` → `store.resolve_due_signals()` →
  `store.performance_report()` round-trip against the real SQLite schema.
- CLI commands import and run cleanly (`edge scan/track/report`).

**Not yet verified against live data:** the exact field names altFINS'
`signal_feed_data` tool returns. `_signal_feed_index()` in `composite.py`
guesses across several plausible key names (`symbol`/`assetSymbol`/
`asset`, `direction`/`signalDirection`/`side`) defensively — it degrades
to "no match" rather than crashing if the real shape differs, but **the
first live run with `ALTFINS_API_KEY` set should print
`get_signal_feed()`'s raw output and confirm/fix the field names** in
`_signal_feed_index()`.

### Next steps to resume

1. Set `ALTFINS_API_KEY` (the user has VIP altFINS access) and run
   `edge scan` for real; fix `_signal_feed_index()` field names against
   the real response if needed.
2. Run `edge scan` + `edge track` on a cron for at least a few weeks
   before trusting `edge report` numbers — there's no shortcut to getting
   real forward-performance data.
3. Phase 3: pick an on-chain data provider (free-tier options worth
   evaluating: Santiment, CryptoQuant free tier, Etherscan/BscScan for
   known exchange wallet flows) and add it as another `components` input
   in `composite.score_symbol()`.
4. Phase 4: once Phase 2 has enough resolved signals to trust the
   shortlist, wire `TradingAgentsGraph` (TauricResearch/TradingAgents) on
   top of the shortlisted candidates for decision synthesis, feeding it
   the composite components instead of (or alongside) its default
   fundamentals/sentiment/news/technical analysts.
5. Phase 5: once Phase 4 is producing decisions with tracked outcomes,
   wrap the loop with `hermes-agent-self-evolution` so weighting/thresholds
   refine from real trace data instead of being hand-tuned constants
   (`TREND_WEIGHT`, `VOLUME_RELATIVE_WEIGHT`, `SCANNER_HIT_WEIGHT`,
   `SIGNAL_FEED_WEIGHT` in `composite.py` today).
