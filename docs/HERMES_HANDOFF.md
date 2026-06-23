# Crypto Trading Edge Scanner — Handoff Brief (for Hermes Agent)

## Goal
Build a system that scans crypto spot/futures markets, scores candidates across
multiple signal sources, and tracks *measured* forward performance (win-rate, avg
return) per symbol/hour/direction — not a system that promises winning every day
(no legitimate system can). Risk management stays the user's responsibility; this
system's job is to surface candidates with validated positive expectancy.

## Repos involved
- **BacktestingMCP** — where all new code lives (Python, MCP server, CLI,
  GPU/CuPy backtesting engine, SQLite via `CryptoDatabase`)
- **trading-strategies-research** — research/docs repo, Pine Script fixes,
  backtest specs (not touched by this work yet)
- **altfinsMCP** — reference repo for altFINS' MCP tool catalog (not modified,
  just consulted)
- **BackTestingSignals** — sibling repo with an existing win-rate-by-segment
  validation pattern this system's Phase 2 mirrors (not modified)

All work so far is on branch **`claude/crypto-trading-edge-system-f89xa0`** in
**BacktestingMCP**, pushed and up to date.

## Architecture
```
 altFINS screener (2000+ symbols)  ─┐
 altFINS signal feed (VIP-channel   │
   equivalent, bullish/bearish)    ─┼─► composite score per symbol ─► log signal ─► track forward outcome ─► win-rate report
 BacktestingMCP TA scanners         │        (Phase 1 + 3)                (Phase 2)         (Phase 2)              (Phase 2)
   (price/volume breakouts, real OHLCV)
 Santiment on-chain exchange flow  ─┘
   (Phase 3)

 [not built yet]
 TradingAgents (multi-agent debate) ──► decision synthesis on shortlist (Phase 4)
 Hermes Agent + self-evolution      ──► refines weights/thresholds from
                                         real tracked outcomes           (Phase 5)
```

## Why this shape (decisions already made — don't re-litigate without new info)
- **altFINS has no on-chain data** (confirmed by inspecting altfinsMCP) → needed
  a separate provider for Phase 3.
- **CryptoQuant's** exchange-netflow API requires a paid Professional/Premium
  plan → ruled out.
- **Santiment** free tier (~1,000 calls/month, exchange_inflow_usd/
  exchange_outflow_usd, whale metrics) is the only on-chain provider with free
  programmatic access → chosen for Phase 3. Caveat: up to 30-day data lag on
  free tier, so it's a slow directional bias, not a fresh signal.
- **Etherscan/BscScan** considered, rejected for v1 — would require building
  custom whale/exchange-wallet aggregation from raw logs; Santiment's
  pre-aggregated metrics are equivalent value for less work.
- **TradingAgents** (TauricResearch, Apache-2.0, LangGraph) chosen for Phase 4
  decision synthesis instead of building analyst/debate/risk-manager logic
  from scratch. Supports local Ollama → zero marginal cost on the RTX 4090.
- **Hermes Agent + hermes-agent-self-evolution** (NousResearch, MIT) chosen for
  Phase 5 instead of bespoke RL — RL needs a labeled training dataset and GPU
  training pipeline; Hermes's self-evolution reads execution traces directly,
  which is exactly what Phase 2's tracking loop produces. **Do not build Phase
  5 before Phase 2 has weeks of resolved signals — there's nothing real to
  learn from otherwise.**
- **No websockets/tick data.** These are 1h-4h swing signals; periodic polling
  (every 5-15 min) is sufficient and far cheaper. Phases 1-3 run fine as a NAS
  cron job, 24/7, near-zero cost. Phase 4 (LLM calls) should only run on the
  ~10-20 symbol shortlist that already passed Phase 1+2 thresholds, never the
  full 2000+ symbol universe.

## What's built (Phase 1, 2, 3 — done)

**Phase 1 — composite scanner:**
- `src/integrations/altfins_client.py` — sync wrapper around altFINS MCP
  server (`https://mcp.altfins.com/mcp`, header `X-Api-Key`). Functions:
  `get_screener_data()`, `get_signal_feed()`, `get_recent_news_counts()`,
  `parse_trend_score()`. Requires `ALTFINS_API_KEY` env var. Raises
  `AltfinsError` if missing/unreachable.
- `src/edge_scanner/composite.py` — `run_composite_scan()` discovers
  candidates from altFINS, cross-checks signal feed, adds on-chain bias
  (Phase 3), confirms with BacktestingMCP's own `evaluate_scan()` breakout
  scanners on real OHLCV. Produces `CandidateScore` (composite score,
  LONG/SHORT/None direction, `components` dict for auditability). Falls back
  to static `CRYPTO_PAIRS` + TA-only scoring if `ALTFINS_API_KEY` unset.
- `src/data/database.py` — `edge_signals` table + `insert_edge_signal` /
  `get_pending_edge_signals` / `resolve_edge_signal` / `get_resolved_edge_signals`.

**Phase 2 — forward-validation loop:**
- `src/edge_scanner/store.py` — `log_signals()` persists actionable signals
  with entry price; `resolve_due_signals()` checks signals past their
  `horizon_hours`, computes directional forward return, labels WIN (>0.3%) /
  LOSS (<-0.3%) / FLAT; `performance_report()` aggregates win-rate/avg-return
  by symbol, hour-of-day, or direction.

**Phase 3 — on-chain exchange-flow bias:**
- `src/integrations/santiment_client.py` — sync GraphQL client for
  Santiment's SanAPI (`https://api.santiment.net/graphql`, header
  `Authorization: Apikey <key>`). `get_onchain_snapshot(symbol)` sums
  trailing `exchange_inflow_usd`/`exchange_outflow_usd`. `_SLUG_MAP` covers
  ~30 large/mid-cap symbols (Santiment uses project slugs, not tickers).
  Requires `SANTIMENT_API_KEY`. Raises `SantimentError` if
  missing/unmapped/unreachable.
- Wired into `composite.score_symbol()`: net exchange outflow → bullish
  accumulation bias; net inflow → bearish sell-pressure bias. Expressed as
  ratio in [-1,1] (comparable across market caps), weight
  `ONCHAIN_NETFLOW_WEIGHT = 2.0`.

**CLI** (all in `src/cli/main.py`):
```
python -m src.cli.main edge scan   --timeframe 1h --lookback-days 30 --per-side 20 --horizon-hours 24
python -m src.cli.main edge track
python -m src.cli.main edge report --group-by symbol   # or hour, direction
```

**Scoring weights** (module constants in `composite.py`, hand-tuned, candidates
for Phase 5 to refine): `TREND_WEIGHT=0.4`, `VOLUME_RELATIVE_WEIGHT=2.0` (cap
3.0), `SCANNER_HIT_WEIGHT=2.5`, `SIGNAL_FEED_WEIGHT=3.0`,
`ONCHAIN_NETFLOW_WEIGHT=2.0`. Direction threshold: `DEFAULT_MIN_ABS_SCORE=3.0`.

**Living plan doc:** `docs/EDGE_SCANNER_PLAN.md` in BacktestingMCP — read this
too, it has full detail and stays updated as phases complete.

## Verified vs. NOT verified
Verified only with synthetic/mocked data (the dev sandbox this was built in
has no live network access to Binance/altFINS/Santiment):
- `composite.score_symbol()` end-to-end scoring + threshold logic
- `evaluate_scan()` breakout triggers on real breakout-shaped data
- `store.log_signals()` → `resolve_due_signals()` → `performance_report()`
  round-trip against real SQLite schema
- On-chain net-outflow/net-inflow scoring branch + graceful degrade on
  `SantimentError`
- CLI commands import/run cleanly

**NOT yet verified against live data (this is the actual blocker — needs your
NAS's real network access):**
1. The exact field names altFINS' `signal_feed_data` tool returns —
   `_signal_feed_index()` in `composite.py` guesses across plausible key
   names (`symbol`/`assetSymbol`/`asset`, `direction`/`signalDirection`/
   `side`) and degrades to "no match" rather than crashing, but needs
   confirming/fixing against a real response.
2. Santiment's actual free-tier metric names/availability/lag — needs
   sanity-checking `get_onchain_snapshot()`'s raw GraphQL response against a
   real `SANTIMENT_API_KEY`.

## Next steps to do (in order)
1. **On the NAS** (real network access, unlike the dev sandbox): clone/pull
   BacktestingMCP, checkout `claude/crypto-trading-edge-system-f89xa0`, set
   `ALTFINS_API_KEY` and `SANTIMENT_API_KEY` in `.env` (placeholders already
   there).
2. Run `python -m src.cli.main edge scan` for real. Print/inspect raw
   `get_signal_feed()` and `get_onchain_snapshot()` output. Fix
   `_signal_feed_index()` field names in `composite.py` and `_METRICS` names
   in `santiment_client.py` if the real shapes differ from what's guessed.
3. Set up `edge scan` on cron every 15-60 min and `edge track` hourly — **as a
   plain cron job, not LLM-driven** (mechanical/deterministic, no need to burn
   agent calls on it). Let it run for **at least a few weeks** before trusting
   `edge report` numbers — there's no shortcut to real forward-performance
   data.
4. Watch Santiment's call budget (~1,000/month free tier) — cache
   `get_onchain_snapshot()` per symbol for several hours, or only query the
   shortlist that already passed altFINS+TA thresholds, not the full
   candidate universe every cycle.
5. **Phase 4** (once Phase 2 has enough resolved signals to trust the
   shortlist): wire `TradingAgentsGraph` (TauricResearch/TradingAgents, local
   Ollama on the RTX 4090 preferred) on top of the shortlisted candidates for
   decision synthesis, feeding it the composite `components` breakdown.
6. **Phase 5** (once Phase 4 is producing decisions with tracked outcomes):
   wrap the loop with `hermes-agent-self-evolution` (DSPy+GEPA) to refine
   `TREND_WEIGHT`/`VOLUME_RELATIVE_WEIGHT`/`SCANNER_HIT_WEIGHT`/
   `SIGNAL_FEED_WEIGHT`/`ONCHAIN_NETFLOW_WEIGHT` from real trace data instead
   of the current hand-tuned constants.

## Note on Hermes's two distinct roles here
- **Right now**: use the general `hermes` CLI agent as a coding/ops assistant
  to do step 1-2 above (it has real network access the dev sandbox didn't) —
  give it this whole brief as context, then the specific task "do step 2."
- **Later (Phase 5 only)**: `hermes-agent-self-evolution` is a *different*
  tool — it reads resolved-signal execution traces to retune the scoring
  weights. There's nothing for it to learn from until step 3 has run for
  weeks. Don't conflate "Hermes is running my cron job" with "Phase 5 is
  wired in" — they're sequential.
