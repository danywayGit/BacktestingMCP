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
 altFINS signal feed (VIP-channel   │
   equivalent, bullish/bearish)    ─┼─► composite score per symbol ─► log signal ─► track forward outcome ─► win-rate report
 BacktestingMCP TA scanners         │        (Phase 1 + 3)                (Phase 2)         (Phase 2)              (Phase 2)
   (price/volume breakouts, real OHLCV)
 Santiment on-chain exchange flow  ─┘
   (Phase 3)

 [not built yet]
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

### Phase 3 — On-chain exchange-flow bias (BUILT)

- `src/integrations/santiment_client.py` — sync GraphQL client for
  Santiment's SanAPI (`https://api.santiment.net/graphql`,
  `Authorization: Apikey <SANTIMENT_API_KEY>`). `get_onchain_snapshot()`
  pulls trailing `exchange_inflow_usd`/`exchange_outflow_usd` over a short
  lookback window. Provider choice and rationale:
  - **CryptoQuant** exchange-netflow API requires a paid
    Professional/Premium plan — no free programmatic access, ruled out.
  - **Santiment** free tier (~1,000 calls/month) does expose
    exchange-flow + whale metrics via API, so it's the only option with
    free programmatic access. Caveat: most free-tier metrics carry up to
    a **30-day data lag** and a 1-year historical cap — this makes
    on-chain here a slow-moving directional bias, not a fresh signal. The
    call budget also means this should not be queried on every scan cycle
    for the full candidate universe once running on a real cron schedule
    (see "Next steps" below).
  - **Etherscan/BscScan** were considered but would require building
    custom whale/exchange-wallet aggregation logic from raw transfer logs
    — far more work than Santiment's pre-aggregated metrics for an
    equivalent free-tier signal, so not pursued for v1.
- `_SLUG_MAP` in `santiment_client.py` covers ~30 large/mid-cap symbols
  (Santiment identifies assets by project "slug", not exchange ticker).
  Symbols outside this map, or with no `SANTIMENT_API_KEY` configured,
  degrade to "no on-chain signal" (`components["onchain_netflow_ratio"]
  = None`) rather than blocking the scan — same defensive pattern as the
  altFINS signal feed fallback.
- Wired into `composite.score_symbol()`: net exchange outflow (coins
  leaving exchanges, i.e. `outflow_usd > inflow_usd`) reads as a bullish
  accumulation bias; net inflow (coins arriving, available to sell) reads
  bearish. Computed as a dimensionless ratio in [-1, 1] so it's comparable
  across assets of very different market cap, weighted by
  `ONCHAIN_NETFLOW_WEIGHT = 2.0`.
- Verified with mocked `get_onchain_snapshot()` responses: net-outflow
  case correctly added a positive (bullish) contribution and flipped
  `onchain_netflow_ratio` positive; a `SantimentError` (unmapped symbol /
  no key / API failure) correctly degraded to `None` without raising out
  of `score_symbol()`.
- **Not yet verified against a live Santiment API key:** exact metric
  availability/lag on the free tier, and whether `exchange_inflow_usd` /
  `exchange_outflow_usd` are the precise metric names returned (Santiment
  has renamed metrics before). First live run with `SANTIMENT_API_KEY` set
  should sanity-check the raw GraphQL response shape the same way altFINS'
  `signal_feed_data` needs checking.

### Verified so far

This sandbox's network egress allowlist does not include `api.binance.com`,
`mcp.altfins.com`, or `api.santiment.net` (confirmed via direct `requests`
calls returning `403 ... Host not in allowlist`), so the full pipeline
could not be exercised against live data from this environment. This is an
environment network-policy setting, not a code issue — see
https://code.claude.com/docs/en/claude-code-on-the-web for how to allow
hosts, or run the live test from the user's own NAS/RTX 4090 machine where
egress isn't restricted. What *was* verified directly (synthetic/mocked
data, no network):
- `composite.score_symbol()` end-to-end with synthetic OHLCV + mocked
  screener/signal-feed input → correct score, correct LONG/SHORT
  threshold behavior.
- `evaluate_scan()` integration triggers correctly on real breakout-shaped
  data (confirmed `unusual_volume_breakout` and `new_local_high_breakout`
  fire as expected).
- `store.log_signals()` → `store.resolve_due_signals()` →
  `store.performance_report()` round-trip against the real SQLite schema.
- CLI commands import and run cleanly (`edge scan/track/report`).
- `composite.score_symbol()` on-chain branch with mocked
  `santiment_client.get_onchain_snapshot()`: net-outflow input correctly
  added bullish score contribution; `SantimentError` correctly degraded
  to `onchain_netflow_ratio = None` without raising.

**Not yet verified against live data:**
- The exact field names altFINS' `signal_feed_data` tool returns.
  `_signal_feed_index()` in `composite.py` guesses across several
  plausible key names (`symbol`/`assetSymbol`/`asset`,
  `direction`/`signalDirection`/`side`) defensively — it degrades to "no
  match" rather than crashing if the real shape differs, but **the first
  live run with `ALTFINS_API_KEY` set should print `get_signal_feed()`'s
  raw output and confirm/fix the field names** in `_signal_feed_index()`.
- Santiment's actual free-tier metric names/availability/lag — first live
  run with `SANTIMENT_API_KEY` set should sanity-check the raw GraphQL
  response shape from `get_onchain_snapshot()`.

### Next steps to resume

1. Set `ALTFINS_API_KEY` (the user has VIP altFINS access) and
   `SANTIMENT_API_KEY` (sign up for Santiment's free tier), then run
   `edge scan` for real from a machine with unrestricted egress (this
   sandbox blocks `api.binance.com`/`mcp.altfins.com`/`api.santiment.net`
   at the network-policy level — see "Verified so far" above). Fix
   `_signal_feed_index()` field names and `santiment_client._METRICS`
   names against the real responses if needed.
2. Run `edge scan` + `edge track` on a cron for at least a few weeks
   before trusting `edge report` numbers — there's no shortcut to getting
   real forward-performance data.
3. Watch Santiment's call budget (~1,000/month free tier): once running
   on a real cron schedule, either cache `get_onchain_snapshot()` results
   per symbol for several hours, or only query it for the shortlist that
   already passed the altFINS+TA thresholds — not every scan cycle for
   the full candidate universe.
4. Phase 4: once Phase 2 has enough resolved signals to trust the
   shortlist, wire `TradingAgentsGraph` (TauricResearch/TradingAgents) on
   top of the shortlisted candidates for decision synthesis, feeding it
   the composite components instead of (or alongside) its default
   fundamentals/sentiment/news/technical analysts.
5. Phase 5: once Phase 4 is producing decisions with tracked outcomes,
   wrap the loop with `hermes-agent-self-evolution` so weighting/thresholds
   refine from real trace data instead of being hand-tuned constants
   (`TREND_WEIGHT`, `VOLUME_RELATIVE_WEIGHT`, `SCANNER_HIT_WEIGHT`,
   `SIGNAL_FEED_WEIGHT`, `ONCHAIN_NETFLOW_WEIGHT` in `composite.py` today).
