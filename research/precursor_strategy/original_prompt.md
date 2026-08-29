# Hermes Task: Data-Driven Precursor-Pattern Strategy for BTCUSDT / ETHUSDT Futures

## 0. Framing (read this first)

The goal is **not** to hand-pick indicators or guess parameters. The goal is to let the
data tell us what tends to happen in the candles *before* a large move (>=5% in either
direction), and only turn that into a strategy once we've shown — with proper
out-of-sample testing — that the pattern is statistically real and not noise.

Important vocabulary correction for the whole project: what we are mining is
**"precursor conditions"** or **"leading patterns"**, not "causes." Markets don't give
us controlled experiments, so we can never prove causation — only that a condition
reliably preceded a move more often than chance would predict. Use "precursor",
"leading indicator", "antecedent pattern" in all code, logs, and strategy names.
Never claim causality in outputs, comments, or reports.

This is a research-then-execution pipeline. Do not skip phases. Do not backtest a
"strategy" before Phase 4 has produced a statistically validated precursor set. Each
phase has a required deliverable — produce it before moving on, and stop for human
review at the checkpoints marked **[HUMAN CHECKPOINT]**.

---

## 1. Objective

Build, validate, and continuously improve a systematic strategy for BTCUSDT and
ETHUSDT perpetual futures that:

1. Detects historical episodes where price moved >=5% within a defined window.
2. Extracts everything knowable about market state in the N candles *before* each
   episode — numeric features, indicator states, and visual chart structure.
3. Mines these episodes for recurring precursor conditions using statistically rigorous
   methods (not eyeballing).
4. Converts validated precursors into explicit, rule-based entry/exit logic.
5. Backtests with realistic costs (fees, slippage, funding), then paper-trades.
6. Runs a continuous feedback loop: every new completed trade and every new 5%+ move
   feeds back into the pattern library, so the strategy keeps re-validating and
   refining itself over time rather than staying frozen on one historical fit.

---

## 2. Repo & Environment Assumptions

State these explicitly in your first message back to me if any don't hold, rather than
guessing:

- Where OHLCV history for BTCUSDT/ETHUSDT perp futures lives or how it will be fetched
  (exchange API, existing data loader in repo, etc.), and what timeframes are available
  (e.g. 1m/5m/15m/1h).
- Whether the repo already has: an indicator library, a backtesting engine, a
  paper-trading/live-execution layer, a logging/reporting convention.
- Where funding rate, open interest, and order book/liquidation data (if available) can
  be sourced — these matter a lot for futures specifically.
- Where to write outputs: a suggested convention is
  `research/precursor_strategy/` for all phase artifacts (data, notebooks/scripts,
  logs, findings), separate from `strategies/` where the final live strategy code goes.

---

## 3. Phase 1 — Event Detection

**Input:** OHLCV history, both symbols, your best available granularity (start with 15m
or 1h; note that "5%" means very different things at 1m vs 1d, so pick and record the
window explicitly).

**Task:**
- Define a "move event" precisely: e.g. "close-to-close return over any rolling W-candle
  window >= +5% or <= -5%", with W as a configurable parameter (test a few: e.g. 4h, 12h,
  24h equivalent in candles).
- Detect all such events historically, tag each with: symbol, direction (up/down),
  start timestamp, end timestamp, magnitude, W used.
- De-duplicate overlapping events (a 5% move often triggers many overlapping windows —
  collapse to the event's actual start).

**Deliverable:** `events.csv` (or equivalent) with one row per distinct move episode,
plus a short markdown summary: how many events found, up vs down split, distribution of
magnitudes and durations, per symbol.

**[HUMAN CHECKPOINT]** — Share the event list summary before proceeding. The move
definition materially changes everything downstream; I want to sanity-check it.

---

## 4. Phase 2 — Precursor Feature Extraction

For each detected event, look at the **K candles immediately preceding** the event start
(test a few values of K, e.g. 10, 20, 50) and extract:

### 4a. Numeric / indicator features
- Price action: candle body/wick ratios, consecutive candle direction streaks,
  higher-highs/lower-lows structure, realized volatility (ATR, stdev of returns),
  recent range compression (Bollinger Band width, Keltner squeeze).
- Volume: volume trend, volume spikes vs rolling average, volume-price divergence.
- Momentum/oscillators: RSI, MACD, stochastic — value *and* slope, not just level.
- Futures-specific: funding rate level and trend, open interest change, long/short
  ratio if available, liquidation clusters if available.
- Cross-asset: was the other symbol (ETH vs BTC) already moving? Lag/lead relationship?
- Time-based: time of day, day of week, proximity to known macro events (FOMC, CPI, major
  options expiry) if you can source a calendar.

### 4b. Visual chart features (this is the part you specifically flagged — don't skip it)
- Render a candlestick chart image of the K candles before each event (plus a bit of
  extra context before that), using your existing charting/plotting capability.
- Use vision-capable analysis on the rendered chart to describe structure a numeric
  feature table can miss: consolidation triangles, flags/pennants, wedges, double
  tops/bottoms, support/resistance tests, wick rejection clusters, visible
  accumulation/distribution ranges.
- Store this as a structured field per event, e.g. a short controlled vocabulary
  (`triangle_consolidation`, `range_compression`, `failed_breakdown_retest`, ...) plus a
  free-text note — not paragraphs of prose per event, since this needs to be
  aggregated and counted later, not read one-by-one.
- Do this for a **sample first** (e.g. 30–50 events per symbol/direction) before
  scaling to all events — visual labeling is more expensive, so validate the labeling
  process and vocabulary on a sample, then decide whether to run it on the full set.

**Deliverable:** a feature table, one row per event, combining 4a + 4b, plus the
controlled-vocabulary chart-pattern label(s) per event.

---

## 5. Phase 3 — Precursor Mining

**Task:**
- Compare the feature distributions of "before an event" candles against a **matched
  control set** of ordinary candles (same symbol, same time-of-day distribution, no
  event following) — this control group is essential; without it you'll just rediscover
  "the market is usually volatile," not anything specific to big moves.
- For each feature, test whether its distribution before events differs meaningfully
  from the control distribution (e.g. compare distributions and effect sizes, not just
  eyeballing means).
- Do the same for the chart-pattern labels: is `triangle_consolidation` overrepresented
  before events vs baseline?
- Separate this analysis by direction (up-move precursors vs down-move precursors are
  probably different) and consider by symbol.
- Rank candidate precursors by effect size and consistency, not by whichever one has the
  prettiest p-value from a single test — apply a correction for the fact that you're
  testing many features at once (multiple-comparisons correction), since testing dozens
  of features will produce some fake winners by chance alone.

**Deliverable:** a ranked list of candidate precursor conditions with effect sizes,
sample sizes, and a plain-language description of each, split by symbol and direction.

**[HUMAN CHECKPOINT]** — Share this ranked list. This is the actual "insight" output of
the research phase — worth a careful look before building trading logic on top of it.

---

## 6. Phase 4 — Out-of-Sample Validation (the step that prevents self-deception)

- Split history into a train period (where Phase 3 mining happened) and a **held-out**
  test period the mining phase never touched.
- For each top candidate precursor (or combination), check: in the held-out period, did
  candles matching this precursor actually precede 5%+ moves more often than base rate?
- Only precursors that survive this out-of-sample check move forward. Log the ones that
  don't — that negative result is useful too and should be kept in the research log, not
  discarded.
- If very little survives, that itself is a valid and important finding — report it
  honestly rather than lowering the bar to find something to report.

**Deliverable:** validated precursor list with in-sample vs out-of-sample performance
side by side.

---

## 7. Phase 5 — Strategy Formalization & Backtest

- Convert validated precursors into explicit entry rules (e.g. "IF [precursor
  conditions] THEN consider entry in [direction]").
- Define exit logic separately from entry logic: stop-loss, take-profit, time-based
  exit, or trailing — this needs its own reasoning, the precursor mining doesn't tell
  you how to exit.
- Backtest with realistic frictions: taker/maker fees, slippage assumption, funding
  rate cost/gain while position is held (this matters a lot on perpetuals), and
  realistic position sizing/leverage assumptions.
- Run this on the out-of-sample period primarily; treat any further parameter tweaking
  from here as a new in-sample fit that needs its *own* fresh out-of-sample slice if
  you keep iterating (walk-forward, not "keep tuning against the same test set").

**Deliverable:** backtest report — equity curve, win rate, average win/loss, max
drawdown, Sharpe or similar, sensitivity to slippage/fee assumptions.

**[HUMAN CHECKPOINT]** — Review before any live/paper capital is involved.

---

## 8. Phase 6 — Paper Trading

- Run the strategy live on paper (no real capital) for a meaningful sample size before
  considering real deployment.
- Log every signal generated, whether taken, and outcome — including near-misses where
  precursor conditions were "almost" met, since that boundary data feeds Phase 9.

---

## 9. Phase 7 — Continuous Self-Improvement Loop

This is the "auto-improve" part:

- On a regular cadence (e.g. weekly), append newly completed events (both traded and
  simply observed 5%+ moves) into the Phase 1–2 dataset.
- Re-run Phase 3 mining periodically to check whether precursor rankings are stable or
  drifting — markets change regimes, so a precursor that worked in one period may decay.
- Track live/paper performance of each precursor-based rule separately, and flag any
  rule whose live hit-rate is diverging meaningfully from its backtested/validated
  hit-rate — this is your early warning that a pattern has stopped working.
- Maintain a single running "research log" (markdown or similar) documenting: what was
  tried, what was validated, what decayed, what was dropped and why. This log is the
  actual long-term value of this project — it's what lets the next iteration build on
  the last instead of restarting blind.
- Any change to live trading rules coming out of this loop should hit a
  **[HUMAN CHECKPOINT]** before being deployed with real capital.

---

## 10. Guardrails (apply throughout, not just at the end)

- No look-ahead bias: every feature used at "time before the event" must only use data
  actually available at that timestamp.
- No survivorship bias: don't quietly drop events that don't fit a hypothesis.
- Effect sizes and sample counts must be reported alongside every claimed pattern —
  a pattern from 6 events is not a pattern.
- Multiple-comparisons awareness: mining dozens of features will produce false
  positives by chance; the out-of-sample step in Phase 4 exists specifically to catch
  this, so don't skip it under time pressure.
- Every output should distinguish "precursor found and validated out-of-sample" from
  "precursor found in-sample only" from "hypothesis not yet tested" — never blur these.
- This project is quantitative research support, not financial advice, and past
  precursor patterns are not guarantees of future performance — regimes shift.

---

## 11. What I want back from you (Hermes) right now

Before writing any code:
1. Confirm/fill in the repo & data assumptions in Section 2.
2. Propose the specific event-window definition (W) and lookback window (K) you'll
   start with, with brief reasoning.
3. Propose the file/output structure you'll use under `research/precursor_strategy/`.
4. Then proceed through Phase 1, stopping at the checkpoint.
