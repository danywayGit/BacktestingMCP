# Schema Reference — events.csv & precursor_features.csv

**Note:** `events.csv` in this folder contains illustrative, synthetic rows only — it's
a schema sketch, not real BTCUSDT/ETHUSDT market data. Hermes should regenerate this
file from actual historical OHLCV once Phase 1 runs for real. Same for the sample
feature rows below.

---

## events.csv — one row per detected 5%+ move episode

| column | type | meaning |
|---|---|---|
| `event_id` | string | unique id, e.g. `BTC-0001` |
| `symbol` | string | `BTCUSDT` / `ETHUSDT` |
| `timeframe` | string | candle timeframe used for detection, e.g. `15m` |
| `window_candles_W` | int | rolling window size (in candles) used to define the move |
| `direction` | string | `up` / `down` |
| `event_start_ts` | ISO8601 | timestamp of the first candle in the triggering window |
| `event_end_ts` | ISO8601 | timestamp of the candle where the 5% threshold was met |
| `start_price` | float | close price at `event_start_ts` |
| `end_price` | float | close price at `event_end_ts` |
| `pct_move` | float | signed % return over the window |
| `duration_candles` | int | number of candles the move took |
| `max_intra_window_pct` | float | largest peak-to-start % excursion within the window (can exceed `pct_move` if price overshot then pulled back) |
| `notes` | string | optional free text, e.g. `liquidation_cascade_suspected` — keep sparse, this is not for analysis, just human context |

De-duplication rule: if two overlapping windows both cross the 5% threshold for the same
directional move, keep only the earliest `event_start_ts` / final `event_end_ts` pair
covering the whole move, don't count it twice.

---

## precursor_features.csv — one row per event, holds the Phase 2 lookback features

Each row corresponds to one `event_id` from `events.csv`, describing the K candles
**before** `event_start_ts`.

### Identity
| column | type |
|---|---|
| `event_id` | string (foreign key to events.csv) |
| `lookback_K` | int — number of candles examined before the event |

### Numeric / indicator features (4a)
| column | type | meaning |
|---|---|---|
| `avg_body_wick_ratio` | float | mean candle body-to-wick ratio over K candles |
| `direction_streak_len` | int | length of the longest consecutive same-direction candle streak |
| `realized_vol_atr` | float | ATR over K candles |
| `bb_width_pctile` | float | Bollinger Band width, as a percentile vs trailing 100 candles (captures "squeeze") |
| `volume_zscore` | float | most recent volume vs rolling mean/stdev |
| `volume_price_divergence` | bool | true if price flat/down but volume rising, or similar mismatch |
| `rsi_level` | float | RSI value at last candle before event |
| `rsi_slope` | float | RSI change over K candles |
| `macd_hist_slope` | float | MACD histogram slope over K candles |
| `funding_rate_level` | float | funding rate at last candle before event |
| `funding_rate_trend` | float | change in funding rate over K candles |
| `open_interest_change_pct` | float | % change in OI over K candles |
| `long_short_ratio` | float | if sourced from exchange |
| `other_symbol_lead_return_pct` | float | return of the *other* symbol (ETH if this row is BTC, vice versa) over same K window |
| `hour_of_day_utc` | int | 0–23 |
| `day_of_week` | int | 0–6 |
| `near_macro_event` | bool | true if within a defined window of FOMC/CPI/major expiry, if calendar sourced |

### Visual / chart features (4b)
| column | type | meaning |
|---|---|---|
| `chart_pattern_labels` | string (semicolon-separated) | controlled vocabulary, e.g. `triangle_consolidation;wick_rejection_cluster` |
| `chart_pattern_confidence` | float 0–1 | confidence of the visual read, self-reported |
| `chart_image_path` | string | path to the rendered chart image used for the visual read, for auditability |
| `chart_notes` | string | short free text, kept brief — for aggregation later, not prose |

### Control-set flag (needed for Phase 3 comparison)
| column | type | meaning |
|---|---|---|
| `is_control` | bool | false for real precursor rows; the same feature extraction is run on a matched sample of *non-event* candles and appended here with `is_control = true` and `event_id` set to a synthetic control id — this is what Phase 3 compares against |

---

## Suggested minimal example row (precursor_features.csv), illustrative only

```
event_id,lookback_K,avg_body_wick_ratio,direction_streak_len,realized_vol_atr,bb_width_pctile,volume_zscore,rsi_level,rsi_slope,funding_rate_level,chart_pattern_labels,is_control
BTC-0001,20,1.8,4,310.5,12.4,2.1,58.3,6.2,0.0021,triangle_consolidation,false
BTC-0001-ctrl-014,20,1.1,1,290.2,54.0,0.2,50.1,-0.4,0.0018,,true
```

Keep the control rows in the *same file*, distinguished by `is_control`, so Phase 3
comparisons are simple filters rather than joins across files.
