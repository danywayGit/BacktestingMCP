# Burn / Buyback Tracker — Multi-Source Rebuild (IMPLEMENTED 2026-08-27)

Status: **LIVE** — approved by Didier 2026-08-27, rebuilt from dead single-source design.
Cron: `burn-tracker` (`ca3bbd01ae79`), Sat 10:00 UTC, no_agent, deliver → Edge Scanner group,
topic 73 (this thread). Shell wrapper: `~/.hermes/scripts/burn_tracker.sh` (timeout 120s).

## 1. Why it was broken

The original tracker depended on a single dead API (Tokenomist → 404) and fell back to
CoinGecko free tier (429 rate-limited). Report degraded to a "trending coins" watchlist —
not burn events. All sources probed live 2026-08-27 before design:

| Source | Status |
|---|---|
| Tokenomist API (all paths) | 404/302 — dead |
| CoinGecko free (price, global) | 429 |
| Santiment (`SANTIMENT_API_KEY`) | Free plan exhausted (4-day lockout) |
| ultrasound.money / burned.top | unreachable |
| CryptoPanic (no key) | 403 |
| **Public RPCs: BSC, ETH, ARB, BASE, OP, POLYGON, Cronos, Solana** | ✅ free, no key |
| **CoinMarketCap web API** (`data-api/v3/.../detail`) | ✅ free, no key, supply fields |
| **Binance announcement CMS API** | ✅ free, no key |
| **RSS: CoinDesk, CoinTelegraph, Decrypt** | ✅ free |
| **GeckoTerminal search API** | ✅ free (contract resolution) |

## 2. Architecture (4 data layers + consensus)

### L1 — On-chain ground truth (exact burned amounts)
- Verified burn addresses sampled each run via public RPCs:
  - native coins: `eth_getBalance(burn_addr)`
  - ERC-20: `eth_call balanceOf(burn_addr)` on token contract (no archive node needed)
- Delta vs stored baseline (`data/burn_state.json`) = amount burned since last run.
- **Verified & enabled (2026-08-27):** BNB (BSC native, 16.5M at 0xdEaD),
  SHIB (ETH, 49.4B at 0xdEaD), CRO (Cronos native, 89.3K at 0xdEaD).
- Rejected during verification: OKB/LEO/KCS → burn to non-0xdEaD addresses (0 at dead),
  BNB-ERC (0.09 stale), HT (104 vestige). OKB/LEO covered by L2 instead.

### L2 — CMC supply cross-check (derived burn)
- `totalSupply` / `maxSupply` / `circulatingSupply` from CMC web API (no key).
- Derived burn = max − total, shown only when gap > 1% of supply (CMC lowers max over
  time, so small gaps are noise).
- **Tracked:** OKB, LEO, SHIB, CRO, BNB, APT.
- Note: KCS CMC slug unresolved (`kucoin-shares` returns no data) — excluded.

### L3 — Official announcements (Binance CMS)
- Pages 1–3 of announcement feed scanned for burn/buyback/buy-back/auto-burn.
- Catches quarterly BNB auto-burn and exchange burn/buyback notices.

### L4 — News RSS (buyback events)
- CoinDesk, CoinTelegraph, Decrypt — keyword scan, dedup by title, links kept.
- Catches buyback announcements before/without official exchange posts
  (e.g. 2026-08-27: Ethena buyback vote).

### Consensus report
All layers merged into one Telegram message. Per-entry failure isolation: a dead RPC or
feed never kills the report — that entry is marked ⚠️, baseline kept.

## 3. Files

- `config/burn_watchlist.json` — onchain[] + supply_cmc[] entries (verified before enable)
- `data/burn_state.json` — baselines (balance, total, dates); first run sets baseline,
  second run shows deltas
- `src/edge_scanner/burn_tracker.py` — full pipeline (L1–L4 + state + report)
- `~/.hermes/scripts/burn_tracker.sh` — cron wrapper (sends to topic 73, timeout 120s)

## 4. Verified end-to-end (2026-08-27)

- Baseline run: sets BNB/SHIB/CRO + CMC supplies, live Ethena buyback news found.
- Second run: shows real deltas (BNB +0.02 in minutes — tiny, meaningful at weekly cadence).
- Full wrapper test: delivered to Edge Scanner topic 73, exit 0.

## 5. Adding a new token (procedure)

1. Probe burn address balance live (script pattern in `/tmp/burn_verify.py` + `_2`/`_3`):
   - native → `eth_getBalance(addr)`, erc20 → `balanceOf(addr)` on contract
2. Only add to `onchain[]` if balance > 0 (real burns to that address).
3. Or add to `supply_cmc[]` if CMC has supply data (slug from CMC detail endpoint).
4. Run tracker twice; confirm baseline + delta rows.

## 6. Limitations / known gaps

- ETH base-fee burn (EIP-1559) has no burn address — needs separate tracker if wanted.
- On-chain buyback detection (treasury address deltas) is per-project, not generic.
- Binance CMS only indexes recent pages (rolling window); old announcements age out.
- Santiment burnRate data available when key quota resets (standby layer, not wired).
- RPC availability varies by provider; `publicnode.com` + native seed nodes most stable.