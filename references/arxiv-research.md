# arXiv Research — Crypto Trading Papers

## How This Integrates with the Edge Scanner

arXiv papers feed into three parts of the system:

1. **Improve existing components** — regime detection, feature weighting, parameter optimization
2. **Validate our approach** — papers confirming funding rate mean-reversion works
3. **New signal ideas** — alternative strategies discovered in literature

## Key Papers Found (Jun 2026)

### 1. Fundamentals of Perpetual Futures
- **ID:** 2212.06888v6
- **Published:** Dec 2022 (revised Aug 2024)
- **Relevance: HIGH** — Directly validates V8.0 funding rate approach
- **Key finding:** Deviations from no-arbitrage prices are larger in crypto than traditional markets, comove across currencies, diminish over time. An implied arbitrage strategy yields high Sharpe ratios.
- **PDF:** https://arxiv.org/pdf/2212.06888v6

### 2. Designing Funding Rates for Perpetual Futures
- **ID:** 2506.08573v1
- **Published:** Jun 2025
- **Relevance: MEDIUM** — Mathematical model of how exchanges design funding rates
- **Key finding:** Path-dependent funding rates can maintain price alignment. Replicating portfolios for hedging.
- **PDF:** https://arxiv.org/pdf/2506.08573v1

### 3. Dynamic Grid Trading Strategy
- **ID:** 2506.11921v1
- **Published:** Jun 2025
- **Relevance: LOW-MEDIUM** — Grid trading on crypto, zero expectation to outperformance
- **PDF:** https://arxiv.org/pdf/2506.11921v1

### 4. Concepts, Components and Collections of Trading Strategies
- **ID:** 1910.02144v2
- **Published:** Sep 2019
- **Relevance: MEDIUM** — Collection of trading strategies and market color
- **PDF:** https://arxiv.org/pdf/1910.02144v2

## How To Search For New Papers

```bash
# Regime detection / market state
curl -s "https://export.arxiv.org/api/query?search_query=all:market+regime+detection+HMM+trading&max_results=5&sortBy=relevance&sortOrder=descending"

# Funding rate / perpetual futures
curl -s "https://export.arxiv.org/api/query?search_query=all:crypto+funding+rate+perpetual+futures&max_results=5&sortBy=relevance&sortOrder=descending"

# Trading signal validation
curl -s "https://export.arxiv.org/api/query?search_query=all:trading+signal+validation+feature+selection+financial&max_results=5&sortBy=relevance&sortOrder=descending"

# Read abstract
web_extract(urls=["https://arxiv.org/abs/ID"])

# Read full paper
web_extract(urls=["https://arxiv.org/pdf/ID"])
```

## Research Cron

A weekly cron job searches for new papers on:
- Funding rate dynamics (V8.0 improvement)
- Market regime classification (regime_detector.py improvement)
- Trading signal validation (auto-promotion statistical rigor)

Results are delivered to Telegram if new papers found.