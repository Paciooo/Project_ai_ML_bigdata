# Coin-flip Monte Carlo and nearest-expiry VIX futures

Exploratory research extension. No source CSV or previous run was modified.

Random-policy trials: 10,000. Target: next observed daily price change in the
nearest unexpired contract, with front-contract changes excluded. All models use the same
test origins and 64 observations of the SAME expiry as historical context.
Other maturities are not separate predictors. The spot file is audited as a reference only.

## VIX directional results

| Model | Accuracy, all outcomes | Balanced accuracy, non-flat | vs majority, pp | 20-day interval, pp |
|---|---:|---:|---:|---|
| majority | 59.59% | 50.00% | +0.00 | [+0.00, +0.00] |
| persistence | 55.06% | 53.20% | -4.53 | [-8.08, -0.97] |
| reversal | 44.94% | 46.80% | -14.66 | [-19.72, -9.70] |
| train_median_reversion | 52.26% | 48.44% | -7.33 | [-12.18, -3.12] |
| logreg | 59.81% | 50.91% | +0.22 | [-0.97, +1.29] |
| xgboost | 54.42% | 48.58% | -5.17 | [-8.08, -2.59] |
| timestral_5M | 50.00% | 50.15% | -9.59 | [-14.33, -5.17] |
| timestral_28M | 48.28% | 47.33% | -11.31 | [-16.38, -6.68] |
| timestral_123M | 52.05% | 49.77% | -7.54 | [-11.53, -3.77] |
| timestral_209M | 51.83% | 49.58% | -7.76 | [-11.75, -4.09] |

Local model selected on validation only: **logreg**. No hyperparameter search was performed.

## Reading the random-decision comparison

- Fair coin: independently UP or DOWN with probability 0.5.
- Matched coin: the same expected UP share as the evaluated policy. This is a conditional diagnostic, not a fitted live strategy.
- Flat outcomes count as misses for both the models and coins. Expected fair-coin accuracy is therefore half the non-flat coverage, not automatically 50%.
- The simulations condition on the observed outcomes. Their 95% bands describe luck from random decisions on this history, not uncertainty over future market paths.
- A model can beat a fair coin while failing to beat the training-majority rule. Use both comparisons.
- conditional_tail_fraction and its Holm adjustment are conditional random-policy diagnostics, not proof of market alpha. Minimum resolution is 1/(simulations+1).
- Session/day-block bootstrap comparisons against majority and a simple training-median reversion rule are saved separately.

## Trading interpretation

No VIX dollar backtest is produced: these files contain index OHLC and futures daily price observations,
without verified price publication times, executable quotes, or account costs. Spot VIX is a calculated
30-day implied-volatility index and cannot itself be bought. Nearest-expiry VIX futures price direction
is a different target from spot VIX direction, S&P 500 direction, or realized S&P 500 volatility.

NQ random-sign profit diagnostics are restricted to saved rows whose assumed entry is no earlier than
the recorded decision. Intraday entries failing that check are explicitly skipped. Daily/session checks
use exactly each model's active timestamps, recorded per-fold thresholds, roll exclusions and the same
zero/base cost assumptions. They remain illustrative bar-price simulations, not verified fills.

## Data and availability limits

- spot: `{"path": "C:\\Users\\aless\\Desktop\\Scuola\\BSAMC\\VIX\\Codice\\data\\spot_LSEG_VIX.csv", "raw_rows": 5650, "unique_rows": 5485, "exact_duplicates_removed": 165, "first": "2004-03-26", "last": "2025-11-19", "sha256": "25f2d4e69d0b2e5167459b8a84d22595b5e02128c6b1ecdf54178ea9a45216a6"}`
- futures: `{"path": "C:\\Users\\aless\\Desktop\\Scuola\\BSAMC\\VIX\\Codice\\data\\futures_LSEG_VIX.csv", "raw_rows": 5650, "unique_rows": 5451, "exact_duplicates_removed": 199, "first": "2004-03-26", "last": "2025-11-20", "sha256": "c9b2adca1668a26c2bc92961b406b7036574c7347c6cba8aad63efb1eb1bb139"}`
- front_dataset: `{"raw_unique_dates": 5330, "usable_origins": 5000, "context": 64, "front_switch_or_missing_next": 255, "first_origin": "2004-06-28", "last_origin": "2025-05-29", "history": "same expiry across historical column positions; no maturity-curve predictors", "expiry_convention": "vendor expiry fields used as identity; actual exchange calendar not verified"}`
- splits: `{"train": {"n": 3140, "first": "2004-06-28", "last": "2017-08-23"}, "validation": {"n": 931, "first": "2017-08-25", "last": "2021-07-13"}, "test": {"n": 928, "first": "2021-07-15", "last": "2025-05-29"}}`

- Exact duplicate full rows are removed before joins. Conflicting duplicate dates cause a failure.
- The chosen expiry is determined using that day's vendor expiry fields, not future prices.
- Roll transitions are excluded; the vendor expiry-date convention has not been validated against an exchange contract master.
- Data after the end-date cutoff is not used for fitting, validation, contexts, or target outcomes.
- Daily publication timestamps are absent. Interpret forecasts as made after the reported daily values become available; no intraday timing claim is made.
- Comparing different instruments does not establish that one is intrinsically easier: eligibility, class balance and sample size differ.

## Timestral completion

- mistralai/Timestral-5M: completed
- mistralai/Timestral-28M: completed
- mistralai/Timestral-123M: completed
- mistralai/Timestral-209M: completed

## Presentation scope

Use one slide for the coin-versus-model chart and one for nearest-expiry VIX results.
Replace two detailed NQ result slides rather than exceeding ten slides.

Sources: https://www.cboe.com/tradable_products/vix/faqs and supplied local files.
