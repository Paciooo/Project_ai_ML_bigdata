# Canonical local run B

This snapshot comes from `20261102T060756Z_a695aa_B_holdout_body` on Windows, not the cloud replica.
Classification predictions and fitted models are unchanged. `rescore.json`
records the corrected execution policy and validation threshold recalculation.
`run.json` records the original training code/environment; it is not rewritten
to pretend those models were fitted with the repair code.

Use `backtest.csv` only with its `economic_status` field. One-minute signals
with a positive latency have no observed minute-open entry before the label
closes, so their economics are explicitly not evaluable. Other frequencies
use bar-price proxies with assumed costs, not guaranteed executable fills.
Drawdown is end-of-day and includes the initial zero-equity point.

`paired_vs_reversal.csv` is an exploratory paired classification comparison,
not proof of bid-ask bounce or a trading edge. Large predictions and fitted
models stay in the local result folder; raw market data is not copied here.
