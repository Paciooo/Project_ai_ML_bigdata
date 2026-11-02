# Coin-flip comparison and nearest-expiry VIX results

Completed 2 November 2026. All four Timestral sizes were run with real weights: 5M, 28M, 123M and 209M. No additional commands are required to obtain these results.

## Main findings

The NQ one-minute classifiers outperform independent coin flips on this historical sample. For example, Run A XGBoost scores 49.26% on all eligible outcomes, compared with a fair-coin mean of 47.97% and a 95% random-policy band of 47.88%-48.05%. Those percentages count flat outcomes as misses, so a fair coin does not automatically average 50%. The majority baseline scores 48.37%.

At daily/session frequency, Run A logistic regression scores 51.46%, inside the fair-coin band of 46.67%-53.33%. Its score is also comparable to a biased random policy with the same expected UP share. This is evidence about these fitted policies on the fixed test data, not proof of future performance.

For nearest-expiry VIX futures, a fair coin averages 49.98% across 10,000 simulated policies, with a 95% band of 46.77%-53.13%. The training-majority direction is DOWN, and always predicting it scores 59.59%. The simple baseline therefore beats a fair coin by a wide margin before a model is introduced.

| Policy / model | VIX test accuracy | Balanced accuracy | Improvement over majority |
|---|---:|---:|---:|
| Training majority (always DOWN) | 59.59% | 50.00% | -- |
| Logistic regression | 59.81% | 50.91% | +0.22 pp |
| XGBoost | 54.42% | 48.58% | -5.17 pp |
| Timestral-5M | 50.00% | 50.15% | -9.59 pp |
| Timestral-28M | 48.28% | 47.33% | -11.31 pp |
| Timestral-123M | 52.05% | 49.77% | -7.54 pp |
| Timestral-209M | 51.83% | 49.58% | -7.76 pp |

All use the same **928 test origins**, from July 2021 through May 2025. Logistic regression was selected over XGBoost on validation balanced accuracy. Its paired 20-day block-bootstrap interval for the accuracy improvement over majority is **-0.97 to +1.29 percentage points**, so the improvement is not established. It predicts UP on only 3.45% of test days, explaining why its overall accuracy is close to the always-DOWN rule. Each Timestral model's accuracy is below majority in the paired block comparison. The small differences among Timestral sizes do not establish a reliable model-size ranking.

The VIX experiment has higher raw accuracy than the NQ experiment because its label distribution differs. These results do not demonstrate that the models forecast volatility better than their appropriate baselines. They also do not demonstrate a profitable VIX futures strategy.

## What was tested

- **NQ:** 10,000 random-policy accuracy simulations against saved predictions from runs A, B and C, across their existing time frames. No NQ models were retrained.
- **Two random controls:** fair independent UP/DOWN decisions, and independent decisions with the same expected UP share as each model. These controls use the same labels and sample size as that model.
- **VIX target:** next observed daily quoted-price direction of the **nearest unexpired futures contract**, excluding a front-contract change between origin and outcome. This is not spot VIX direction, S&P 500 price direction, or realized S&P 500 volatility.
- **VIX history:** 64 daily observations of the SAME expiry. The contract is tracked through its historical column positions before becoming front month. Other maturities are not used as separate predictors.
- **Spot file:** audited as a reference, not used as a predictor in the chosen front-contract-only experiment.
- **Splits:** train before 25 August 2017; validation before 15 July 2021; test after that date. Labels crossing a split boundary are purged. Data after 30 May 2025 is excluded from every feature, context and outcome. There are 3,037 non-flat training observations, 931 validation origins and 928 test origins.
- **Data handling:** removed 165 exact full-row duplicates from the spot file and 199 from the futures file, in memory. No conflicting duplicate dates were found. Originals were left unchanged. The expiry convention comes from the vendor fields; it has not been reconciled to an exchange contract master.

## Monte Carlo interpretation

The simulation keeps historical outcomes fixed and varies the random decisions. The accuracy simulator uses independent binomial counts, mathematically equivalent to tossing a coin for every observation in every run. This avoids allocating billions of individual predictions.

The random-policy bands describe how lucky a coin can be on this fixed history. They are not confidence intervals over future market paths. Beating a fair coin is insufficient when the majority rule already performs well. The matched-UP-rate coin helps expose that issue: VIX logistic regression's conditional tail fraction is about 0.081 against this control, versus roughly 0.0001 against a fair coin. These are conditional random-policy diagnostics, not market-alpha significance tests. A Holm-adjusted diagnostic across the reported comparisons is also saved. Paired chronological block-bootstrap comparisons against majority and a training-median reversion rule are supplied separately.

Illustrative random-sign NQ profit simulations use exactly the model's active times, its recorded fold-specific thresholds, approximate roll exclusions and identical zero/base costs. Economic Monte Carlo is restricted to session frequency for this compact study. One-minute economics is not evaluable at positive latency; corrected longer intraday backtests are in the local NQ reports. Session-level results retain bar-price/fill limitations. No VIX dollar backtest was produced: futures observations lack verified publication times and executable quotes.

## Deliverables

Authoritative complete run:

`results/extensions/runs/20261102T065506Z_d1d861_coinflip_vix_front/`

A small snapshot for GitHub is prepared in `reports/coinflip_vix_front/`: summary CSVs, charts and run metadata only. The full predictions, fitted models and downloaded weights remain local.

- `SUMMARY.md`: detailed findings, source hashes, splits and limitations.
- `coinflip_accuracy.csv`: NQ and VIX random-policy comparisons.
- `coinflip_pnl.csv`: conditional NQ session profit diagnostics and explicit skip reasons.
- `vix_metrics.csv`, `vix_paired_comparisons.csv`: VIX model scores and block comparisons.
- `timestral_status.csv`: completion and resolved checkpoint paths for all four sizes.
- `vix_predictions.parquet`: aligned predictions, expiry identifiers and outcomes.
- `coinflip_comparison.png`, `vix_model_comparison.png`: two presentation charts.
- `run.json`: arguments, source hashes, data audit and environment details.

Earlier extension folders are intermediate runs with missing model weights. Use the complete folder above. The final charts were rerendered with a shorter title; numerical outputs are unchanged.

Local raw-input and cached-weight verification is recorded in `reports/verification/local_verification.json`. Implementation: `tools/coinflip_vix.py`. Regression coverage: `tests/test_coinflip_vix.py`, including nearest-expiry selection, same-contract contexts, roll exclusion, future-data invariance, duplicate handling, purged boundaries and random-policy mathematics. See `docs/AUDIT_FIXES.md` for the final regression-test count.

Optional reproduction, from the existing virtual environment in the project folder:

```powershell
python -u tools/coinflip_vix.py --simulations 10000 --timestral-models 5M 28M 123M 209M
```

Weights are now cached; this command works offline with the saved input files. It creates a new result folder. The script accepts `--spot`, `--futures` and `--nq-runs` for different source locations. Do not move the originals or re-download data. This is an additional script; the existing experiment engine and prior results were preserved.

## Revised presentation outline: maximum 10 slides

1. Research question: direction prediction versus useful trading evidence.
2. Data: NQ minutes and nearest-expiry VIX futures, with target definitions.
3. Controls: majority, persistence/reversal, synthetic null and coin flips.
4. Method: chronological splits, model families and block uncertainty.
5. NQ results across candle time frames.
6. Candle-body and walk-forward checks, combined on one slide.
7. Monte Carlo comparison: model versus random decisions and majority.
8. VIX results: logistic regression, XGBoost and all four Timestral sizes.
9. Economic interpretation and remaining execution/data limitations.
10. Conclusion: short-horizon NQ signal, no demonstrated VIX-model advantage over majority, profitability unproven.

## External context

Cboe describes VIX as a constant 30-day expected-volatility measure derived from SPX options. VIX futures are distinct instruments based on it. Forecasting nearest-expiry futures is not a direct test of realized S&P 500 volatility.

- https://www.cboe.com/tradable-products/volatility-trading/
- https://www.cboe.com/tradable_products/vix/faqs
