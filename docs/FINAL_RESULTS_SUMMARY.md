# Results summary for a maximum of 10 slides

Follow-up: the coin-flip Monte Carlo and nearest-expiry VIX experiments, including all four Timestral sizes, are complete. See `docs/MONTE_CARLO_VIX_RESULTS.md` for the combined findings and revised 10-slide outline. The NQ results below remain the original four-run summary.

Updated 2 November 2026. All four planned local runs finished. No additional model-size or feature search is needed for the presentation. Numerical results are exploratory: the historical test period has already been inspected.

## 1. Research question

Can models predict NQ futures' next-candle direction better than simple rules, and does the answer change with candle length? Treat economic value as a separate question requiring an executable strategy and measured or explicit assumed costs.

## 2. Data and scope

The primary source is 6,193,869 historical NQ one-minute observations from 2006 through May 2025, stored as yearly Parquet files. All larger candles are constructed from that history. The recent IBKR tick download is separate; it was not used in runs A, B, C or Timestral below. Daily/session candles follow the project's session conventions and use last-trade closes, not official settlement. Continuous-contract adjustments, inferred schedules and approximate roll dates limit economic interpretation.

## 3. Targets and validation

Run A predicts sign(next close - current close). Run B predicts sign(next close - next open). Equal-price outcomes remain in the dataset; acc_all counts them as misses for these binary models, while acc_nonflat is conditional on a nonzero move. These metrics must not be mixed. Chronological train/validation/test splits use session grouping and label purging. Preprocessing is fitted on training data. Run C uses four expanding walk-forward test blocks. The scripts report paired session-block bootstrap intervals; these do not make the already inspected period confirmatory.

## 4. Models and controls

Compare logistic regression and XGBoost with majority, persistence and reversal rules. Include Timestral-5M as a separate pretrained NQ comparison and all four sizes on VIX. Synthetic smoke outputs and tiny RANDOM Timestral outputs are functionality checks, not market evidence. The existing legacy GBM experiment can illustrate a null control, clearly labeled as the earlier pipeline. The local suite passes 70 tests after the repairs; tests do not establish market profitability.

## 5. Run A: time-frame comparison

XGBoost's accuracy improvement over the majority rule, on all eligible outcomes, in percentage points:

| Time frame | Run A | 20-session block interval |
|---|---:|---|
| 1 minute | +0.89 | +0.71 to +1.07 |
| 5 minutes | +0.74 | +0.45 to +1.02 |
| 15 minutes | -0.28 | -0.70 to +0.14 |
| 1 hour | -0.35 | -1.04 to +0.36 |
| 4 hours | -2.20 | -4.11 to -0.18 |
| Daily/session | -2.69 | -5.85 to +0.47 |

The intervals support small classification gains at 1 and 5 minutes under this protocol. They do not show an advantage from moving to longer candles. Samples differ by eligibility: for example the session comparison has 855 eligible test sessions, versus 1,002 session groups at minute frequency. On non-flat one-minute outcomes, XGBoost accuracy is 51.35% versus 50.42% for majority. This is a different denominator from acc_all.

## 6. Run B: predicting candle bodies

| Time frame | XGBoost improvement over majority, pp |
|---|---:|
| 1 minute | +0.40 |
| 5 minutes | +0.63 |
| 15 minutes | -0.10 |
| 1 hour | -0.86 |
| 4 hours | -0.97 |
| Daily/session | -2.69 |

The one-minute gain is smaller for candle-body direction. This supports distinguishing the two targets; it does not prove a specific microstructure mechanism. A reversal baseline also performs relatively well at short horizons, which is consistent with microstructure effects but does not establish bid-ask bounce as the sole cause without further evidence. The decision-delay limitation in slide 9 also prevents calling the current body experiment a fully validated pre-open trading forecast.

## 7. Run C: walk-forward comparison

XGBoost's pooled accuracy improvement over majority is +1.00 pp at one minute (interval +0.81 to +1.18), +0.13 pp at 15 minutes, -0.36 pp hourly, and -2.22 pp at session frequency. The last three intervals include zero. Logistic regression has a +0.52 pp one-minute improvement. These are pooled out-of-sample comparisons across four blocks, not proof that each block or each future regime has the same advantage. Fold-specific thresholds are now applied in the corrected economic outputs; the execution limitations in section 9 still apply.

## 8. Timestral-5M on matched origins

Accuracy on all eligible sampled outcomes:

| Time frame | Origins | Timestral | Majority | Difference, pp |
|---|---:|---:|---:|---:|
| 1 minute | 2,000 | 48.00% | 48.00% | 0.00 |
| 1 hour | 2,000 | 51.90% | 51.85% | +0.05 |
| Daily/session | 855 | 51.70% | 52.75% | -1.05 |

All Timestral difference intervals include zero. This experiment detects no improvement over majority. The matched supervised baselines in the Timestral CSV use the same sampled origins; do not compare Timestral's 2,000-row score directly against the full one-minute test score as if sample uncertainty were equal. Only 5M was tested with real weights; no conclusion about the other model sizes follows. Pretraining overlap with this history remains unknown.

## 9. Corrected economic interpretation

The entry-timing, fold-threshold and initial-drawdown defects are fixed.
Saved classifier predictions were verified against their models and were not
changed. Validation thresholds were reselected from saved models under the
corrected execution policy; no classifier was refitted. Updated outputs and
provenance are in `reports/local_run_A_holdout_close/`,
`reports/local_run_B_holdout_body/` and `reports/local_run_C_walkforward/`.

One-minute P&L is not evaluable with positive latency and only minute prices.
Other time frames use the first observed minute open after the decision and
before target exit. These remain bar-price proxies with assumed costs, not
verified executable bid/ask fills. Drawdown is end-of-day and includes zero
initial equity. No learned strategy has a positive lower daily-P&L interval under both base and high costs.

Cloud A is superseded and its numbers must not be mixed with local A.
See `docs/AUDIT_FIXES.md` and `reports/verification/local_verification.json`.

## 10. Conclusion

With the tested close-derived features, small short-horizon classification advantages exist in this exploratory historical evaluation. Longer candles and Timestral-5M show no demonstrated improvement over their baselines. The study does not establish a profitable strategy. Its useful contribution is the distinction between statistical direction prediction and executable trading value, including the limits of the current implementation.

## Local evidence

- Run A: results/nq/runs/20261102T060506Z_892e22_A_holdout_close/
- Run B: results/nq/runs/20261102T060756Z_a695aa_B_holdout_body/
- Run C: results/nq/runs/20261102T061204Z_cdd340_C_walkforward/
- Timestral: results/nq/runs/20261102T061421Z_ef43c1_timestral_Timestral-5M/
- Key files: metrics.csv, paired_vs_majority.csv, backtest.csv, folds.csv, run.json, timestral_results.csv and predictions_1min.parquet.
- Current implementation evidence: `backtest.attach_execution` (entry timing), `experiment.run_frequency` and `backtest.frame_positions` (per-row fold thresholds), `backtest.summarize` (drawdown including initial equity), and `tools/repair_results.py` (verification and rescoring of saved runs).
