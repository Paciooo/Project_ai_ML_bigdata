> Historical review/prompt. Current fixes and verified results are documented in `docs/AUDIT_FIXES.md` and `reports/README.md`. Findings below describe the earlier code, not the present repair status.

# Project review: next-candle direction prediction

Reviewed 2 November 2026. Local checkout: d3c4ca2; config.py has an existing local change setting DATA_SOURCE to nq_ticks. This review does not establish what fixes exist upstream.

## Verdict

This is a useful university research project and a reasonable starting point for investigating predictability. It is not yet a validated trading system. The saved results suggest a small conditional classification advantage at one-minute frequency; they do not demonstrate positive returns after costs, robust future performance, or executable next-tick predictions.

The immediate priority is reliable data, explicit prediction timing, and honest evaluation. Larger models should come after those improvements. Claude Code having written the code is not itself a reason to trust or reject it; independent checks of behavior and research design matter.

## Scope and evidence

Inspected all 11 project Python files, requirements, README, saved CSVs, local tick files, and download logs. Read all seven pages of the supplied paper, including visual inspection of pages 5 and 6 containing results and profit tables. Executed read-only data checks and small reproductions of timing, cleaning, split, and downloader behavior. Python parsing and a small in-memory classifier smoke check passed.

Did not run the multi-million-row training experiments, connect to IBKR, download model weights, or pull Git changes. The historical futures-data repository is absent from the surrounding workspace, so the 2006-2025 results were inspected as saved outputs rather than independently regenerated. No application source or existing results were changed.

## What is already good

- Synthetic GBM is a useful negative control, repeated on five independent simulated paths. Finding no stable advantage is an informative result.
- Features use current/past closes; scaling is fitted inside training pipelines.
- Train/validation/test are chronological; the code selects the model by validation balanced accuracy.
- A majority-class baseline is trained on training labels and evaluated on the corresponding test examples.
- Logistic regression, trees, random forest, small MLPs and XGBoost provide sensible complexity comparisons.
- Modules are short and readable. The Timestral comparison is a separate script, and the IBKR downloader is separate from both model pipelines.
- IBKR downloading has pacing, reconnects, logging and partial checkpoints. Loading ignores partial files and preserves equal-timestamp row order.

These strengths do not remove the specific limitations below.

## What the saved results say

| Experiment | Model | Test examples | Accuracy | Majority accuracy | Difference |
|---|---|---:|---:|---:|---:|
| Historical NQ, 1-minute | Logistic regression | 1,318,336 | 51.32% | 50.42% | +0.91 percentage points |
| Historical NQ, 1-minute | NN 3 layers, selected on validation | 1,318,336 | 51.41% | 50.42% | +1.00 percentage points |
| Historical NQ, hourly | Decision tree, selected on validation | 23,769 | 50.89% | 51.75% | -0.86 percentage points |
| Historical NQ, daily | XGBoost, selected on validation | 1,002 | 50.20% | 53.29% | -3.09 percentage points |
| Timestral-5M, 1-minute subsample | Timestral | 20,000 | 50.50% | 50.47% | +0.04 percentage points |

Values are rounded; differences use the saved comparison fields. The Timestral row is a different sample from the full supervised-model test set. The smaller 2,000-example Timestral run scored 49.9%; the 20,000-example run offers much better precision. Neither establishes a useful advantage. Only 5M Timestral results are present locally. Claims about other sizes remain untested here.

The one-minute supervised models are close together. The saved difference between the deeper network and logistic regression is roughly 0.09 percentage points. There is no strong practical case yet for preferring complexity.

## Priority findings

### 1. The target needs a precise definition

features.make_target predicts close[t+1] > close[t], dropping equal closes. This is next close relative to the current close. A green next candle usually means close[t+1] > open[t+1], which can disagree, especially across gaps. Raw tick mode predicts the next trade price, not a candle.

Decide the primary objective explicitly. Record feature availability, decision time, entry time and label end time separately. Keep any alternative target as a separately named experiment.

Dropping flat outcomes is valid for the conditional question 'up or down given that the next close changes'. It cannot establish unconditional live accuracy because you cannot know beforehand which next bars will be flat. Preserve the existing benchmark, report coverage, and evaluate all eligible outcomes for deployment and P&L.

### 2. Cleaning depends on the future

data.py:189-190 removes a spike using the following return. In a three-price fixture, changing only the third price changed whether the second row survived. It also changes adjacent labels and time spacing.

This is retrospective cleaning, not information available at the candidate decision time. It does not prove the entire observed one-minute advantage comes from leakage; the README reports only five removed historical spikes. It does mean a strictly causal claim is not established. Use externally verified corrections with provenance, or an explicit delayed/causal policy, and report sensitivity to uncorrected data.

### 3. Candle labels are not availability times

data.py:209 uses pandas' default hourly resampling although historical minute bars are documented as end-stamped. A fixture showed an hourly row labeled 10:00 containing a 10:59 close. The 10:00 endpoint minute is also assigned to the subsequent hour under the current bins.

Tick-derived minute/hour bars are likewise labeled at the start of their aggregation interval. Daily rows are labeled by session day rather than actual completion. This does not by itself prove all classification scores are invalid: the sequence of finished closes can still be evaluated correctly. It does make index-based availability claims and potential execution timing incorrect, and mixed-frequency split boundaries need auditing.

Use explicit start/end/availability times and source-specific bin closure. Define CME sessions using an exchange-aware timezone policy; split by availability and label endpoints. Do not 'fix' all sources with one blind resampling change.

### 4. Downloader errors can become empty successes

tools/download_ticks_ibkr.py:156-164 retries an empty response associated with an error three times, then accepts the fourth empty error response. A mocked IB instance reproduced four error 10187 responses returning []. download_day can then advance to the next active minute and eventually save a final parquet file.

This is a correctness bug even if the network recovery improvements work. Distinguish request failure from confirmed no-data, correlate errors to request IDs, and leave unresolved ranges incomplete. A completed day needs an explicit coverage manifest, not just an existing filename. Checkpoints/final files should be atomic.

The log ends with error 10187 and then a saved day. That alone does not prove that this specific day lost trades, but it contradicts treating the logs as proof of error-free completeness. check_ticks reports no gaps greater than three minutes; that cannot detect missing trades within seconds or shorter gaps.

IBKR documents that a historical tick request may return more than the requested count to finish a whole second. The downloader's comment that a second exceeding 1,000 necessarily loses its remainder is therefore misleading. Its overlapping-page branch can flag a fully repeated second as 'possibly incomplete'. Test same-second pagination and do not discard legitimate trades merely because timestamp, price and size match.

### 5. The tick sample is large in trades, short in market history

Local files contain 8,773,033 trades across 25 UTC-date files, 21 September through 19 October 2026. The current grouping produces 22 session labels; the first and last sessions are partial. Approximately 86.77% of adjacent within-file trades share the same second; 23.79% have an unchanged next trade price.

Using the current split logic, validation starts 8 October 2026 UTC and test starts 15 October 2026 UTC. The resulting test contains only 4,077 usable one-minute examples and 68 hourly examples. The midnight boundaries also cut through CME sessions. Millions of related events in a few weeks do not establish robustness over many independent market conditions.

Preserved same-second API order can support an event-sequence experiment. It does not establish subsecond observation times or achievable fills. Collect bid/ask observations for spread-aware analysis and more sessions/contracts for robustness.

### 6. Accuracy is not profitability

There is no trading simulation or measured execution-cost model in the project. Accuracy weights a small correct move the same as a large wrong move. It also ignores spread, commissions, slippage, latency, turnover and drawdown.

Expected net P&L per trade is p * average winning amount - (1-p) * average losing amount - costs. Under an illustrative equal-win/equal-loss assumption, 51.41% accuracy produces only 2.82% of that common move size before costs. This is not a measured project return.

For NQ, the contract multiplier is $20 per point and the outright minimum tick is 0.25 points ($5). Supply actual broker/account costs rather than assuming a generic number. Use bid/ask fills when available; bar-only cost scenarios must be labeled approximations. The decision occurs after the input bar is available; filling retrospectively at its close is optimistic.

### 7. Uncertainty and validation need strengthening

evaluation.py:19 computes an independent-binomial accuracy interval. That is a rough diagnostic, not a paired test of improvement over a baseline, and the code does not assess dependence in prediction errors. Use paired resampling in chronological blocks, report sensitivity to block size, and report fold/day outcomes. With only a few test sessions, acknowledge that dependence-aware inference is itself unreliable.

One chronological holdout is useful, but it does not establish stability. Add expanding/rolling walk-forward evaluations with selection restricted to each fold's training/validation history. Because current test results have already been inspected, a final confirmatory claim needs genuinely untouched future data. Repeatedly trying models on the same visible test set turns it into development data.

The fixed supervised training period ends in 2017; validation and test span subsequent years without retraining. Compare that deliberate fixed-model experiment with a separately defined rolling retraining policy.

### 8. Timestral comparisons need matched examples and provenance

The script correctly slices historical contexts ending at the forecast origin. Its median extraction matches the installed (batch, variate, horizon) output convention. No obvious forecast-horizon shift bug was found there.

However, changing n_samples changes the evaluated examples, and RNG consumption also depends on which frequencies run. Save a shared timestamp manifest and score all model sizes and local models on those same examples. Save per-origin predictions, configuration, data hash, model revision, package versions and timing. A single appended metrics CSV is insufficient to reproduce comparisons.

Independent-binomial 95% accuracy half-widths near 50% are approximately 3.1 percentage points for 1,000 samples and 4.4 for 500. These are not confidence bounds for the difference between two models. Such runs cannot reliably resolve the roughly one-point advantage seen in the supervised one-minute experiment.

The local Timestral script contains no HF_HUB_DISABLE_XET assignment. The pasted description may refer to a newer upstream commit. Add the workaround before importing Hugging Face-dependent code. Do not discard config.py just to pull; preserve the existing local choice.

Mistral's generic forecasting benchmarks are not NQ directional trading benchmarks. Its model card documents quantile forecasting, but generic forecasting quality does not establish profitable sign prediction. The project's records do not establish whether pretraining overlapped this historical evaluation data; document that as unknown unless verified.

### 9. Features and engineering

All eight features come from closes. Volume, intrabar ranges, candle bodies, session time, spread and order flow are unused. Add a small causally available feature set and measure incremental value with ablation; do not assume more indicators create an edge. At tick level, second-resolution timestamps mean elapsed time can be zero, so blindly dividing by sqrt(delta time) is unsafe.

The loader silently omits missing requested historical years. There are no checked-in tests, dependency lock, run manifests, saved predictions or saved fitted models. Invalid DATA_SOURCE values silently fall through to synthetic mode. Re-running a given experiment overwrites its tables, despite separate folders protecting different experiments. Older synthetic results live directly in results/, not results/synthetic/.

features.calendar_split also mishandles gaps larger than a split: negative iloc stop indices keep rows instead of rejecting the split. A fixture with two rows and gap=3 incorrectly retained one row. Default gap=1 on this dataset is not affected, but minimum-sample validation is needed.

## The supplied paper

Kyong Joo Oh and Kyoung-jae Kim, 'Analyzing stock market tick data using piecewise nonlinear model', Expert Systems with Applications 22 (2002), 249-255.

The proposed model is specific: choose lag count using a chaos/embedding analysis; segment historical daily KOSPI 200 using Pettitt change-point tests and binary segmentation; use a backpropagation neural network to predict the daily regime; then use a regime-specific backpropagation network for one-minute forecasts. The basic model uses five lagged inputs. This is a regime-based ensemble of small neural networks, not evidence for a modern large transformer. Its 'tick' forecasts are one-minute observations, not the raw next transaction.

Data covers 3,069 daily observations, January 1990-August 2000. Training ends March 2000. Results compare two networks on ten randomly selected test days from April-August 2000. Table 1 reports total MAPE falling from 1.530% to 0.799%. Table 3 reports 42.71 profit points for the proposed method.

Treat those as hypotheses, not a transferable NQ trading edge:

- Ten selected days in one historical market are weak evidence of future robustness.
- Lower price-level RMSE/MAPE is not the same as better next-candle direction or executable profit. A last-price forecast baseline is essential for level forecasts.
- The article mentions trading fees but does not provide enough quantitative spread/slippage/latency detail to reproduce realistic net returns.
- Table 3 has negative point entries with positive currency amounts; its total dollar amount also differs from the approximate amount in the text. These are reporting inconsistencies, not proof the strategy fails, but they reduce confidence in reproducing profitability.
- The reported paired t-tests do not establish dependence-robust trading performance.
- For implementation, regimes inferred from the whole test history or the current day's unfinished daily close would leak future information. Fit historical segmentation inside training only, and use completed data for inference. Do not assume chaos or five lags transfers to NQ.

The most useful experiment is pooled logistic/XGBoost versus the same models conditioned on causally available regimes, plus a small paper-inspired MLP ensemble. Determine regime count, minimum segment size and feature choices on training/validation only. A causal online change detector is another possible research branch. Retain a pooled fallback when regimes have too few observations. More complex sequence/order-book architectures should wait until relevant data and sample length exist.

## Sources

- Local implementation, CSVs, log and tmp/review_checks.json provide the project-specific evidence.
- Supplied paper: intresting paper/1-s2.0-S0957417401000586-main.pdf.
- Mistral model card: https://huggingface.co/mistralai/Timestral-209M
- Mistral inference source: https://github.com/mistralai/timestral
- IBKR historical Time & Sales: https://interactivebrokers.github.io/tws-api/historical_time_and_sales.html
- CME NQ contract specifications: https://www.cmegroup.com/markets/equities/nasdaq/e-mini-nasdaq-100.contractSpecs.html
- Adams and MacKay, Bayesian Online Changepoint Detection: https://arxiv.org/abs/0710.3742
- Bailey et al., The Probability of Backtest Overfitting: https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf
