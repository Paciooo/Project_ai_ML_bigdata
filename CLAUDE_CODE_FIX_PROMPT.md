# Prompt to paste into Claude Code

Audit and improve this repository as a research engineer. The objective is reliable out-of-sample prediction of the next completed NQ candle, with a separate test of economic value after realistic costs. Do not promise profitability or optimize toward a chosen headline accuracy.

First read PROJECT_REVIEW.md, the existing code, results, and the paper in `intresting paper/1-s2.0-S0957417401000586-main.pdf`. Independently verify the review's findings. Work incrementally, starting with correctness. Preserve existing raw data, result files, and the user's local config.py changes. Do not run git restore/reset, overwrite existing experiments, trade, or connect to a live brokerage account. Use mocked IB responses for automated checks. Do not launch full training or model-weight downloads as smoke tests. Keep the code small and understandable for a university project.

Implement the following in priority order, with concrete changes and meaningful regression tests. If external data or model weights are unavailable, implement and test offline behavior and clearly report the unverified integration.

## 1. Define target and information timing

- Name the current target `next_close_direction`: sign(close[t+1]-close[t]). Raw tick mode predicts the next trade, and must be named separately.
- Add `next_candle_body_direction` as a separately configured alternative if useful: sign(close[t+1]-open[t+1]), decided before that candle's open. Do not silently replace the old target.
- Store bar start/end, feature availability time, decision time, target start/end and session/contract identity. Verify that every feature is available by the decision time and that labels and fills occur afterward. Account for inference latency.
- Preserve flat next outcomes. Retain the legacy conditional non-flat benchmark with explicit coverage, and add evaluation over all eligible outcomes with a documented flat/three-class policy. Use all realized outcomes, including flat ones, in any P&L calculation. Do not remove flat rows before shifting the target, which would change the horizon.
- Define scheduled-session breaks and missing-data behavior; never silently call the next available observation 'one minute later' across an outage or overnight gap.

## 2. Fix data availability and validation

- Replace implicit resampling conventions with explicit source-specific interval closure and labels. Historical TradeStation minutes are documented as end-stamped; IB trade timestamps are event times. Test exact boundary trades and end-stamped minutes. An hourly row labeled 10:00 must not be treated as available at 10:00 if its close came from 10:59.
- Use timezone-aware source timestamps and an explicit CME session policy in America/Chicago or America/New_York, including daylight saving transitions and partial sessions. Separate session identifiers from bar availability times. Do not assume the same bin settings suit both sources.
- Replace future-dependent spike removal with a documented causal/delayed policy or independently verified corrections with provenance. Quarantine suspicious data rather than silently using a future reversal. Make the cleaner's causality policy explicit and test prefix invariance where it claims online availability. Provide a sensitivity comparison for verified corrected versus raw data.
- Validate OHLC relationships, finite/positive prices, nonnegative sizes, monotonicity, minute timestamp uniqueness, requested-year coverage and missing ranges. Equal tick timestamps are allowed; identical tick records can be legitimate trades and must not be blindly deduplicated.
- Split using availability times and label endpoints, with complete-session grouping when requested. Purge training/validation labels whose outcomes cross the next boundary. Past-only feature history may cross a boundary legitimately. Validate fractions, nonempty splits, gap values and adequate sample sizes; reject gap >= split size instead of using negative iloc stop indices.
- Carry contract and continuous-series adjustment metadata. Keep actual-contract tradable prices separate from back-adjusted features and handle rolls explicitly.

## 3. Fix IBKR download correctness

- In request(), an empty response accompanied by an unresolved error must never become successful no-data after retries. The current behavior accepts the fourth empty error response. Use typed outcomes/errors, request-ID-specific correlation, bounded retry/backoff and explicit incomplete ranges. Distinguish known closed-market/no-data intervals from entitlement, session, connection, pacing and permission failures.
- Implement pagination consistent with IBKR's full-second completion behavior: responses may exceed the requested 1,000 ticks. Preserve same-second sequence and valid repeated trades, handle repeated pages and lack of progress, and resume without loss or duplication. Do not infer that a repeated full second proves truncation.
- Persist a download manifest containing contract conId/month, UTC ranges, request/coverage status, counts, unresolved errors and schema/version. Mark a day complete only when coverage checks pass. Existing filename alone is not evidence of completeness. Existing files without manifests are unverified until checked.
- Use atomic checkpoints and final writes. A failure must leave resumable state and must not promote a partial day to final. Retain trade attributes/exchange/conditions when available, documenting any filtering.
- Add mocks for repeated 10187, disconnects, timeouts, holiday/no-data, repeated boundary seconds, >1,000 trades within a second, restart/resume, and failed coverage. Extend check_ticks to consume manifests; no >3-minute gaps is only one diagnostic.

## 4. Make experiments reproducible

- Add CLI arguments such as --data-source, --target, --frequency, --seed and --smoke so users need not edit tracked config.py. Validate choices and never silently fall back to synthetic for an unknown source. Resolve paths predictably from the project/config location.
- Save each run under a unique run directory inside its data-source folder. Preserve old outputs. Save complete configuration, Git commit plus dirty-state indicator, dependency versions, source file hashes, timezone/bar conventions, contract metadata, split boundaries, target and model parameters.
- Save per-origin labels, predictions, probabilities where supported, and availability/label times. Save fitted model/preprocessing artifacts where applicable. Add a reproducible dependency lock or tested pinned environment and optional dependency groups for IBKR, Timestral and XGBoost.
- Test the installed interfaces. Do not invent a model/API method based on memory. Add offline smoke runs that finish quickly and never overwrite real results.

## 5. Strengthen evaluation before adding models

- Keep chronological holdouts, and add configurable rolling/expanding walk-forward evaluation. Fit preprocessing, feature choices, regime models, hyperparameters, probability calibration and thresholds using each fold's training/validation history only. Log retraining policy.
- Compare majority, persistence and reversal rules with logistic regression, random forest, small MLP and XGBoost on identical rows. Specify zero-return handling for the simple rules. Use last-value/no-change forecasts as baselines for regression metrics.
- Report accuracy, balanced accuracy, class counts, confusion matrices, macro F1, and probability calibration/Brier/log loss where probabilities are meaningful. Keep existing metrics for compatibility.
- Add paired chronological block-bootstrap intervals for model-minus-baseline accuracy and net P&L; make block definitions explicit and report sensitivity. Do not label independent-binomial intervals as tests of an edge. Report day/fold/year/session results, not just one aggregate. If there are too few independent sessions, mark the result inconclusive.
- Treat already inspected historical test results as development evidence. Define a protocol for a new untouched forward holdout before confirmatory claims. Keep a record of model/feature/threshold trials and distinguish exploratory from confirmatory results.
- Preserve the synthetic null experiment and add a small positive-control fixture proving correct alignment can recover a deliberately injected signal. Tests must catch leakage/timing errors rather than assert that market accuracy exceeds 50%.

## 6. Make Timestral comparisons fair

- Set HF_HUB_DISABLE_XET=1 before any Hugging Face-dependent import when using the workaround; respect an explicit user override if appropriate and document it. The reviewed local script lacks this claimed fix.
- Save a common forecast-origin manifest and evaluate Timestral-5M/28M/123M/209M plus supervised baselines on those same origins, context rules and target. Use a common comparison sample when compute budgets differ; separate additional larger-sample runs. Save per-origin predictions and model revisions.
- Do not change sample membership just because --frequency all versus a single frequency changes RNG consumption. Validate positive context/sample/batch arguments and handle empty eligible samples clearly.
- Report inference time, device and uncertainty. Runs with 500 or 1,000 examples cannot reliably rank models separated by about one percentage point.
- Keep the close-median sign forecast as a named baseline. Optional quantile-based direction/confidence experiments require a documented mapping and validation-only calibration; quantiles are not automatically exact probabilities of an up move.
- Record pretraining overlap/provenance as unknown unless established by reliable documentation. Do not equate general forecasting benchmark rankings with profitable NQ trading.

## 7. Add a transparent economic test

- Implement an offline backtest consuming timestamped out-of-sample signals. Specify decision, entry and exit rules before evaluation. Never fill at an input close that was already needed to compute the signal, or at an unavailable future price.
- Model NQ's $20-per-point multiplier and 0.25-point tick, configurable per-side commissions/fees, bid/ask spread, slippage, inference/execution latency and position-change turnover. Avoid double-counting spread when fills already use quotes. Use actual account costs when supplied; otherwise label scenarios as assumptions.
- Quote-based execution is preferred. For bar-only data, make limitations explicit and provide conservative cost sensitivities. Same-second historical ticks do not establish subsecond executable fills. Do not invent bid/ask data or assume a limit order filled merely because a bar touched it.
- Support no-trade thresholds selected on validation only. Evaluate all subsequent outcomes and coverage; a future flat outcome is not an entry filter. Define how consecutive signals, reversals, overlapping trades, rolls and session closures affect positions and costs.
- Report gross/net P&L, turnover, trade count, exposure, mean win/loss, expectancy, drawdown, daily returns and suitable uncertainty. Report capital assumptions before percentage returns. Compare against no-trade and the same execution rules applied to simple baselines. Cost stress tests must not be tuned on the final holdout.
- Do not add live order submission.

## 8. Add one controlled paper-inspired experiment last

- The paper uses five lagged price inputs, Pettitt/binary segmentation of historical daily data, a neural-network regime classifier, and regime-specific one-minute neural networks. Implement a clearly labeled adaptation to NQ direction, rather than claiming an exact replication on different data and targets.
- Fit segmentation only within each training fold. Select lag count, regime count and minimum segment size on training/validation. Predict regimes from causally available completed data; never use today's final daily close to route that day's intraday predictions. Do not segment test history retrospectively or use smoothed states that see the future.
- Compare pooled versus regime-conditioned logistic regression/XGBoost and one small MLP ensemble, with identical features, evaluation origins and execution assumptions. Include a pooled fallback for sparse regimes. Avoid launching a large architecture search.
- Add a small ablation for volume, range/body, past realized volatility and session-time features where available. Record any trade-sign estimation as an estimate. Order-book models require genuine order-book data, not just trade price/size.
- Report whether regimes improve paired out-of-sample classification and net economics. A null result is acceptable. Do not assume chaos, five lags or the paper's historical profits transfer to NQ.

## 9. Compare candle time frames on futures

- Extend the bar experiment to a configurable, predeclared set: 1 minute, 5 minutes, 15 minutes, 1 hour, 4 hours and one full CME trading session. Keep raw next-trade prediction as a separate task. Avoid a broad search over arbitrary bar lengths.
- Prefer one validated long-history minute dataset to construct all these time frames on common date coverage. The existing saved NQ experiment has 5,017 daily bars, but its source futures-data repository was absent from the reviewed workspace. Detect that absence clearly; do not claim historical results were regenerated or synthesize replacement market data. The recent tick sample yields only 22 session labels, including partial boundary sessions, and cannot support meaningful daily model training.
- Add a separate historical-bar downloader if IBKR is used. Request bars directly through reqHistoricalData rather than downloading all underlying ticks. Validate availability and entitlements. Current IBKR documentation limits individual expired-contract history to contracts expired within two years; continuous futures have separate constraints, including an empty endDateTime. Do not promise reconstruction of 2006-2025 history from IBKR without verifying the actual returned coverage.
- Define session anchoring, regular-hours versus extended-hours scope, daylight saving transitions, overnight gaps and shortened sessions. Explicitly handle the final incomplete intraday bin when a session length is not divisible by four hours; do not silently mix partial candles with full candles. Compare time frames over identical walk-forward test dates, while allowing their observation counts to differ.
- Distinguish last-trade session close from official settlement. IBKR daily futures closes can be settlement prices published after session end. A settlement-based feature is unavailable until publication; enforce availability or use a documented last-trade construction. Do not mix daily sources silently.
- Report sample counts, flat outcomes, paired improvement over each time frame's own baseline, block-aware uncertainty, net P&L, turnover and drawdown. Raw accuracies alone cannot rank different horizons with different class balance. Treat time-frame selection as model selection inside training/validation; retain a final untouched holdout.
- Start with logistic regression, XGBoost and the majority/persistence/reversal baselines; add the existing MLP and Timestral comparisons only after timing and alignment checks pass. Express feature lookbacks explicitly in bars and elapsed/session time; compare a controlled small set rather than assuming ten bars represent the same horizon at every frequency.
- Separately test whether completed hourly/daily features improve an intraday target using backward availability-time joins. That is a different experiment from predicting the next daily candle. Add tests preventing use of an unfinished daily bar or later settlement revision in an earlier intraday decision.

Finish by running focused regression tests and offline smoke runs, then updating README to distinguish implemented behavior, legacy results, fresh validated runs and unverified integrations. Report changed files, checks performed, remaining limitations, and exact commands for a small offline run and a later full run. State plainly whether any new evidence supports a profitable edge; do not infer it from accuracy alone.
