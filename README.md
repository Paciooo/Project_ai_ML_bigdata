# What Can Machine Learning Predict in NASDAQ Markets?

University project (Waseda University, *AI, Machine Learning and Big Data for
Finance and Management*; assignment: data analysis with AI, machine learning
and/or big data, judged on originality, usefulness and feasibility). Author:
Alessandro Pacifici.

**Problem.** Markets are hard to forecast, but not every characteristic of
the market need be equally hard. **Research question:** which characteristic
of the next NASDAQ futures candle — its return, its direction or its realized
volatility — can machine learning predict, and on which candle length?

**Originality.** Experimental: one pipeline, one data set and one test period
for three targets and five horizons (5 minutes to 1 day), so "what is
predictable" is compared like for like across tasks. Applied: machine
learning and a new time-series foundation model (Mistral AI's Timestral) are
set against the classical HAR benchmark for realized volatility (Corsi, 2009).

**Usefulness and implications.** Realized-volatility forecasts are inputs to
risk limits and position sizing; the study shows on which horizons machine
learning adds to HAR and where it adds nothing. The study is purely
statistical: it contains no trading rules, P&L, costs or option strategies,
and it does not show that any forecast is profitable.

* Deck (cover + 9 slides): [presentation/NASDAQ_multitarget_report.pptx](presentation/NASDAQ_multitarget_report.pptx)
  ([PDF](presentation/NASDAQ_multitarget_report.pdf)); technical appendix A1–A5
  (reproduction, selection and statistics, sample checks, Timestral on the
  same sample, data profile): [presentation/NASDAQ_multitarget_appendix.pptx](presentation/NASDAQ_multitarget_appendix.pptx)
  ([PDF](presentation/NASDAQ_multitarget_appendix.pdf)).
* Result tables and charts: [reports/multitarget_final/](reports/multitarget_final/).
* Earlier phases of the project (next-candle direction with long/short
  trading tests, IBKR tick tools, the first decks) are kept in Git on the
  branch **`archive/nq-direction-study`**; they are no longer part of the
  main project.

**All results are exploratory.** The 2021–2025 test period was examined in
earlier phases of the project, and the volatility selection score was revised
after a first report had printed test numbers (see "Selection"). Nothing
here is a blind, confirmatory test.

## Results (run `20261112T050812Z_82953a_final`)

**Answer (exploratory): realized volatility is the most predictable
characteristic in this study, on every candle length tested. For next-candle
returns no improvement over forecasting zero was demonstrated with the
models, data and horizons tried. For next-candle direction a small accuracy
gain was measured on 5-minute candles only, and it is not stable across the
test years.**

Metric conventions. For volatility, the "error cut" (an out-of-sample R²) is
the reduction of the **squared log-RV forecast error** relative to a
benchmark: 40.0% vs persistence for the 5-minute DNN means its squared error
is 40% smaller. This is not the RMSE reduction (RMSE 0.901 vs 1.163, i.e.
22.5% lower) and not the gain over HAR-OLS (6.8% smaller squared error than
HAR-OLS). For returns, R² is relative to the zero forecast; for direction,
the gain is accuracy minus the majority-class accuracy, in percentage points.

### What is predictable on which horizon (`timeframe_map.csv`, slide 4)

In each cell, the trained model with the highest validation evidence z for
that task and candle length, scored on the test (95% interval):

| Task (test) | 5 min | 15 min | 30 min | 1 hour | 1 day |
|---|---|---|---|---|---|
| Return: R² vs zero (%) | −0.04 [−0.10, +0.02] | −0.01 [−0.07, +0.04] | +0.00 [−0.02, +0.02] | −0.01 [−0.06, +0.04] | −1.95 [−6.11, +1.73] |
| Direction: accuracy gain vs majority (pp) | **+0.69 [+0.40, +0.98]** | +0.25 [−0.07, +0.57] | +0.16 [−0.12, +0.38] | +0.26 [−0.08, +0.55] | −0.78 [−1.52, +0.00] |
| Volatility: squared log-RV error cut vs persistence (%) | **40.0 [39.5, 40.4]** | **36.5 [35.1, 37.7]** | **42.6 [40.8, 44.3]** | **52.0 [50.0, 54.1]** | **18.7 [9.7, 24.5]** |
| Volatility: squared log-RV error cut vs HAR-OLS (%) | **6.8 [6.2, 7.3]** | **10.3 [8.8, 11.6]** | **16.7 [14.5, 18.7]** | **26.1 [23.8, 28.5]** | 7.6 [−0.9, 14.6] |

Bold: interval above zero. The gain over HAR-OLS grows with candle length up
to one hour; on daily candles (642 test sessions, see "Sample checks") the
interval includes zero.

### Five models per task (slides 6–8, `comparison_by_task.csv`)

One model per family on the task's best validation timeframe (5 minutes for
all three): a linear model (Ridge, or logistic regression for direction),
random forest, XGBoost, the best network architecture on validation, and
Timestral (frozen weights; for volatility with the supervised calibration).
Rows are ranked by validation z; the values are test results. Timestral is
scored on its fixed random sample of 20,000 test candles.

| Task | #1 | #2 | #3 | #4 | #5 |
|---|---|---|---|---|---|
| Return (R² vs zero) | DNN 256-128-64-32 −0.04% | XGBoost +0.00% | Ridge −0.03% | Random forest −0.05% | Timestral −2.55% |
| Direction (pp vs majority) | DNN 256-128-64-32 +0.69 | Random forest +0.62 | Logistic +0.50 | XGBoost +0.58 | Timestral −0.59 |
| Volatility (error cut vs persistence / vs HAR-OLS) | DNN 128-64-32 40.0% / 6.8% | XGBoost 42.0% / 9.9% | Random forest 37.1% / 2.3% | Ridge 35.7% / 0.1% | Timestral + calibration 12.2% / −35.9% |

* **Return:** no model improves on the zero forecast on any candle length; the
  5-minute network's forecasts span less than ±0.5 bp against a typical move
  of 7.7 bp. Consistent with near-random-walk returns at these horizons, but
  not a proof of it.
* **Direction:** the 5-minute gain of +0.69 pp is about 7 more right calls per
  1,000 candles, and comes from the early test years: +1.18 / +1.33 / +0.82
  pp in 2021 / 2022 / 2023 (intervals above zero), +0.02 and −0.12 pp in 2024
  and 2025 (intervals include zero; chart 2, `direction_by_year`).
* **Volatility:** HAR-OLS already captures volatility clustering, mean
  reversion and the intraday pattern (36% error cut vs persistence at 5
  minutes); what machine learning adds is the gain over HAR-OLS. It concerns
  realized, not implied, volatility (Andersen & Bollerslev, 1998).
* **Deeper networks:** no consistent gain; the best depth differs by task and
  timeframe, and differences between networks are small next to the gap to
  the benchmark.

### Strongest evidence (slide 9)

Ranking rule: (1) every model × timeframe is scored on validation by its
improvement over its task's naive benchmark; (2) z = that improvement / its
block-bootstrap standard error; (3) the highest z wins within each task.
Across tasks, z only says where the evidence of beating the **task's own**
naive benchmark is strongest; it is not a common scale of usefulness and does
not show that one task's model is "better" than another's. Task winners:
volatility DNN 128-64-32 at 5 minutes (z 44.0), direction DNN 256-128-64-32
at 5 minutes (z 11.9), return DNN 256-128-64-32 at 5 minutes (z 3.1). The
rule was applied as computed on validation; it is not a pre-registered
choice, because the volatility score itself was revised after a first look
at test numbers. On the test XGBoost (#2 by validation z) cuts the
volatility error slightly more (42.0% vs 40.0%); the ranking was not
re-chosen on the test.

**Statistical vs economic performance.** The metrics above are statistical
and are reproducible from the saved forecasts. Whether a statistical gain is
economically useful was not studied: commissions, bid-ask spread, slippage
and latency would decide that, and a small gain such as +0.69 pp in
direction may not survive them.

## Data analysis (test sample, `data_profile.csv`)

| Candle | Rows (test) | UP / DOWN / doji | Return s.d. (bp) | ACF return | ACF log RV |
|---|---|---|---|---|---|
| 5 min | 273,340 | 49.9% / 48.3% / 1.8% | 7.7 | −0.005 | 0.75 |
| 15 min | 90,189 | 50.7% / 48.3% / 1.0% | 13.3 | +0.003 | 0.84 |
| 30 min | 44,395 | 51.1% / 48.1% / 0.8% | 18.5 | −0.002 | 0.84 |
| 1 hour | 21,507 | 51.5% / 48.0% / 0.5% | 26.2 | −0.000 | 0.83 |
| 1 day | 642 | 54.2% / 45.8% / 0.0% | 116.9 | −0.001 | 0.81 |

ACF = lag-1 autocorrelation (intraday: pairs within a session; daily:
consecutive sessions). Candle direction is close to a coin flip, consecutive
returns are practically uncorrelated, while log realized variance is strongly
persistent. This matches the results: volatility is the predictable target.
The same statistics for the validation block are in `data_profile.csv`.

### Deviations found while reporting

1. **Direction majority bug.** The first run computed the majority class over
   all training rows, counting dojis as "not up", so for 5, 15 and 30 minutes
   it chose DOWN, a weak baseline that inflated the gains to about +2 pp.
   Fixed in `multitarget.majority_direction` (non-doji candles only).
   `tools/mt_fix_majority.py` recomputed the constant from the training
   data, without refitting any model; the previous file is kept as
   `dataset_info_before_majority_fix.json`.
2. **Volatility selection score revised after seeing test numbers** (see
   "Selection").
3. **Timestral volatility calibration** (review of 2026-11-12): the first
   report scored the calibrated Timestral variant on the same validation rows
   used to fit its calibration. A first fix used 2-fold cross-fitting, but
   one fold calibrated on the LATER half and scored the earlier one, which
   is not causal. Its validation score is now past-to-future only (below):
   5 min selection z 8.19 → 8.09 (2-fold) → 6.37, log-RV R² vs persistence
   +18.4% → +18.2% → +18.3% [+12.3, +22.7] on the 16,036 scorable rows
   (daily −32.1% → −39.5%). The ranking of the selected configurations, the
   five configurations and every test number are unchanged.

## Sample checks (`tools/sample_checks.py` → `sample_checks.json`, appendix A3)

Measured from the 1-minute data and the saved forecasts, without training.

**A. Why the daily test ends on 2025-01-30.** All 85 sessions from 2025-01-31
to 2025-05-30 are dropped. 84 of them fall within 50 sessions after a session
containing a flagged 1-minute bar: `study.build` keeps a row only if none of
the last 50 candles contains one (`clean50`), and `quality.flag_suspect`
flags any 1-minute bar whose log return exceeds 2% in absolute value. The
flags are on 2025-02-03 (−2.7%), 2025-04-07 (−4.5%) and 2025-04-09 (+2.6%);
2025-01-31 itself is dropped because its label session (2025-02-03) contains
the flag. Over 2006–2025 there are 16 flagged bars; 9 come right after a gap
of more than an hour (weekend, holiday or daily break) and the 2022 flags fall
at 8:30 ET, when US macro data are released, so they look like genuine price
moves rather than bad prints. Impact: 304 of the 1,002 daily test sessions
(train 150, validation 115) lie in such 50-session windows, so the **daily
sample under-represents turbulent periods**; the 642-session daily results
should be read with that in mind. Intraday the window is 50 candles, a
negligible loss. A correction (for example not flagging opening gaps)
changes the sample and needs a new daily run; the saved results were not
altered (see "Future experiments").

**B. Incomplete intraday candles.** `timeframes.aggregate` stores the number
of 1-minute bars in each candle (`n_minutes`) but does not use it: a
full-length 5-minute candle with 4 of its 5 minutes is accepted as complete
(`tests/test_timing_and_splits.py` documents this). TradeStation writes no
bar for a minute without trades; scheduled halts and the overnight break lie
outside the candles. Share of full-length candles with missing minutes:

| Candle | All years | Test | Test, 08:30–15:00 CT | Test rows whose label candle is incomplete |
|---|---|---|---|---|
| 5 min | 18.0% | 0.23% | 0.003% | 601 of 273,340 |
| 15 min | 25.8% | 0.43% | 0.008% | 385 of 90,189 |
| 30 min | 30.6% | 0.65% | 0.008% | 279 of 44,395 |
| 1 hour | 34.3% | 1.01% | 0.000% | 201 of 21,507 |

Missing minutes are concentrated overnight in 2006–2017 (54% of 5-minute
candles in 2006, 0.02% in 2025), consistent with thin trading; in the test a
few sessions with long gaps coincide with quarterly contract-roll days
(e.g. 2021-09-09, 2022-09-08) and are dataset gaps. Rescoring the saved test
forecasts without rows whose label candle is incomplete moves the rescored
metrics (HAR-OLS, the 5-minute networks and XGBoost; return, direction,
volatility) by at most 0.4 points (e.g. 5-minute DNN volatility 39.99% →
40.00%, 1-hour XGBoost 51.95% → 52.34%): no conclusion changes.

## Timestral on the same sample (`tools/timestral_same_sample.py`, appendix A4)

On exactly the same random test candles (saved forecasts, 95% block
intervals in `timestral_same_sample.csv`):

| Candle | Return R²: Timestral / DNN | Direction pp, Timestral: body / close-to-close | Direction pp, DNN (body) | Volatility error cut vs persistence: Timestral / DNN | Correlation with next log RV: Timestral spread / current RV |
|---|---|---|---|---|---|
| 5 min | −2.55% / +0.20% | −0.59 / −0.66 | +0.51 | +12.2% / +39.6% | 0.76 / 0.76 |
| 15 min | −2.23% / −0.10% | +0.01 / −0.08 | +1.09 | −37.1% / +34.8% | 0.74 / 0.84 |
| 30 min | −1.46% / +0.02% | −1.76 / −1.88 | +0.14 | −56.2% / +41.3% | 0.71 / 0.84 |
| 1 hour | −1.59% / +0.00% | −2.28 / −2.36 | +0.02 | −74.3% / +41.8% | 0.64 / 0.83 |
| 1 day | −1.83% / +0.17% | −1.87 / −1.56 | +0.00 | −19.3% / +18.7% | 0.75 / 0.82 |

* **Target:** Timestral forecasts the next close (direction = median vs last
  close) while the study scores the candle body. Scored on the close-to-close
  direction it does no better, so the mismatch does not explain its result.
* **Sample:** the trained models keep their advantage on Timestral's own
  20,000-candle sample (daily: all 642 sessions), so the gap is not a
  sampling artefact.
* **Volatility:** Timestral's volatility is indirect, the q10–q90 spread of
  its price forecast mapped to log RV by a linear calibration fitted on
  validation; the trained models forecast log RV directly. The raw spread
  correlates with next log RV slightly less than the current RV does, and the
  calibrated forecast is worse than persistence beyond 5 minutes.
* **Scope:** one small model (5M parameters), frozen weights, a 512-close
  context, one market. These results do not generalise to time-series
  foundation models in general (Mistral AI, 2026).

## Data

* Instrument: **NQ**, E-mini Nasdaq-100 futures (CME), back-adjusted
  continuous TradeStation 1-minute series (sessions before the 2025-06-02
  confirmatory hold-out), $20 per index point. No QQQ, MNQ, cash index or VIX
  data is mixed in.
* Timeframes: 5 min, 15 min, 30 min, 1 hour, 1 day (one CME session,
  17:00–16:00 CT), all rebuilt from the 1-minute bars inside each session
  (`timeframes.aggregate`; `30min` was added to `FREQUENCIES` for this study).
  1-minute candles are excluded: the realized variance of a 1-minute candle
  from 1-minute data would be a single squared return.
* Same chronological boundaries for every timeframe and task; labels that
  cross a split boundary are purged; intraday labels never cross a session.
  Actual sample dates and rows (signal sessions, `dataset_info.json`):

| Timeframe | Train | Validation | Test |
|---|---|---|---|
| 5 min | 2006-01-10 → 2017-08-24, 769,058 | 2017-08-25 → 2021-07-14, 269,167 | 2021-07-15 → 2025-05-30, 273,340 |
| 15 min | 2006-01-10 → 2017-08-24, 263,691 | 2017-08-25 → 2021-07-14, 88,336 | 2021-07-15 → 2025-05-30, 90,189 |
| 30 min | 2006-01-10 → 2017-08-24, 132,606 | 2017-08-25 → 2021-07-14, 44,513 | 2021-07-15 → 2025-05-30, 44,395 |
| 1 hour | 2006-01-10 → 2017-08-24, 64,813 | 2017-08-25 → 2021-07-14, 21,642 | 2021-07-15 → 2025-05-30, 21,507 |
| 1 day | 2006-03-13 → 2017-08-23, 1,991 | 2017-08-25 → 2021-07-13, 822 | 2021-07-15 → **2025-01-30**, 642 |

  The daily sample starts later because the 50-candle moving-average
  feature needs 50 sessions of history. It ends on 2025-01-30 because of the
  flagged-bar filter (measured cause in "Sample checks").
* The test period was already inspected for the direction task in earlier
  project phases, so the test results are **exploratory**, not a clean
  confirmation.

## Targets (all for candle t+1, decided at the close of candle t)

| Task | Target | Benchmark | Metrics |
|---|---|---|---|
| Return | r = log(C(t+1)/C(t)) in basis points | zero return (also: training mean) | MAE, RMSE, out-of-sample R² vs zero, sign accuracy, correlation |
| Direction | UP if C(t+1) > O(t+1), DOWN if lower; doji kept and counted as a miss | majority class of the non-doji training candles | accuracy, balanced accuracy, precision, recall, F1, confusion matrix |
| Volatility | RV(t+1) = Σ squared 1-minute log returns inside candle t+1 (first term log(C/O) of its first minute) | persistence RV(t+1) = RV(t); HAR-OLS (also: rolling mean of 22 candles) | MAE/RMSE of log RV, R² vs persistence and vs HAR-OLS, QLIKE on the variance level, correlation; RV direction (UP if RV(t+1) > RV(t)): accuracy, balanced accuracy, precision, recall, F1 vs majority and vs HAR-OLS |

Squared returns are a variance proxy; volatility is its square root. Models
forecast log RV; the level forecast used for QLIKE is
`c · exp(log forecast)`, with `c` estimated on the training residuals
(Timestral: on the calibration rows).

## Features (known at the close of candle t)

Past returns (lags 1, 2, 3, 5), momentum (5, 10), rolling volatility (5, 10,
20), relative volume, range, body, wicks, RSI(14), moving-average gaps (20,
50), time of day (sin/cos), and log realized variance of the last 1, 5 and 22
candles plus, intraday, the mean RV of the next candle's time slot over the 5
previous sessions. HAR-OLS uses only the realized-variance features.

## Models

See slide 5 and `reports/multitarget_final/models.csv` (every fit, with
epochs / best iteration / chosen hyper-parameter).

* Benchmarks: zero return, training mean, majority class, persistence,
  rolling mean of 22 candles.
* Statistical: HAR-OLS (volatility), Ridge (α ∈ {0.1, 10, 1000} chosen on
  validation; return, volatility), logistic regression (C ∈ {0.01, 1} chosen on
  validation; direction).
* Machine learning: random forest (200 trees, depth 8), XGBoost (depth 4,
  learning rate 0.03, early stopping on validation).
* Deep learning: feed-forward networks 64-32-16 (the repository's original
  architecture), 64-32, 128-64-32, 256-128-64-32; ReLU, AdamW, weight decay
  1e-4, early stopping on the validation loss (patience 4). Return networks
  are trained on a winsorised target (0.1% / 99.9% training quantiles) and
  scored on the raw one.
* **Mistral AI — Timestral-5M** (`mistralai/Timestral-5M`, Mistral's
  pretrained time-series foundation model, local weights, no API, no cost; it
  is not a language model and does not read text). Role in the repository:
  `tools/timestral_direction.py` (unchanged) ran it
  zero-shot for next-close direction. Here two variants are distinguished:
  * **Timestral-5M, frozen weights** (return, direction; zero-shot): from the
    last 512 closes of the same timeframe it forecasts quantiles of the next
    close. Return forecast = log(median / last close). Direction = sign of
    (median − last close), i.e. a next-**close** forecast, while the
    classification target is the **body** of the next candle (close vs
    open); the mismatch is part of what is being tested.
  * **Timestral-5M + supervised volatility calibration** (volatility; not
    zero-shot): the q10–q90 spread gives a variance proxy, mapped to log RV
    by a linear regression fitted on **validation targets**. For the
    validation score used in selection the calibration runs only from past
    to future: the validation sample is cut into 5 consecutive blocks of
    origin sessions, and block k (k = 1..4) is scored with a regression
    fitted on every earlier row whose target session ends before block k
    starts (expanding window), so each calibration target is already known
    when the scored forecasts are made. Block 0 has no earlier calibration
    and is excluded from the score, for Timestral and for the benchmarks
    compared with it (about 16,000 of the 20,000 intraday rows and 657 of
    the 822 daily rows are scored). Test forecasts use the calibration
    fitted on the whole validation sample, whose targets all precede the
    first test origin. Coefficients and the calibration / evaluation date
    ranges of every block are in `timestral_calibration.json`.
  Both are scored on a fixed random sample of 20,000 validation and 20,000
  test origins per timeframe (all of them for daily candles), against the
  benchmarks on exactly those rows. Pretraining-data overlap with NQ is
  unknown. Timestral-28M, -123M and -209M (config and weights available in
  `data_raw/model_cache`) were **not tested in this
  study**. There is no Mistral chat/LLM API integration in the repository.
* Considered, not run: LSTM. The feature networks already reach their
  early-stopping optimum within a few epochs, and a sequence model would add
  cost without a reason in the validation results.

## Selection

`multitarget_report.py`. The criteria use validation data only, but the
selection is **not blind**:

* **Initial criterion (first report):** evidence score = z-statistic
  (20-session moving-block bootstrap) of the paired validation improvement
  over the naive benchmark: squared-error reduction vs zero (return), accuracy
  gain vs majority (direction), **QLIKE reduction vs persistence**
  (volatility).
* **Revision (after the first report had printed test numbers):** on
  validation >99% of persistence's QLIKE came from a handful of limit-locked
  candles in March 2020 with almost zero variance, so the volatility score
  was ~1 for every model. The volatility score was replaced by the
  **squared-error reduction of log RV vs persistence**. Return and direction
  scores are unchanged.
* **Criterion actually used:** z of the validation squared-error reduction
  vs zero (return), of the accuracy gain vs majority (direction), and of the
  squared-error reduction of log RV vs persistence (volatility). Each task
  gets its highest-z configuration; the two remaining slots go to the two
  tasks whose best configuration on a **different timeframe** has the highest
  z (at most two per task). Benchmarks are never selected. The five are then
  scored on the test block.

Because the revision came after test numbers were seen, and the test period
had been examined before, the selection and all test results are exploratory.

The deck uses these z scores in three ways: the five-model tables (one model
per family on each task's best validation timeframe), the timeframe map (the
highest-z trained model per task and timeframe) and the task winners on
slide 9. The earlier five-configuration selection (`selected_five.csv`, the
rule above with two extra slots on a different timeframe) still provides the
charts.

## Statistics

* Differences (accuracy gains, RV-direction gains vs majority or HAR-OLS,
  selection z): moving-block bootstrap with 20-session blocks of the paired
  per-row differences, 2,000 replicates.
* Relative improvements (R² vs zero, R² of log RV vs persistence and vs
  HAR-OLS, QLIKE reduction): model and benchmark losses are resampled
  **together** by session blocks and the ratio 1 − Σ model loss / Σ benchmark
  loss is recomputed in every replicate (`block_boot_ratio`). (The first
  report divided an interval of the loss difference by a fixed denominator;
  point estimates were unaffected, intervals moved by up to a few tenths of a
  point.)
* Return chart (slide 6): each decile's realized-mean interval is a
  20-session block bootstrap within the decile (first report: independent
  standard errors).
* Classification convention (`cls_stats`, scikit-learn's
  `zero_division=0`): precision = 0 without positive predictions, recall = 0
  without positive cases, F1 = 2TP / (2TP + FP + FN), and 0 when that
  denominator is 0. (First report: NaN; this changed only the
  RV-direction precision/F1 of the persistence benchmark, which never
  predicts UP.)


## Reproduce

Windows PowerShell from the repository root. Environment of the final run
(`run.json`): Python 3.12.3, Windows 11, CPU; exact versions in
`requirements-lock.txt` (numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1,
xgboost 3.4.1, torch 2.15.0+cpu, …) and `requirements-timestral.txt`
(timestral 1.0.0 at a pinned commit). Node.js 18+ for the deck (pptxgenjs
3.12.0 pinned in `presentation/package-lock.json`). Timestral-5M weights must
be in the Hugging Face cache (first download:
`python -c "from timestral import TimestralForecaster; TimestralForecaster.from_pretrained('mistralai/Timestral-5M')"`
with `HF_HUB_OFFLINE` unset; `multitarget.py` itself runs offline).

The NQ 1-minute data (TradeStation, 2006–2025) are licensed and **not in
Git**; neither are `results/` (saved forecasts) and `data_raw/` (model cache).
Point `FUTURES_DATA_DIR` to a local copy (default `..\futures-data`).

```powershell
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements-lock.txt -r requirements-timestral.txt
$env:FUTURES_DATA_DIR = "C:\path\to\futures-data"
python multitarget.py --tag final                       # full training, ~30 min on 16 CPU threads
python multitarget_report.py results\multitarget\runs\<run id> reports\multitarget_final
python tools\timestral_same_sample.py results\multitarget\runs\<run id> reports\multitarget_final
python tools\sample_checks.py results\multitarget\runs\<run id> reports\multitarget_final   # reads the 1-minute data, ~1 min
copy reports\multitarget_final\deck_data.json, reports\multitarget_final\chart_*.png, reports\multitarget_final\sample_checks.json, reports\multitarget_final\timestral_same_sample.json presentation\
python verify_multitarget.py results\multitarget\runs\<run id> reports\multitarget_final presentation
cd presentation
npm ci                                                  # installs pptxgenjs 3.12.0 from package-lock.json
node build_deck.js                                      # main deck + appendix
cd ..
python -m pytest -q                                     # from the repository root
```

Pipeline check in a minute:
`python multitarget.py --freqs 1h session --years 2015 2019 --quick --timestral-n 500`.

**What the checks cover.** `verify_multitarget.py` recomputes, from the saved
forecasts and without `multitarget_report.py`, every point estimate in the
three results tables (including the past-to-future Timestral calibration,
with a check that no calibration target falls after the origins it is used
for, and the HAR-OLS comparisons), the ranking rules, the five-model tables,
the timeframe map, the selected-five fields, every value of
`data_profile.csv`, the per-year direction gains and the deck data (run id,
rows, charts, appendix inputs, no trading fields, no VIX). On the final run,
with the presentation folder: 3,960 checks, 0 problems. The unit tests
(`pytest`, 41 tests) cover targets, timing, sessions, splits,
bootstrap, the ranking rule and the causal Timestral calibration.

**What they do not cover.** The raw 1-minute data and the candle building
(checked separately by `tools/sample_checks.py`); the training (models were
not re-fitted); the bootstrap intervals (only their order is checked). The
final run used an uncommitted version of `multitarget.py`: its hash in
`run.json` (`a448847f…`) matches no committed version. The known later
changes are the majority-class fix (applied to the saved outputs by
`tools/mt_fix_majority.py`) and docstrings; the other source files used by
the run match the committed versions (hashes in `run.json`). A bit-identical
re-run is therefore not guaranteed.

Outputs of `multitarget.py` (`results/multitarget/runs/<id>/`):
`predictions_<freq>.parquet` (validation and test rows, every model's
forecast), `models.csv`, `dataset_info.json`, `console.log`, `run.json`.
Outputs of `multitarget_report.py`: `results_return.csv`,
`results_direction.csv`, `results_volatility.csv` (every model × timeframe ×
split, plus the Timestral-sample rows), `data_profile.csv`,
`selection_validation_ranking.csv`, `comparison_by_task.csv`,
`timeframe_map.csv`, `selected_five.csv` (the earlier five-configuration
selection, source of the charts), `configurations_tested.csv`,
`timestral_calibration.json`, `chart_cfg<id>_*.png`, `deck_data.json`.
Tools: `sample_checks.json`, `timestral_same_sample.csv/.json`.

## Future experiments (proposed, not run; by benefit and cost)

1. **Daily sample fix** — high benefit, low cost (daily data set is small):
   flag only isolated reverting spikes, not opening gaps, then re-run
   `multitarget.py --freqs session` as a new run; compare with the saved one.
2. **Aligned Timestral targets** — medium benefit, low–medium cost: forecast
   open and close of t+1 (body direction) and the quantiles of the next
   candle's 1-minute returns, derive RV from Timestral's own path, and add
   Timestral-28M/123M/209M (weights already cached) on the same sample.
3. **Pre-registered confirmation** — high benefit, medium cost: freeze the
   models and the ranking rule now and score them once on data after
   2025-06-02 (the reserved hold-out), turning the exploratory result into a
   confirmatory one.
4. **Economic evaluation of volatility forecasts** — medium benefit, medium
   cost: e.g. volatility-targeted position sizing with realistic costs,
   reported separately from the statistical results.
5. **Incomplete-candle policy** — low benefit (impact < 0.4 points), medium
   cost: use `n_minutes` to drop or flag candles in dataset-gap sessions, then
   a full re-run.

## Repository map

| Path | Content |
|---|---|
| `multitarget.py` | builds the data for each timeframe, trains every model, runs Timestral, saves the forecasts |
| `multitarget_report.py` | tables, data profile, rankings, five-model tables, timeframe map, charts, `deck_data.json` (reads saved forecasts only) |
| `verify_multitarget.py` | independent recomputation of the reported numbers |
| `config.py`, `data.py`, `timeframes.py`, `quality.py`, `features.py`, `evaluation.py`, `runlog.py` | data loading, CME sessions and candles, data checks, features and splits, run folders |
| `study.py`, `experiment.py`, `backtest.py`, `strategy.py`, `regimes.py`, `plots.py` | shared pipeline code from the earlier direction study; `multitarget.py` uses `study.build` (data set) and `experiment.load_minutes` / `bars_for`; their trading functions are not used here |
| `tools/sample_checks.py`, `tools/timestral_same_sample.py` | sample checks (appendix A3) and Timestral on the same sample (appendix A4) |
| `tools/mt_fix_majority.py`, `tools/timestral_direction.py` | majority-class fix without refitting; original zero-shot Timestral direction tool |
| `tests/` | unit tests (`python -m pytest -q`) |
| `presentation/` | deck generator, deck data, charts, main deck and appendix (`.pptx`, `.pdf`) |
| `reports/multitarget_final/` | final tables, charts and check outputs |

The earlier direction-and-trading study (`study.py` run, `report.py`,
`verify.py`, the NQ decks, IBKR download tools) is on the Git branch
`archive/nq-direction-study`.

## References

* Andersen, T. G., & Bollerslev, T. (1998). Answering the skeptics: Yes,
  standard volatility models do provide accurate forecasts. *International
  Economic Review*, 39(4), 885–905.
* Corsi, F. (2009). A simple approximate long-memory model of realized
  volatility. *Journal of Financial Econometrics*, 7(2), 174–196.
* Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary
  observations. *Annals of Statistics*, 17(3), 1217–1241.
* Patton, A. J. (2011). Volatility forecast comparison using imperfect
  volatility proxies. *Journal of Econometrics*, 160(1), 246–256.
* Mistral AI (2026). Timestral: time series foundation models. Announcement
  https://mistral.ai/news/timestral; code https://github.com/mistralai/timestral
  (commit a534bdbd); weights https://huggingface.co/mistralai/Timestral-5M.
