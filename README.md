# What Can Machine Learning Predict in NASDAQ Markets?

University project (Waseda University, *AI, Machine Learning and Big Data for
Finance and Management*). Author: Alessandro Pacifici.

Research question: **which characteristic of the next NASDAQ futures candle —
its return, its direction or its realized volatility — can machine learning
predict, and on which timeframe?** The study is purely statistical: forecast
accuracy against simple benchmarks, with uncertainty intervals. It contains
no trading rules, P&L or cost assumptions.

* Deck (cover + 8 slides: five models compared per task, the best model overall, reproducibility):
  [presentation/NASDAQ_multitarget_report.pptx](presentation/NASDAQ_multitarget_report.pptx)
  ([PDF](presentation/NASDAQ_multitarget_report.pdf)).
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
characteristic in this study. For next-candle returns no improvement over
forecasting zero was demonstrated with the models, data and horizons tried.
For next-candle direction a small accuracy gain was measured, and it is not
stable across the test years.**

The five configurations (chosen with validation data, scored on the test
block; 95% intervals from 20-session moving-block bootstraps, see
"Statistics"):

| ID | Task | Model | Timeframe | Key metric (test) | Baseline | Improvement |
|---|---|---|---|---|---|---|
| 1 | Return | DNN 256-128-64-32 | 5 min | RMSE 7.71 bp | zero return: RMSE 7.71 bp | R² −0.04% [−0.10, +0.02] |
| 2 | Direction | DNN 256-128-64-32 | 5 min | accuracy 50.59% (balanced 51.37%) | majority (UP) 49.90% | +0.69 pp [+0.40, +0.98] |
| 3 | Direction | DNN 64-32 | 15 min | accuracy 50.99% (balanced 50.83%) | majority (UP) 50.74% | +0.25 pp [−0.07, +0.57] |
| 4 | Volatility | DNN 128-64-32 | 5 min | log-RV RMSE 0.901; RV UP/DOWN 72.2% | persistence 1.163, majority 51.1%; HAR-OLS 0.933, 71.0% | vs persistence R² +40.0% [+39.5, +40.4]; vs HAR-OLS R² +6.8% [+6.2, +7.3], RV direction +1.13 pp [+0.95, +1.31] |
| 5 | Volatility | XGBoost | 1 hour | log-RV RMSE 0.572; RV UP/DOWN 74.8% | persistence 0.825, majority 47.7%; HAR-OLS 0.666, 70.5% | vs persistence R² +52.0% [+50.0, +54.1]; vs HAR-OLS R² +26.1% [+23.8, +28.5], RV direction +4.28 pp [+3.53, +4.88] |

Accuracy counts dojis as misses for every call, so the majority baseline
can be below 50%. Test periods: 2021-07-15 → 2025-05-30 for the intraday
timeframes, 2021-07-15 → 2025-01-30 for daily candles (see "Data").

* **Return:** no selected or unselected model shows an improvement over the
  zero forecast on the test block; the selected network's forecasts span less
  than ±0.5 bp against a typical 5-minute move of 7.7 bp. This is consistent
  with near-random-walk returns at these horizons but does not prove it.
* **Direction:** configuration 2 gains +0.69 pp over the majority class
  (about 7 more right calls per 1,000 candles), but the gain comes from the
  early test years: +1.18 / +1.33 / +0.82 pp in 2021 / 2022 / 2023 (intervals
  above zero), +0.02 and −0.12 pp in 2024 and 2025 (intervals include zero;
  `deck_data.json`, `direction_by_year`, and chart 2). Configuration 3 (DNN
  64-32, 15 min) is not distinguishable from the majority class.
* **Volatility, two benchmarks:** persistence is naive; HAR-OLS (linear
  regression on past log RV and the time-of-day slot) already captures
  volatility clustering, mean reversion and the intraday pattern: it removes
  36% (5 min) and 35% (1 h) of persistence's squared log-RV error and gets the
  RV direction right 71.0% / 70.5% of the time. What ML adds is the gain
  **over HAR-OLS**: +6.8% and +26.1% of HAR's squared error, +1.13 and +4.28
  pp of RV-direction accuracy, all with intervals above zero. Intraday, every
  trained volatility model beats persistence on the test (log-RV R² +29% to
  +52%); on daily candles the gain is smaller and not universal (+9% to +22%,
  random forest −19%; `results_volatility.csv`). It concerns realized, not
  implied, volatility: it says nothing about option prices, which were not
  computed.
* **Mistral / Timestral-5M** (test sample of each timeframe; not among the five):
  * frozen weights, return: R² −2.55% (5 min), −2.23% (15 min), −1.46%
    (30 min), −1.59% (1 h), −1.83% (1 day) vs forecasting zero, intervals
    below zero except daily [−4.38, +0.19];
  * frozen weights, direction: −0.59, 0.00, −1.76, −2.28, −1.87 pp vs the
    majority class (30 min and 1 h intervals below zero, the others include 0);
  * with supervised volatility calibration: log-RV R² vs persistence +12.2%
    [+9.7, +14.8] at 5 min, negative on every other timeframe (−37% to −74%,
    daily −19%).
* **Deeper networks:** no consistent gain. On the selected configurations
  the best network depth differs by task and timeframe (256-128-64-32,
  128-64-32 and 64-32), and differences between networks are a fraction of
  the gap to the benchmark (`results_*.csv`).

## Five models per task and the best model overall (slides 5–8)

For each task the deck compares five models, one per family, on the task's
best validation timeframe (5 minutes for all three): a linear model (Ridge,
or logistic regression for direction), random forest, XGBoost, the best
neural-network architecture on validation, and Timestral (frozen weights;
for volatility with the supervised calibration). Rows are ranked by the
validation evidence score z (`comparison_by_task.csv`); all other columns
are test results.

| Task | #1 | #2 | #3 | #4 | #5 |
|---|---|---|---|---|---|
| Return (R² vs zero) | DNN 256-128-64-32 −0.04% | XGBoost +0.00% | Ridge −0.03% | Random forest −0.05% | Timestral −2.55% |
| Direction (pp vs majority) | DNN 256-128-64-32 +0.69 | Random forest +0.62 | Logistic +0.50 | XGBoost +0.58 | Timestral −0.59 |
| Volatility (log-RV R² vs persistence / vs HAR-OLS) | DNN 128-64-32 +40.0% / +6.8% | XGBoost +42.0% / +9.9% | Random forest +37.1% / +2.3% | Ridge +35.7% / +0.1% | Timestral + calibration +12.2% / −35.9% |

Timestral is scored on its fixed random sample of 20,000 test candles.

**Ranking rule (slide 8).** (1) Every model × timeframe is scored on
validation by its improvement over the task's naive benchmark; (2) the
evidence score z = improvement / block-bootstrap standard error is
unit-free, so tasks can be compared; (3) the highest z wins within each
task, and the best model overall is the task winner with the highest z;
(4) only then is it scored on the test. Task winners: volatility DNN
128-64-32 (z 44.0), direction DNN 256-128-64-32 (z 11.9), return DNN
256-128-64-32 (z 3.1). **Best model overall: DNN 128-64-32 forecasting
5-minute realized volatility.** On the test XGBoost (#2 on validation) is
slightly better (+42.0% vs +40.0% over persistence); the ranking follows the
validation rule fixed in advance, not the test.

**Laboratory conditions.** Every number is measured on clean historical
data with no commissions, bid-ask spread, slippage or latency. In live
markets these costs apply, so the values shown are not reproducible in
practice and a small edge (such as +0.69 pp in direction) may disappear.

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
  feature needs 50 sessions of history. It ends on 2025-01-30: later daily
  rows are removed by the filters of the shared pipeline (label eligibility,
  suspect-bar windows); the exact cause was not investigated in this study.
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

See slide 4 and `reports/multitarget_final/models.csv` (every fit, with
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

## Selection of the five configurations

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
* Return chart (slide 5): each decile's realized-mean interval is a
  20-session block bootstrap within the decile (first report: independent
  standard errors).
* Classification convention (`cls_stats`, scikit-learn's
  `zero_division=0`): precision = 0 without positive predictions, recall = 0
  without positive cases, F1 = 2TP / (2TP + FP + FN), and 0 when that
  denominator is 0. (First report: NaN; this changed only the
  RV-direction precision/F1 of the persistence benchmark, which never
  predicts UP.)


## Reproduce

Windows PowerShell from the repository root, Python 3.12
(`requirements.txt`, plus `requirements-timestral.txt` for Timestral; the run
used torch 2.15.0+cpu, xgboost 3.4.1, scikit-learn 1.9.1, timestral 1.0.0),
Node.js 18+, and the Timestral-5M weights in the Hugging Face cache (first
download: `python -c "from timestral import TimestralForecaster; TimestralForecaster.from_pretrained('mistralai/Timestral-5M')"`
with `HF_HUB_OFFLINE` unset; `multitarget.py` itself runs offline).

The NQ 1-minute data (TradeStation, 2006–2025) are licensed and **not in
Git**; neither are `results/` (saved forecasts) and `data_raw/` (model cache).
Point `FUTURES_DATA_DIR` to a local copy (default `..\futures-data`).

```powershell
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-timestral.txt
$env:FUTURES_DATA_DIR = "C:\path\to\futures-data"
python multitarget.py --tag final                       # full training, ~30 min on 16 CPU threads
python multitarget_report.py results\multitarget\runs\<run id> reports\multitarget_final
copy reports\multitarget_final\deck_data.json, reports\multitarget_final\chart_*.png presentation\
python verify_multitarget.py results\multitarget\runs\<run id> reports\multitarget_final presentation
cd presentation
npm ci                                                  # installs pptxgenjs 3.12.0 from package-lock.json
node build_deck.js NASDAQ_multitarget_report.pptx
cd ..
python -m pytest -q                                     # from the repository root
```

Pipeline check in a minute:
`python multitarget.py --freqs 1h session --years 2015 2019 --quick --timestral-n 500`.

`verify_multitarget.py` recomputes, from the saved predictions and without
`multitarget_report.py`, every point estimate in the three results tables
(including the past-to-future Timestral calibration, with a check that no
calibration target falls after the origins it is used for, and the HAR-OLS
comparisons), the selection rule, the selected-five test fields, every value
of `data_profile.csv`, the per-year direction gains and the deck data (run
id, rows, charts, no trading fields, no VIX). It does not re-run the
bootstraps (it checks interval order only), rebuild the data from the
1-minute bars or refit models. On the final run, with the presentation
folder: 3,922 checks, 0 problems.

Outputs of `multitarget.py` (`results/multitarget/runs/<id>/`):
`predictions_<freq>.parquet` (validation and test rows, every model's
forecast), `models.csv`, `dataset_info.json`, `console.log`, `run.json`
(package versions and the hash of every source file; the final run was made
from a working copy of commit `35be806`). Outputs of `multitarget_report.py`:
`results_return.csv`, `results_direction.csv`, `results_volatility.csv`
(every model × timeframe × split, plus the Timestral-sample rows),
`data_profile.csv`, `selection_validation_ranking.csv`, `comparison_by_task.csv` (slides 5–8), `selected_five.csv` (the earlier five-configuration selection),
`configurations_tested.csv`, `timestral_calibration.json`,
`chart_cfg<id>_*.png`, `deck_data.json`.

## Repository map

| Path | Content |
|---|---|
| `multitarget.py` | builds the data for each timeframe, trains every model, runs Timestral, saves the forecasts |
| `multitarget_report.py` | tables, data profile, selection of the five, charts, `deck_data.json` (reads saved forecasts only) |
| `verify_multitarget.py` | independent recomputation of the reported numbers |
| `config.py`, `data.py`, `timeframes.py`, `quality.py`, `features.py`, `evaluation.py`, `runlog.py` | data loading, CME sessions and candles, data checks, features and splits, run folders |
| `study.py`, `experiment.py`, `backtest.py`, `strategy.py`, `regimes.py`, `plots.py` | shared pipeline code from the earlier direction study; `multitarget.py` uses `study.build` (data set) and `experiment.load_minutes` / `bars_for`; their trading functions are not used here |
| `tools/mt_fix_majority.py`, `tools/timestral_direction.py` | majority-class fix without refitting; original zero-shot Timestral direction tool |
| `tests/` | unit tests (`python -m pytest -q`) |
| `presentation/` | deck generator, deck data, charts, `.pptx` and `.pdf` |
| `reports/multitarget_final/` | final tables and charts |

The earlier direction-and-trading study (`study.py` run, `report.py`,
`verify.py`, the NQ decks, IBKR download tools) is on the Git branch
`archive/nq-direction-study`.
