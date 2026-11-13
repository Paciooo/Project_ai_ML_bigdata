# NASDAQ multi-target forecasting (experimental branch)

Branch `research/nasdaq-multi-target-forecasting`, worktree
`../nasdaq-multi-target-forecasting`. It starts from commit `35be806` plus the
uncommitted fixes of the direction study (threshold status, verify.py), so the
original study is reproducible from here too. The original checkout and the
`claude/nq-direction-baseline-*` worktree were not modified.

Research question: **which characteristic of the next NASDAQ candle — return,
direction or realized volatility — can machine learning predict, and on which
timeframe?**

Deck: [presentation_multitarget/NASDAQ_multitarget_report.pptx](presentation_multitarget/NASDAQ_multitarget_report.pptx)
([PDF](presentation_multitarget/NASDAQ_multitarget_report.pdf)), 8 slides.
Result tables: [reports/multitarget_final/](reports/multitarget_final/).

**All results are exploratory.** The 2021–2025 test period was examined in
earlier phases of the project, and the volatility selection score was revised
after a first report had printed test numbers (see "Selection"). Nothing
here is a blind, confirmatory test.

## Results (run `20261112T050812Z_82953a_final`)

**Answer (exploratory): realized volatility is the most predictable
characteristic in this study. For next-candle returns no improvement over
forecasting zero was demonstrated with the models, data and horizons tried.
For next-candle direction a small accuracy gain was measured, far below the
cost of a trade.**

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
* **Direction and trading (ID 2, $2.50 per round trip, idealised fills):**
  Long & Short −$515.9k on 273,340 trades (gross +$167.4k, $0.61 per trade);
  long only −$307.7k; short only −$208.3k; at $5 per round trip −$1.20M. A
  small but measurable accuracy gain does not pay the commission. Every
  direction model × timeframe × strategy is in `trading_direction_test.csv`:
  some daily and 30-minute configurations are positive on the test, but they
  were not selected and are multiple comparisons, not findings.
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
  implied, volatility: it says nothing about option prices or option P&L,
  which were not computed.
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

### Deviations found while reporting (stated on the slides)

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
  (`timeframes.aggregate`; `30min` was added to `FREQUENCIES` in this branch).
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
  `tools/timestral_direction.py` (restored in this branch, unchanged) ran it
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
  the original project's `data_raw/model_cache`) were **not tested in this
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

## Trading test (direction task)

UP → long, DOWN → short (Long & Short); Long only; Short only. Entry at the
open of candle t+1, exit at its close, one NQ contract ($20 per point),
**$2.50 per round trip** (commission only), zero slippage, 100% fills, zero
latency: idealised. Real limit orders are not guaranteed to fill. Net P&L at
$5 per round trip (the main study's assumption) is reported next to it.
Drawdown is measured on trade-by-trade equity starting at 0.

## Reproduce

Windows PowerShell from the repository root, Python 3.12
(`requirements.txt`, plus `requirements-timestral.txt` for Timestral; the run
used torch 2.15.0+cpu, xgboost 3.4.1, scikit-learn 1.9.1, timestral 1.0.0),
Node.js 18+, and the Timestral-5M weights in the Hugging Face cache (first
download: `python -c "from timestral import TimestralForecaster; TimestralForecaster.from_pretrained('mistralai/Timestral-5M')"`
with `HF_HUB_OFFLINE` unset; `multitarget.py` itself runs offline).

```powershell
$env:FUTURES_DATA_DIR = "C:\path\to\futures-data"
python multitarget.py --tag final                       # full training, ~30 min on 16 CPU threads
python multitarget_report.py results\multitarget\runs\<run id> reports\multitarget_final
python verify_multitarget.py results\multitarget\runs\<run id> reports\multitarget_final presentation_multitarget
copy reports\multitarget_final\deck_data.json, reports\multitarget_final\chart_*.png presentation_multitarget\
cd presentation_multitarget
npm ci                                                  # installs pptxgenjs 3.12.0 from package-lock.json
node build_deck.js NASDAQ_multitarget_report.pptx
cd ..
python -m pytest -q                                     # from the repository root
```

(Run `verify_multitarget.py` again after copying the deck data: it also
checks that the presentation copies equal the report.) Pipeline check in a
minute: `python multitarget.py --freqs 1h session --years 2015 2019 --quick --timestral-n 500`.

`verify_multitarget.py` recomputes, from the saved predictions and without
`multitarget_report.py`, every point estimate in the three results tables
(including the past-to-future Timestral calibration, with a check that
no calibration target falls after the origins it is used for, and the HAR-OLS
comparisons), the selection rule, every trading row, the selected-five test
fields and the deck data (run id, rows, trading, charts, costs, no VIX). It
does not re-run the bootstraps (it checks interval order only), rebuild the
data from the 1-minute bars or refit models. On the final run, with the
presentation folder: 4,401 checks, 0 problems (an altered P&L or HAR ratio
in a copied report is flagged).

Outputs of `multitarget.py` (`results/multitarget/runs/<id>/`):
`predictions_<freq>.parquet` (validation and test rows, every model's
forecast), `models.csv`, `dataset_info.json`, `console.log`, `run.json`.
Outputs of `multitarget_report.py`: `results_return.csv`,
`results_direction.csv`, `results_volatility.csv` (every model × timeframe ×
split, plus the Timestral-sample rows), `trading_direction_test.csv` (every
direction model × timeframe × strategy), `selection_validation_ranking.csv`,
`selected_five.csv`, `configurations_tested.csv`,
`timestral_calibration.json`, `chart_cfg<id>_*.png`, `deck_data.json`.
