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

## Results (run `20261112T050812Z_82953a_final`, test 2021-07-15 → 2025-05-30)

**Answer: realized volatility is by far the most predictable characteristic.
Next-candle returns are not predictable out of sample. Next-candle direction
is very slightly predictable, but the edge is far below the cost of a
trade.** The test period is exploratory (see Data).

The five configurations (selected on validation, scored once on test; 95%
moving-block bootstrap intervals, 20-session blocks):

| ID | Task | Model | Timeframe | Key metric (test) | Baseline | Improvement |
|---|---|---|---|---|---|---|
| 1 | Return | DNN 256-128-64-32 | 5 min | RMSE 7.71 bp | zero return: RMSE 7.71 bp | R² −0.04% [−0.11, +0.02] |
| 2 | Direction | DNN 256-128-64-32 | 5 min | accuracy 50.59% (balanced 51.37%) | majority (UP) 49.90% | +0.69 pp [+0.40, +0.98] |
| 3 | Direction | DNN 64-32 | 15 min | accuracy 50.99% (balanced 50.83%) | majority (UP) 50.74% | +0.25 pp [−0.07, +0.57] |
| 4 | Volatility | DNN 128-64-32 | 5 min | log-RV RMSE 0.901; RV UP/DOWN 72.2% | persistence 1.163; majority 51.1% | R² +40.0% [+39.3, +40.6]; direction +21.1 pp |
| 5 | Volatility | XGBoost | 1 hour | log-RV RMSE 0.572; RV UP/DOWN 74.8% | persistence 0.825; majority 47.7% | R² +52.0% [+49.5, +54.8]; direction +27.1 pp |

Accuracy counts dojis as misses for every call, so the majority baseline
can be below 50%.

* **Return:** no model beats the zero forecast on the test block; the
  selected network's forecasts span less than ±0.5 bp against a typical
  5-minute move of 7.7 bp.
* **Direction and trading (ID 2, $2.50 per round trip, idealised fills):**
  Long & Short −$515.9k on 273,340 trades (gross +$167.4k, $0.61 per trade);
  long only −$307.7k; short only −$208.3k; at $5 per round trip −$1.20M. A
  small but measurable accuracy gain does not pay the commission. Every
  direction model × timeframe × strategy is in `trading_direction_test.csv`:
  some daily and 30-minute configurations are positive on the test, but they
  were not selected on validation and are multiple comparisons, not findings.
* **Volatility:** intraday, much of the predictability is volatility
  clustering, mean reversion and the time-of-day pattern. The linear HAR-OLS
  benchmark already removes 36% (5 min) and 35% (1 h) of persistence's
  error; the ML models add 4 and 17 points. Intraday, every trained
  volatility model beats persistence on the test (log-RV R² +29% to +52%);
  on daily candles the gain is smaller and not universal (+9% to +22%,
  random forest −19%; `results_volatility.csv`). The RV UP/DOWN accuracy of 72–75% reflects the
  same effects. It concerns realized, not implied, volatility: it says
  nothing about option prices or option P&L, which were not computed.
* **Mistral / Timestral-5M (zero-shot, test sample):** return R² −1.5% to
  −2.5% on every timeframe (worse than forecasting zero); direction 0 to
  −2.3 pp vs the majority class; volatility better than persistence only at
  5 min (R² +12%), worse elsewhere. Not among the five.
* **Deeper networks:** no consistent gain. On the selected configurations
  the best network depth differs by task and timeframe (256-128-64-32,
  128-64-32 and 64-32), and differences between networks are a fraction of
  the gap to the benchmark (`results_*.csv`).

### Deviations, found while reporting (stated on the slides)

1. **Direction majority bug.** The first run computed the majority class over
   all training rows, counting dojis as "not up", so for 5, 15 and 30 minutes
   it chose DOWN, a weak baseline that inflated the gains to about +2 pp.
   Fixed in `multitarget.majority_direction` (non-doji candles only).
   `tools/mt_fix_majority.py` recomputed the constant from the training
   data, without refitting any model; the previous file is kept as
   `dataset_info_before_majority_fix.json`.
2. **Volatility selection score.** The rule first used QLIKE vs persistence.
   On validation, >99% of persistence's QLIKE came from a handful of
   limit-locked candles in March 2020 with almost zero variance, so the score
   was meaningless (z ≈ 1 for every model). It was replaced by the squared
   error of log RV vs persistence. **This change was made after the first
   report had printed test numbers**, so the volatility selection is not
   blind. The conclusion does not hinge on it: intraday every trained
   volatility model beats persistence on the test by a wide margin. QLIKE is
   still reported.

## Data

* Instrument: **NQ**, E-mini Nasdaq-100 futures (CME), back-adjusted
  continuous TradeStation 1-minute series, 2006-01 to 2025-05 (sessions before
  the 2025-06-02 confirmatory hold-out), $20 per index point. No QQQ, MNQ,
  cash index or VIX data is mixed in.
* Timeframes: 5 min, 15 min, 30 min, 1 hour, 1 day (one CME session,
  17:00–16:00 CT), all rebuilt from the 1-minute bars inside each session
  (`timeframes.aggregate`; `30min` was added to `FREQUENCIES` in this branch).
  1-minute candles are excluded: the realized variance of a 1-minute candle
  from 1-minute data would be a single squared return.
* Splits, identical for every timeframe and task: train 2006-01 → 2017-08-24,
  validation 2017-08-25 → 2021-07-14, test 2021-07-15 → 2025-05-30. Labels
  that cross a split boundary are purged; intraday labels never cross a
  session. Row counts per timeframe: `dataset_info.json` and slide 3.
* The test period was already inspected for the direction task in earlier
  project phases, so the test results are **exploratory**, not a clean
  confirmation.

## Targets (all for candle t+1, decided at the close of candle t)

| Task | Target | Benchmark | Metrics |
|---|---|---|---|
| Return | r = log(C(t+1)/C(t)) in basis points | zero return (also: training mean) | MAE, RMSE, out-of-sample R² vs zero, sign accuracy, correlation |
| Direction | UP if C(t+1) > O(t+1), DOWN if lower; doji kept and counted as a miss | majority class of the training set | accuracy, balanced accuracy, precision, recall, F1, confusion matrix |
| Volatility | RV(t+1) = Σ squared 1-minute log returns inside candle t+1 (first term log(C/O) of its first minute) | persistence RV(t+1) = RV(t) (also: rolling mean of 22 candles, HAR-OLS) | MAE/RMSE of log RV, QLIKE on the variance level, correlation; RV direction (UP if RV(t+1) > RV(t)): accuracy, balanced accuracy, precision, recall, F1 vs majority |

Squared returns are a variance proxy; volatility is its square root. Models
forecast log RV; the level forecast used for QLIKE is
`c · exp(log forecast)`, with `c` estimated on the training residuals.

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
  pretrained time-series foundation model). Role in the repository:
  `tools/timestral_direction.py` (restored in this branch, unchanged) ran it
  zero-shot for next-close direction; in the final direction study it had
  been removed. Here it is used zero-shot on all three tasks: from the last
  512 closes of the same timeframe it forecasts quantiles of the next close;
  the median gives the return and the direction, and the q10–q90 spread
  gives a variance forecast, mapped to log RV by a linear calibration fitted
  on the validation sample only. It runs from local weights (Hugging Face
  cache, `HF_HUB_OFFLINE=1`), with no API and no cost. It is not a language
  model and does not read text. It is scored on a fixed random sample of
  20,000 validation and 20,000 test origins per timeframe (all of them for
  daily candles), and compared with the benchmarks on exactly those rows.
  Pretraining-data overlap with NQ is unknown. Larger sizes (28M–209M) need
  new downloads and were not run. There is no Mistral chat/LLM API
  integration in the repository.
* Considered, not run: LSTM. The feature networks already reach their
  early-stopping optimum within a few epochs, and a sequence model would add
  cost without a reason in the validation results.

## Selection of the five configurations (validation only)

`multitarget_report.py`, rule fixed before the test results were read:

1. Evidence score per configuration = z-statistic (95% moving-block bootstrap
   with 20-session blocks) of its paired **validation** improvement over the
   task's naive benchmark: squared-error reduction vs zero (return), accuracy
   gain vs majority (direction), QLIKE reduction vs persistence (volatility).
   Benchmarks are never selected.
2. Each task gets its highest-z configuration.
3. The two remaining slots go to the two tasks whose best configuration on a
   **different timeframe** has the highest z (at most two per task).
4. The five are scored once on the test block.

## Trading test (direction task)

UP → long, DOWN → short (Long & Short); Long only; Short only. Entry at the
open of candle t+1, exit at its close, one NQ contract ($20 per point),
**$2.50 per round trip** (commission only), zero slippage, 100% fills, zero
latency: idealised. Real limit orders are not guaranteed to fill. Net P&L at
$5 per round trip (the main study's assumption) is reported next to it.
Drawdown is measured on trade-by-trade equity starting at 0.

## Reproduce

Windows PowerShell, Python 3.12 (`requirements.txt`, plus
`requirements-timestral.txt` for Timestral; the run used torch 2.15.0+cpu,
xgboost 3.4.1, scikit-learn 1.9.1, timestral 1.0.0), and the Timestral-5M
weights in the Hugging Face cache (first download:
`python -c "from timestral import TimestralForecaster; TimestralForecaster.from_pretrained('mistralai/Timestral-5M')"`
with `HF_HUB_OFFLINE` unset).

```powershell
$env:FUTURES_DATA_DIR = "C:\path\to\futures-data"
python multitarget.py --tag final                       # ~30 min on 16 CPU threads
python multitarget_report.py results\multitarget\runs\<run id> reports\multitarget_final
copy reports\multitarget_final\deck_data.json, reports\multitarget_final\chart_*.png presentation_multitarget\
cd presentation_multitarget; node build_deck.js NASDAQ_multitarget_report.pptx
python -m pytest -q
```

Pipeline check in a minute: `python multitarget.py --freqs 1h session --years 2015 2019 --quick --timestral-n 500`.

Outputs of `multitarget.py` (`results/multitarget/runs/<id>/`):
`predictions_<freq>.parquet` (validation and test rows, every model's
forecast), `models.csv`, `dataset_info.json`, `console.log`, `run.json`.
Outputs of `multitarget_report.py`: `results_return.csv`,
`results_direction.csv`, `results_volatility.csv` (every model × timeframe ×
split, plus the Timestral-sample rows), `trading_direction_test.csv` (every
direction model × timeframe × strategy), `selection_validation_ranking.csv`,
`selected_five.csv`, `configurations_tested.csv`, `chart_cfg<id>_*.png`,
`deck_data.json`.
