# Financial Market Direction Prediction Using Machine Learning

University project for *AI, Machine Learning and Big Data for Finance and Management*.

**Question:** can simple ML models predict whether the next price move is
**UP (1)** or **DOWN (0)**, and does this change across **tick**, **hourly** and
**daily** data?

The project is designed as two experiments:

1. **Synthetic experiment:** do ML models wrongly appear to find predictive
   patterns in a market whose returns are independent (Geometric Brownian
   Motion)? The honest answer should be "no": about 50% accuracy, no better
   than a simple baseline. (Results below.)
2. **Real-market experiment:** do real market features predict the next
   move's direction out of sample, better than simple baselines, and does
   this differ by frequency? Same pipeline on NQ (Nasdaq-100 E-mini futures)
   1-minute bars, 2006-2025. (Results in
   [Real-market experiment](#real-market-experiment-nq-futures-2006-2025).)

---

## How to run

```bash
pip install -r requirements.txt
python main.py
```

`DATA_SOURCE` in `config.py` chooses the experiment:

* `"synthetic"`: 5 simulated paths × 3 frequencies × 7 models, about
  5 minutes on a laptop. Output in `results/synthetic/`.
* `"nq"`: real NQ futures. Needs a local clone of the `futures-data`
  repository; tell the program where it is (PowerShell:
  `$env:FUTURES_DATA_DIR="C:\path\to\futures-data"`). Output in
  `results/nq/`. The full 2006-2025 run is much slower (the models train on
  2.7 million 1-minute rows); for a quick test set
  `FUTURES_YEARS = (2023, 2025)` (about 2 minutes).

This folder is not tracked by Git (it is in `.gitignore`): every run
regenerates it, so each person keeps their own copy and `git pull` never
conflicts with it.

To change the simulation (drift, volatility, length, seeds, trade rate), the
features, the split or the model hyperparameters, edit `config.py`. To run
faster while experimenting, use a single seed: `SEEDS = [42]`.

---

## Project structure

| File | Purpose |
|---|---|
| `config.py` | All parameters (GBM, ticks, seeds, features, split, model hyperparameters) |
| `data.py` | GBM tick simulation, OHLC aggregation, aggregation checks; loading, cleaning and aggregating real 1-minute futures bars |
| `features.py` | Features, UP/DOWN target, common calendar 60/20/20 split with gap |
| `models.py` | Majority baseline, Logistic Regression, Decision Tree, Random Forest, two small neural networks, XGBoost (optional) |
| `evaluation.py` | Accuracy, balanced accuracy, F1, confusion matrix, improvement over baseline |
| `plots.py` | All figures |
| `main.py` | Runs the synthetic (all seeds) or the real NQ experiment |
| `tools/timestral_direction.py` | Zero-shot UP/DOWN test with Mistral's Timestral on the NQ test period |
| `tools/download_ticks_ibkr.py` | Downloads recent NQ tick data (CME trades) from Interactive Brokers |
| `tools/download_ticks_databento.py` | Cost estimate / download of CME tick data from Databento (paid) |
| `tools/check_ticks.py` | Summary of downloaded tick files |

---

## Method

### 1. Synthetic tick data: one GBM path, three frequencies

Prices follow $dS_t = \mu S_t\,dt + \sigma S_t\,dW_t$.

**Ticks = trades at random times.** Real tick data is irregular: trades
arrive whenever someone trades. The simulation reproduces this. Trade times
follow a **Poisson process**: the waiting times between trades are independent
exponential variables, with on average `TICKS_PER_HOUR = 20` trades per hour.
Between two trades the price moves with the exact GBM solution over the
actual elapsed time $\Delta t_i$:

$$S_{i}=S_{i-1}\exp\!\left[\left(\mu-\tfrac12\sigma^2\right)\Delta t_i+\sigma\sqrt{\Delta t_i}\,Z_i\right],\qquad Z_i\sim N(0,1)\ \text{iid}$$

| Parameter | Value |
|---|---|
| Initial price | 100 |
| Annual drift $\mu$ | 8% |
| Annual volatility $\sigma$ | 20% |
| Length | 3 years, 24/7 market, 365 days/year |
| Trade arrivals | Poisson, 20 trades/hour on average (≈ 525,000 ticks per path) |
| Seeds | 42, 43, 44, 45, 46 (five independent paths) |

Each seed gives **one** tick path. Hourly and daily candles are built from
that same path, so the number of ticks per candle varies (3–46 per hour):

| Dataset | Observations per path | How it is built |
|---|---|---|
| Tick | ≈ 525,000 | every simulated trade |
| Hourly | 26,280 | OHLC of the trades in each hour |
| Daily | 1,095 | OHLC of the trades in each day |

`main.py` checks the aggregation on every run:

* every hour and day is present and no candle is empty;
* no tick is lost or counted twice;
* daily OHLC equals the hourly OHLC re-aggregated;
* each close is the last trade of its period, and high/low bound every candle.

It also standardises each tick return by its own elapsed time. Theory says
the result is iid N(0, 1), and the simulation matches: mean ≈ 0, std ≈ 1.000
and autocorrelation ≈ 0 for every seed.

Trade sizes (volume) are iid noise, independent of prices, so they carry no
signal and are not used as a feature.

*Why prices are not rounded to a tick size (e.g. 0.01):* rounding would create
unchanged prices, but it also adds an artificial pattern (consecutive price
changes become negatively correlated) that the models could pick up. Leaving it
out keeps the synthetic market free of any learnable signal.

### 2. Target

* `1` = next close > current close
* `0` = next close < current close
* Unchanged next close → row removed (the count is reported in
  `dataset_summary.csv`; it is 0 here because prices are continuous, but it
  matters for real tick data)

### 3. Features (same for all frequencies, past/current data only)

| Feature | Definition (r = log return) |
|---|---|
| `ret_lag1`, `ret_lag2`, `ret_lag3`, `ret_lag5` | $r_t, r_{t-1}, r_{t-2}, r_{t-4}$ (lag 1 = most recent return) |
| `mom_5`, `mom_10` | $\ln(P_t / P_{t-n})$ |
| `vol_5`, `vol_10` | rolling std of $r$ over the last $n$ periods |

At tick level a "period" is one trade, so these are trade-by-trade features.
The first 10 rows (warm-up) and the last row (no next price) are dropped.

*Limitation of the tick volatility feature:* trades are irregularly spaced, so
the last 5 or 10 tick returns cover different amounts of clock time. At tick
level, `vol_5` and `vol_10` therefore measure volatility per trade, not per
unit of time. The feature is kept as is for simplicity. When moving to real
data, consider adding the elapsed time between trades as a feature and
dividing each tick return by $\sqrt{\Delta t}$ before taking the rolling std.

### 4. Training and testing

* **Common calendar split:** the 1,095 days are split 60% / 20% / 20% into
  train / validation / test, with no shuffling. The **same dates** apply to
  tick, hourly and daily data (here: training until 2024-10-18, validation
  from 2024-10-19, test from 2025-05-26). Every frequency is therefore tested
  on exactly the same period. A split by observation count would put the
  boundaries at slightly different times, because ticks arrive irregularly.
* **One-observation gap** at the end of training and of validation
  (`SPLIT_GAP = 1`). The label at time $t$ uses the price at $t+1$, so the
  last row before a boundary has a label that reaches into the next period.
  That row is dropped, so every label depends only on prices inside its own
  period. The gap is one observation at each frequency: one trade, one hour
  or one day.
* **No leakage from scaling:** every model sits in a
  `Pipeline(StandardScaler → model)`, so the scaling statistics are fitted on
  the training set only.
* **Validation for selection, test for evaluation:** the models are trained on
  the training set. The model with the best validation balanced accuracy is
  flagged in `selected_on_val`. The test set is used only for the final
  evaluation.
* **Same hyperparameters for every frequency** (`config.py`):
  * Logistic Regression: C=1
  * Decision Tree: depth 5, min 50 samples per leaf
  * Random Forest: 200 trees, depth 6, min 50 samples per leaf
  * XGBoost: 200 trees, depth 3, learning rate 0.05
  * Neural network, 1 hidden layer: 16 neurons
  * Neural network, 3 hidden layers: 64 → 32 → 16 neurons
  * Both networks (scikit-learn `MLPClassifier`): ReLU, Adam, L2 penalty
    0.001, batch 256, max 200 epochs. Early stopping is off because it would
    draw a *shuffled* validation sample from the training data.
* **Baseline:** always predict the majority class **of the training set**.
* **Five seeds:** the experiment is repeated on five independent GBM paths
  with the same parameters. The mean and the spread across seeds show how much
  of any single result is luck.

### How results are reported

* **Improvement over baseline** (`acc_vs_baseline_pp`): the model's test
  accuracy minus the majority baseline's accuracy on the same test sample, in
  percentage points. This is the headline number, because a model is only
  useful if it beats the trivial rule. It is reported for each seed, and as
  mean ± std across seeds, together with how many of the 5 seeds the model
  beat the baseline on.
* **Balanced accuracy:** the baseline always scores exactly 0.500, so balanced
  accuracy above 0.5 also means "better than the baseline".
* **Accuracy ± 95% interval** (`acc_ci95`): the normal approximation to the
  binomial. It is only a rough guide to how much accuracy moves by chance
  alone on a test set of that size. It is **not** a test of whether a model
  beats the baseline; use the improvement and its variation across seeds for
  that.

### Models highlighted in the presentation

All seven models stay in the code. The presentation focuses on four, which go
from simple to complex: **Logistic Regression → Random Forest → NN with 1
hidden layer → NN with 3 hidden layers**, compared with the majority baseline.

---

## Results: synthetic experiment

### Across 5 seeds (main result)

Mean ± std over five independent GBM paths. "Beats baseline" counts the seeds
on which the model's test accuracy was above the baseline's.

| Frequency | Model | Accuracy | Improvement over baseline (pp) | Beats baseline | Balanced acc. |
|---|---|---|---|---|---|
| tick | Majority baseline | 0.501 ± 0.002 | – | – | 0.500 |
| tick | Logistic Regression | 0.499 ± 0.002 | −0.16 ± 0.07 | 0 / 5 | 0.499 |
| tick | Random Forest | 0.500 ± 0.003 | −0.03 ± 0.21 | 2 / 5 | 0.501 |
| tick | NN 1 layer | 0.500 ± 0.002 | −0.09 ± 0.25 | 3 / 5 | 0.500 |
| tick | NN 3 layers | 0.500 ± 0.002 | −0.10 ± 0.17 | 1 / 5 | 0.500 |
| hourly | Majority baseline | 0.506 ± 0.005 | – | – | 0.500 |
| hourly | Logistic Regression | 0.501 ± 0.005 | −0.57 ± 0.44 | 0 / 5 | 0.499 |
| hourly | Random Forest | 0.494 ± 0.009 | −1.26 ± 0.96 | 1 / 5 | 0.492 |
| hourly | NN 1 layer | 0.495 ± 0.006 | −1.13 ± 0.91 | 1 / 5 | 0.494 |
| hourly | NN 3 layers | 0.503 ± 0.005 | −0.35 ± 0.44 | 1 / 5 | 0.503 |
| daily | Majority baseline | 0.516 ± 0.052 | – | – | 0.500 |
| daily | Logistic Regression | 0.504 ± 0.042 | −1.19 ± 1.67 | 0 / 5 | 0.502 |
| daily | Random Forest | 0.489 ± 0.053 | −2.66 ± 6.71 | 2 / 5 | 0.492 |
| daily | NN 1 layer | 0.520 ± 0.020 | +0.46 ± 4.39 | 3 / 5 | 0.518 |
| daily | NN 3 layers | 0.513 ± 0.045 | −0.28 ± 5.20 | 4 / 5 | 0.515 |

Decision Tree and XGBoost behave the same way; see `results/multi_seed_summary.csv`.

### Single path in detail (seed 42)

Observations per split (train / validation / test): tick 315,932 / 105,343 /
105,249; hourly 15,757 / 5,255 / 5,255; daily 646 / 218 / 218. Share of UP
moves in each split: `results/dataset_summary.csv`.

| Frequency | Model | Test obs. | Accuracy ± approx. 95% CI | vs baseline (pp) | Balanced acc. | F1 |
|---|---|---|---|---|---|---|
| tick | Majority baseline | 105,249 | 0.501 ± 0.003 | – | 0.500 | 0.667 |
| tick | Logistic Regression | 105,249 | 0.499 ± 0.003 | −0.2 | 0.499 | 0.472 |
| tick | Decision Tree | 105,249 | 0.501 ± 0.003 | −0.0 | 0.501 | 0.199 |
| tick | Random Forest | 105,249 | 0.500 ± 0.003 | −0.0 | 0.501 | 0.432 |
| tick | NN 1 layer | 105,249 | 0.501 ± 0.003 | +0.0 | 0.500 | 0.595 |
| tick | NN 3 layers | 105,249 | 0.500 ± 0.003 | −0.0 | 0.501 | 0.357 |
| tick | XGBoost | 105,249 | 0.498 ± 0.003 | −0.3 | 0.498 | 0.468 |
| hourly | Majority baseline | 5,255 | 0.511 ± 0.013 | – | 0.500 | 0.677 |
| hourly | Logistic Regression | 5,255 | 0.499 ± 0.013 | −1.2 | 0.497 | 0.543 |
| hourly | Decision Tree | 5,255 | 0.489 ± 0.013 | −2.2 | 0.491 | 0.436 |
| hourly | Random Forest | 5,255 | 0.493 ± 0.013 | −1.9 | 0.491 | 0.529 |
| hourly | NN 1 layer | 5,255 | 0.494 ± 0.013 | −1.7 | 0.494 | 0.505 |
| hourly | NN 3 layers | 5,255 | 0.511 ± 0.013 | −0.0 | 0.512 | 0.491 |
| hourly | XGBoost | 5,255 | 0.500 ± 0.013 | −1.1 | 0.500 | 0.511 |
| daily | Majority baseline | 218 | 0.541 ± 0.066 | – | 0.500 | 0.702 |
| daily | Logistic Regression | 218 | 0.541 ± 0.066 | +0.0 | 0.541 | 0.561 |
| daily | Decision Tree | 218 | 0.505 ± 0.066 | −3.7 | 0.501 | 0.542 |
| daily | Random Forest | 218 | 0.546 ± 0.066 | +0.5 | 0.539 | 0.596 |
| daily | NN 1 layer | 218 | 0.537 ± 0.066 | −0.5 | 0.541 | 0.535 |
| daily | NN 3 layers | 218 | 0.578 ± 0.066 | +3.7 | 0.579 | 0.593 |
| daily | XGBoost | 218 | 0.537 ± 0.066 | −0.5 | 0.538 | 0.547 |

F1 is computed for the UP class. When the baseline predicts only UP, its F1 is
high even though it has no skill; when it predicts only DOWN, its F1 is 0.
This is why balanced accuracy and the improvement over baseline are the
fairer headline metrics.

### Interpretation

* **No model beats the baseline.** Averaged over five independent paths, the
  improvement over the majority baseline is between −2.7 and +0.5 pp for every
  model at every frequency. The only positive mean (daily 1-layer network,
  +0.46 pp) comes with a seed-to-seed std of 4.4 pp, so it is indistinguishable
  from zero. Some models beat the baseline on a few seeds (at most 4 out of 5),
  but never by a stable margin. Past returns, momentum and
  volatility carry no information about the next move in a GBM market, and
  the pipeline correctly reports that.
* **More complexity does not help.** Going from Logistic Regression to Random
  Forest to a 1-layer and then a 3-layer neural network brings no consistent
  improvement. On hourly and daily data the networks hit the 200-epoch limit
  without converging: they keep fitting noise in the training set
  (overfitting).
* **Single daily results are mostly luck.** With only 218 daily test
  observations, a model's improvement over the baseline on a single path
  ranges from −14 pp to +5 pp. The baseline's own accuracy ranges from 46%
  to 59%, depending on whether the test period happened to trend. At tick
  level the same range is only about ±0.5 pp. Even moving the test period by
  a single day changed one daily result by 6 pp.
* **The drift is too small to show up.** In theory P(UP) is about 50.03% per
  tick, 50.1% per hour and 50.6% per day, which is smaller than the sampling
  noise.
* **Lesson for real data:** a result should be judged by its improvement over
  a simple baseline on the same test period, and by how stable that
  improvement is across samples. A single daily backtest could easily show
  55% accuracy from pure noise.

---

## Real-market experiment: NQ futures 2006-2025

### Data

* **Instrument:** NQ, the E-mini Nasdaq-100 future traded on CME Globex.
  It is a traded instrument with nearly 23 hours of trading per day.
  (The cash index NDX is not used: it is a calculated value that cannot be
  traded, and its stale component prices make short-horizon returns look
  predictable when they are not.)
* **Source:** 1-minute OHLCV bars exported from TradeStation, stored in the
  `futures-data` repository: 6,193,869 bars from 2006-01-03 to 2025-05-30.
  Continuous contract, back-adjusted at each roll by adding a constant.
* **Cleaning** (`data.clean_minute_bars`): 35 Saturday bars and 5 isolated
  bad prints removed. The bad prints are single bars about 990 points away
  from both neighbours (a price that was not back-adjusted, e.g.
  2010-03-16 23:00); they are detected as a 1-minute move above 2% that
  immediately reverses.
* **Frequencies:** the 1-minute bars themselves, plus hourly and daily candles
  built from the same minutes (`data.minute_datasets`). Daily candles follow
  the CME session, not the calendar day: the export is in Italian time, so in
  the weeks when US and European daylight saving time differ a session starts
  at 23:01 of the previous day. Checked against TradeStation's own daily file:
  open, high and low match on 95-99.6% of days; the close differs slightly
  because TradeStation uses the official settlement price, while we use the
  last trade of the session.
* **Split:** same calendar dates for all frequencies, 60/20/20 of the
  trading days: training until 2017-08-21, validation 2017-08-22 to
  2021-07-13, test 2021-07-14 to 2025-05-30. Because recent years have more
  traded minutes, the 1-minute split is 2.71M / 1.24M / 1.32M observations.
* Unchanged closes are dropped: 15% of 1-minute bars (927,991), 3% of hours,
  16 days.

### Results (test set 2021-07-14 to 2025-05-30)

| Frequency | Model | Test obs. | Accuracy ± approx. 95% CI | vs baseline (pp) | Balanced acc. |
|---|---|---|---|---|---|
| 1-min | Majority baseline | 1,318,336 | 0.504 ± 0.001 | – | 0.500 |
| 1-min | Logistic Regression | 1,318,336 | 0.513 ± 0.001 | +0.91 | 0.513 |
| 1-min | Decision Tree | 1,318,336 | 0.513 ± 0.001 | +0.84 | 0.513 |
| 1-min | Random Forest | 1,318,336 | 0.513 ± 0.001 | +0.91 | 0.513 |
| 1-min | NN 1 layer | 1,318,336 | 0.514 ± 0.001 | +0.95 | 0.513 |
| 1-min | NN 3 layers | 1,318,336 | 0.514 ± 0.001 | +0.97 | 0.514 |
| 1-min | XGBoost | 1,318,336 | 0.513 ± 0.001 | +0.94 | 0.513 |
| hourly | Majority baseline | 23,769 | 0.517 ± 0.006 | – | 0.500 |
| hourly | Logistic Regression | 23,769 | 0.506 ± 0.006 | −1.14 | 0.495 |
| hourly | Decision Tree | 23,769 | 0.509 ± 0.006 | −0.86 | 0.503 |
| hourly | Random Forest | 23,769 | 0.516 ± 0.006 | −0.12 | 0.507 |
| hourly | NN 1 layer | 23,769 | 0.505 ± 0.006 | −1.19 | 0.500 |
| hourly | NN 3 layers | 23,769 | 0.504 ± 0.006 | −1.39 | 0.500 |
| hourly | XGBoost | 23,769 | 0.508 ± 0.006 | −0.98 | 0.501 |
| daily | Majority baseline | 1,002 | 0.533 ± 0.031 | – | 0.500 |
| daily | Logistic Regression | 1,002 | 0.524 ± 0.031 | −0.90 | 0.508 |
| daily | Decision Tree | 1,002 | 0.525 ± 0.031 | −0.80 | 0.514 |
| daily | Random Forest | 1,002 | 0.531 ± 0.031 | −0.20 | 0.500 |
| daily | NN 1 layer | 1,002 | 0.502 ± 0.031 | −3.09 | 0.488 |
| daily | NN 3 layers | 1,002 | 0.486 ± 0.031 | −4.69 | 0.479 |
| daily | XGBoost | 1,002 | 0.513 ± 0.031 | −2.00 | 0.506 |

### Interpretation

* **1 minute: a small but real effect.** Every model beats the baseline by
  about +0.9 pp (balanced accuracy 0.513). With 1.3 million test minutes this
  is far outside chance (±0.1 pp). The Random Forest puts 57% of its
  importance on the last 1-minute return, and a one-line rule, "predict the
  opposite of the last move", scores 51.2% on the same test period, stable in
  every year 2021-2025 (51.0-51.4%). This is the **bid-ask bounce**: trade
  prices jump between the bid and the ask, so an up-minute is slightly more
  often followed by a down-minute. Complex models add nothing over this rule.
* **It is not a trading opportunity.** A 1-minute move is typically one or
  two ticks; a 51% hit rate on such moves does not cover the bid-ask spread
  and commissions.
* **Hourly and daily: no predictability.** No model beats the majority
  baseline (improvements from −4.7 to −0.1 pp). The baseline itself is above
  50% (51.7% hourly, 53.3% daily) only because NQ rose over the test period.
* **Compared with the synthetic experiment:** GBM has no bid-ask bounce, so
  there even the 1-minute level showed nothing. Real data differs from
  the random walk only at the shortest horizon, and only through market
  microstructure.

### Limitations

* **Back-adjusted prices:** additive back-adjustment keeps price
  *differences* (and therefore every UP/DOWN label) exact, but percentage
  returns in early years are smaller than the real ones (in 2006 the series
  is about 2,400 points above the actual price). Feature scales therefore
  drift over the 20 years.
* **Labels across breaks:** the "next close" after a session break or a
  weekend is the first bar of the next session. These are a tiny share of
  the 1-minute observations.
* **One history:** unlike the synthetic experiment, there is a single real
  path, so there is no seed-to-seed spread. The yearly stability of the
  1-minute effect plays that role.

---

## Output files (`results/`)

Synthetic run: `results/synthetic/`; NQ run: `results/nq/` (same files except
the multi-seed ones; Timestral results go to `results/nq/timestral_results.csv`).

| File | Content |
|---|---|
| `01_price_path.png` | Simulated tick path (or NQ hourly closes) and daily closes |
| `02_candlesticks.png` | Sample daily and hourly candlestick charts |
| `03_accuracy_comparison.png` | Test accuracy by model and frequency, seed 42, with approx. 95% CI |
| `04_results_table.png` | Results table with improvement over baseline (seed 42, for slides) |
| `05_rf_feature_importance.png` | Random Forest feature importance per frequency (seed 42) |
| `06_confusion_matrices.png` | Test-set confusion matrices (seed 42) |
| `07_multi_seed_accuracy.png` | **Focus models across 5 seeds**: mean ± std, individual seeds, baseline range |
| `results_table.csv` | Full metrics for seed 42 (test, validation, confusion-matrix counts) |
| `results_all_seeds.csv` | The same for every seed |
| `multi_seed_summary.csv` | Mean / std / min / max across seeds, improvement over baseline, seeds beating it |
| `dataset_summary.csv` | Observations, period dates, dropped rows and class distribution per split, per seed |
| `rf_feature_importance.csv` | Feature importance values (seed 42) |

---

## Timestral (Mistral's pretrained forecasting model)

[Timestral](https://github.com/mistralai/timestral) is a pretrained
time-series foundation model (5M-209M parameters) that forecasts without
training on our data ("zero-shot"). `tools/timestral_direction.py` asks it
for the next close given the last 512 closes and predicts UP if the forecast
median is above the current close. It is scored on a random sample of the
same NQ test bars (default 2,000 per frequency), against the same majority
baseline.

```bash
pip install git+https://github.com/mistralai/timestral.git   # needs torch >= 2.4
python tools/timestral_direction.py                            # Timestral-5M
python tools/timestral_direction.py --model mistralai/Timestral-209M --n-samples 500
```

The weights are downloaded from Hugging Face on the first run (Modified MIT
licence; the code is Apache 2.0).
