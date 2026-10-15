# Financial Market Direction Prediction Using Machine Learning

University project for *AI, Machine Learning and Big Data for Finance and Management*.

**Question:** can simple ML models predict whether the next price move is
**UP (1)** or **DOWN (0)**, and does this change across **tick**, **hourly** and
**daily** data?

The project is designed as two experiments:

1. **Synthetic experiment (this stage):** do ML models wrongly appear to
   find predictive patterns in a market whose returns are independent
   (Geometric Brownian Motion)? The honest answer should be "no": about 50%
   accuracy, no better than a simple baseline.
2. **Real-market experiment (next stage):** do real market features predict
   the next move's direction out of sample, better than simple baselines, and
   does this differ by frequency? The same pipeline is reused.

---

## How to run

```bash
pip install -r requirements.txt
python main.py
```

The full run (5 simulated paths × 3 frequencies × 7 models) takes about
5 minutes on a laptop. All tables and figures are written to `results/`.
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
| `data.py` | GBM tick simulation, OHLC aggregation, aggregation checks, CSV loader for real data |
| `features.py` | Features, UP/DOWN target, common calendar 60/20/20 split with gap |
| `models.py` | Majority baseline, Logistic Regression, Decision Tree, Random Forest, two small neural networks, XGBoost (optional) |
| `evaluation.py` | Accuracy, balanced accuracy, F1, confusion matrix, improvement over baseline |
| `plots.py` | All figures |
| `main.py` | Runs the full experiment for every seed |

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
* **One-observation gap** at the start of validation and of test
  (`SPLIT_GAP = 1`). The label at time $t$ uses the price at $t+1$. Without
  the gap, the last training label would depend on the first price of the
  validation period. The gap is one observation at each frequency: one trade,
  one hour or one day.
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

## Results

### Across 5 seeds (main result)

Mean ± std over five independent GBM paths. "Beats baseline" counts the seeds
on which the model's test accuracy was above the baseline's.

| Frequency | Model | Accuracy | Improvement over baseline (pp) | Beats baseline | Balanced acc. |
|---|---|---|---|---|---|
| tick | Majority baseline | 0.501 ± 0.002 | – | – | 0.500 |
| tick | Logistic Regression | 0.499 ± 0.002 | −0.17 ± 0.06 | 0 / 5 | 0.499 |
| tick | Random Forest | 0.501 ± 0.003 | −0.03 ± 0.23 | 2 / 5 | 0.501 |
| tick | NN 1 layer | 0.500 ± 0.002 | −0.10 ± 0.16 | 1 / 5 | 0.500 |
| tick | NN 3 layers | 0.500 ± 0.002 | −0.06 ± 0.16 | 1 / 5 | 0.500 |
| hourly | Majority baseline | 0.506 ± 0.005 | – | – | 0.500 |
| hourly | Logistic Regression | 0.500 ± 0.006 | −0.62 ± 0.48 | 0 / 5 | 0.499 |
| hourly | Random Forest | 0.494 ± 0.010 | −1.24 ± 1.04 | 1 / 5 | 0.493 |
| hourly | NN 1 layer | 0.496 ± 0.005 | −1.00 ± 0.90 | 1 / 5 | 0.496 |
| hourly | NN 3 layers | 0.502 ± 0.011 | −0.43 ± 1.41 | 1 / 5 | 0.502 |
| daily | Majority baseline | 0.515 ± 0.054 | – | – | 0.500 |
| daily | Logistic Regression | 0.497 ± 0.042 | −1.84 ± 1.38 | 0 / 5 | 0.495 |
| daily | Random Forest | 0.488 ± 0.047 | −2.77 ± 5.83 | 1 / 5 | 0.491 |
| daily | NN 1 layer | 0.515 ± 0.019 | +0.00 ± 4.97 | 3 / 5 | 0.514 |
| daily | NN 3 layers | 0.511 ± 0.006 | −0.46 ± 5.10 | 2 / 5 | 0.513 |

Decision Tree and XGBoost behave the same way; see `results/multi_seed_summary.csv`.

### Single path in detail (seed 42)

Observations per split (train / validation / test): tick 315,933 / 105,343 /
105,248; hourly 15,758 / 5,255 / 5,254; daily 647 / 218 / 217. Share of UP
moves in each split: `results/dataset_summary.csv`.

| Frequency | Model | Test obs. | Accuracy ± approx. 95% CI | vs baseline (pp) | Balanced acc. | F1 |
|---|---|---|---|---|---|---|
| tick | Majority baseline | 105,248 | 0.501 ± 0.003 | – | 0.500 | 0.667 |
| tick | Logistic Regression | 105,248 | 0.499 ± 0.003 | −0.2 | 0.499 | 0.472 |
| tick | Decision Tree | 105,248 | 0.501 ± 0.003 | −0.0 | 0.501 | 0.199 |
| tick | Random Forest | 105,248 | 0.500 ± 0.003 | −0.0 | 0.500 | 0.431 |
| tick | NN 1 layer | 105,248 | 0.501 ± 0.003 | −0.0 | 0.501 | 0.430 |
| tick | NN 3 layers | 105,248 | 0.500 ± 0.003 | −0.1 | 0.500 | 0.419 |
| tick | XGBoost | 105,248 | 0.499 ± 0.003 | −0.1 | 0.499 | 0.468 |
| hourly | Majority baseline | 5,254 | 0.511 ± 0.013 | – | 0.500 | 0.677 |
| hourly | Logistic Regression | 5,254 | 0.499 ± 0.013 | −1.3 | 0.497 | 0.543 |
| hourly | Decision Tree | 5,254 | 0.489 ± 0.013 | −2.2 | 0.491 | 0.436 |
| hourly | Random Forest | 5,254 | 0.494 ± 0.013 | −1.7 | 0.493 | 0.529 |
| hourly | NN 1 layer | 5,254 | 0.496 ± 0.013 | −1.6 | 0.494 | 0.532 |
| hourly | NN 3 layers | 5,254 | 0.508 ± 0.013 | −0.3 | 0.508 | 0.508 |
| hourly | XGBoost | 5,254 | 0.500 ± 0.013 | −1.1 | 0.499 | 0.511 |
| daily | Majority baseline | 217 | 0.539 ± 0.066 | – | 0.500 | 0.701 |
| daily | Logistic Regression | 217 | 0.521 ± 0.067 | −1.8 | 0.523 | 0.527 |
| daily | Decision Tree | 217 | 0.535 ± 0.066 | −0.5 | 0.523 | 0.610 |
| daily | Random Forest | 217 | 0.535 ± 0.066 | −0.5 | 0.530 | 0.577 |
| daily | NN 1 layer | 217 | 0.530 ± 0.066 | −0.9 | 0.533 | 0.532 |
| daily | NN 3 layers | 217 | 0.516 ± 0.067 | −2.3 | 0.519 | 0.520 |
| daily | XGBoost | 217 | 0.530 ± 0.066 | −0.9 | 0.531 | 0.540 |

F1 is computed for the UP class. When the baseline predicts only UP, its F1 is
high even though it has no skill; when it predicts only DOWN, its F1 is 0.
This is why balanced accuracy and the improvement over baseline are the
fairer headline metrics.

### Interpretation

* **No model beats the baseline.** Averaged over five independent paths, the
  improvement over the majority baseline is zero or negative for every model
  at every frequency (from −2.8 to +0.0 pp). No model beats the baseline
  consistently; the best case is 3 seeds out of 5. Past returns, momentum and
  volatility carry no information about the next move in a GBM market, and
  the pipeline correctly reports that.
* **More complexity does not help.** Going from Logistic Regression to Random
  Forest to a 1-layer and then a 3-layer neural network brings no consistent
  improvement. On hourly and daily data the networks hit the 200-epoch limit
  without converging: they keep fitting noise in the training set
  (overfitting).
* **Single daily results are mostly luck.** With only 217 daily test
  observations, a model's improvement over the baseline on a single path
  ranges from −12 pp to +6 pp. The baseline's own accuracy ranges from 46%
  to 59%, depending on whether the test period happened to trend. At tick
  level the same range is only ±0.3 pp.
* **The drift is too small to show up.** In theory P(UP) is about 50.03% per
  tick, 50.1% per hour and 50.6% per day, which is smaller than the sampling
  noise.
* **Lesson for real data:** a result should be judged by its improvement over
  a simple baseline on the same test period, and by how stable that
  improvement is across samples. A single daily backtest could easily show
  55% accuracy from pure noise.

---

## Output files (`results/`)

| File | Content |
|---|---|
| `01_gbm_price_path.png` | Simulated tick path and daily closes (seed 42) |
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

## Using real market data later

The ML pipeline only needs a DataFrame with a timestamp index and a `close`
column. For real data, replace the simulation in `main.py` (`run_seed`) with:

```python
import data
ticks = data.load_ohlcv_csv("my_trades.csv")          # transaction-level data
datasets = {
    "tick":   ticks,                                   # one row per trade
    "hourly": data.load_ohlcv_csv("my_hourly.csv"),    # or aggregate the trades
    "daily":  data.load_ohlcv_csv("my_daily.csv"),
}
```

For tick data, set `open = high = low = close = trade price`. With a `price`
column, `data.ticks_to_ohlc(ticks, "1h")` builds the candles. Drop the
GBM-specific checks and use a single "seed" (there is only one real history).
Real tick data often contains unchanged prices; they are removed and counted
automatically. Features, models and evaluation stay the same.
