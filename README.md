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

To change the simulation (drift, volatility, length, seeds, trade rate), the
features, the split or the model hyperparameters, edit `config.py`. To run
faster while experimenting, use a single seed: `SEEDS = [42]`.

---

## Project structure

| File | Purpose |
|---|---|
| `config.py` | All parameters (GBM, ticks, seeds, features, split, model hyperparameters) |
| `data.py` | GBM tick simulation, OHLC aggregation, aggregation checks, CSV loader for real data |
| `features.py` | Features, UP/DOWN target, chronological 60/20/20 split with gap |
| `models.py` | Majority baseline, Logistic Regression, Decision Tree, Random Forest, two small neural networks, XGBoost (optional) |
| `evaluation.py` | Accuracy, balanced accuracy, F1, confusion matrix |
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

### 4. Training and testing

* **Chronological split** for each frequency: 60% train / 20% validation /
  20% test, with no shuffling.
* **One-observation gap** between train/validation and validation/test
  (`SPLIT_GAP = 1`). The label at time $t$ uses the price at $t+1$. Without
  the gap, the last training label would depend on the first price of the
  validation period.
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

### Models highlighted in the presentation

All seven models stay in the code. The presentation focuses on four, which go
from simple to complex: **Logistic Regression → Random Forest → NN with 1
hidden layer → NN with 3 hidden layers**, compared with the majority baseline.

---

## Results

### Across 5 seeds (main result)

Test accuracy over five independent GBM paths:

| Frequency | Model | Mean accuracy | Std across seeds | Min – max | Mean balanced acc. |
|---|---|---|---|---|---|
| tick | Majority baseline | 0.501 | 0.002 | 0.498 – 0.503 | 0.500 |
| tick | Logistic Regression | 0.499 | 0.002 | 0.496 – 0.501 | 0.499 |
| tick | Random Forest | 0.500 | 0.002 | 0.497 – 0.502 | 0.500 |
| tick | NN 1 layer | 0.499 | 0.001 | 0.499 – 0.500 | 0.500 |
| tick | NN 3 layers | 0.501 | 0.001 | 0.500 – 0.503 | 0.501 |
| hourly | Majority baseline | 0.506 | 0.005 | 0.499 – 0.511 | 0.500 |
| hourly | Logistic Regression | 0.500 | 0.007 | 0.490 – 0.506 | 0.499 |
| hourly | Random Forest | 0.495 | 0.009 | 0.488 – 0.510 | 0.494 |
| hourly | NN 1 layer | 0.497 | 0.006 | 0.489 – 0.506 | 0.497 |
| hourly | NN 3 layers | 0.504 | 0.005 | 0.498 – 0.509 | 0.503 |
| daily | Majority baseline | 0.514 | 0.055 | 0.454 – 0.593 | 0.500 |
| daily | Logistic Regression | 0.502 | 0.039 | 0.454 – 0.556 | 0.500 |
| daily | Random Forest | 0.486 | 0.043 | 0.440 – 0.532 | 0.489 |
| daily | NN 1 layer | 0.527 | 0.031 | 0.491 – 0.569 | 0.525 |
| daily | NN 3 layers | 0.518 | 0.017 | 0.505 – 0.546 | 0.521 |

Decision Tree and XGBoost behave the same way; see `results/multi_seed_summary.csv`.

### Single path in detail (seed 42)

Observations per split: tick ≈ 316k / 105k / 105k; hourly 15,761 / 5,253 / 5,253;
daily 650 / 216 / 216. Share of UP moves in each split: `results/dataset_summary.csv`.

± is the 95% binomial confidence half-width of accuracy, i.e. how much the
accuracy can vary by chance alone.

| Frequency | Model | Test obs. | Accuracy | Balanced acc. | F1 |
|---|---|---|---|---|---|
| tick | Majority baseline | 105,305 | 0.501 ± 0.003 | 0.500 | 0.667 |
| tick | Logistic Regression | 105,305 | 0.499 ± 0.003 | 0.499 | 0.472 |
| tick | Decision Tree | 105,305 | 0.501 ± 0.003 | 0.501 | 0.199 |
| tick | Random Forest | 105,305 | 0.500 ± 0.003 | 0.500 | 0.431 |
| tick | NN 1 layer | 105,305 | 0.499 ± 0.003 | 0.500 | 0.191 |
| tick | NN 3 layers | 105,305 | 0.501 ± 0.003 | 0.501 | 0.478 |
| tick | XGBoost | 105,305 | 0.498 ± 0.003 | 0.498 | 0.469 |
| hourly | Majority baseline | 5,253 | 0.511 ± 0.013 | 0.500 | 0.677 |
| hourly | Logistic Regression | 5,253 | 0.499 ± 0.013 | 0.497 | 0.543 |
| hourly | Decision Tree | 5,253 | 0.489 ± 0.013 | 0.491 | 0.437 |
| hourly | Random Forest | 5,253 | 0.496 ± 0.013 | 0.494 | 0.535 |
| hourly | NN 1 layer | 5,253 | 0.493 ± 0.013 | 0.493 | 0.508 |
| hourly | NN 3 layers | 5,253 | 0.509 ± 0.013 | 0.507 | 0.538 |
| hourly | XGBoost | 5,253 | 0.505 ± 0.013 | 0.505 | 0.515 |
| daily | Majority baseline | 216 | 0.542 ± 0.066 | 0.500 | 0.703 |
| daily | Logistic Regression | 216 | 0.518 ± 0.067 | 0.521 | 0.527 |
| daily | Decision Tree | 216 | 0.532 ± 0.067 | 0.519 | 0.610 |
| daily | Random Forest | 216 | 0.532 ± 0.067 | 0.529 | 0.570 |
| daily | NN 1 layer | 216 | 0.532 ± 0.067 | 0.537 | 0.526 |
| daily | NN 3 layers | 216 | 0.546 ± 0.066 | 0.545 | 0.574 |
| daily | XGBoost | 216 | 0.509 ± 0.067 | 0.514 | 0.505 |

F1 is computed for the UP class. When the baseline predicts only UP, its F1 is
high even though it has no skill; when it predicts only DOWN, its F1 is 0.
This is why balanced accuracy is the fairer headline metric.

### Interpretation

* **No model finds a pattern that isn't there.** Averaged over five
  independent paths, every model is at about 50% at tick and hourly
  frequency, with balanced accuracy at 0.49–0.50. Past returns, momentum and
  volatility carry no information about the next move in a GBM market, and
  the pipeline correctly reports that.
* **More complexity does not help.** Going from Logistic Regression to Random
  Forest to a 1-layer and then a 3-layer neural network brings no consistent
  improvement. On hourly and daily data the networks hit the 200-epoch limit
  without converging: they keep fitting noise in the training set
  (overfitting).
* **Single daily results are mostly luck.** With only 216 daily test
  observations, the same model on different paths ranges from about 44% to
  57%. The majority baseline alone ranges from 45% to 59%, depending on
  whether the test period happened to trend. The daily neural networks'
  slightly higher mean (0.52–0.53) sits well inside this spread, comes from
  just five paths and is one of many comparisons, so it is not evidence of
  skill.
* **The drift is too small to show up.** In theory P(UP) is about 50.03% per
  tick, 50.1% per hour and 50.6% per day, which is smaller than the sampling
  noise.
* **Lesson for real data:** a reported accuracy is only meaningful next to
  its chance band and its variability across samples. Here, at the daily
  frequency, a single backtest could easily show 55% from pure noise.

---

## Output files (`results/`)

| File | Content |
|---|---|
| `01_gbm_price_path.png` | Simulated tick path and daily closes (seed 42) |
| `02_candlesticks.png` | Sample daily and hourly candlestick charts |
| `03_accuracy_comparison.png` | Test accuracy by model and frequency, seed 42, with 95% CI |
| `04_results_table.png` | Results table as an image (seed 42, for slides) |
| `05_rf_feature_importance.png` | Random Forest feature importance per frequency (seed 42) |
| `06_confusion_matrices.png` | Test-set confusion matrices (seed 42) |
| `07_multi_seed_accuracy.png` | **Focus models across 5 seeds**: mean ± std, individual seeds, baseline range |
| `results_table.csv` | Full metrics for seed 42 (test, validation, confusion-matrix counts) |
| `results_all_seeds.csv` | The same for every seed |
| `multi_seed_summary.csv` | Mean / std / min / max across seeds |
| `dataset_summary.csv` | Observations, dropped rows and class distribution per split, per seed |
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
