# Financial Market Direction Prediction Using Machine Learning

University project for *AI, Machine Learning and Big Data for Finance and Management*.

**Question:** can simple ML models predict whether the next price move is
**UP (1)** or **DOWN (0)**, and does this change across **tick**, **hourly** and
**daily** data?

This first experiment uses only **synthetic** prices from a Geometric Brownian
Motion (GBM). GBM returns are independent, so there is nothing to learn: the
experiment checks that the pipeline works and sets a benchmark that real market
data can later be compared with.

---

## How to run

```bash
pip install -r requirements.txt
python main.py
```

It takes about 1 minute (the neural networks on 189k tick rows take most of it). All tables and figures are written to `results/`.

To change the simulation (drift, volatility, length, seed), the features or the
model hyperparameters, edit `config.py`.

---

## Project structure

| File | Purpose |
|---|---|
| `config.py` | All parameters (GBM, features, split, model hyperparameters) |
| `data.py` | GBM simulation, OHLC aggregation, aggregation checks, CSV loader for real data |
| `features.py` | Features, UP/DOWN target, chronological 60/20/20 split |
| `models.py` | Majority baseline, Logistic Regression, Decision Tree, Random Forest, two small neural networks, XGBoost (optional) |
| `evaluation.py` | Accuracy, balanced accuracy, F1, confusion matrix |
| `plots.py` | All figures |
| `main.py` | Runs the full experiment |

---

## Method

### 1. Synthetic data: one GBM path, three frequencies

Prices follow $dS_t = \mu S_t\,dt + \sigma S_t\,dW_t$. They are simulated with the exact discrete solution

$$S_{t+\Delta t}=S_t\exp\!\left[\left(\mu-\tfrac12\sigma^2\right)\Delta t+\sigma\sqrt{\Delta t}\,Z_t\right],\qquad Z_t\sim N(0,1)\ \text{iid}$$

| Parameter | Value |
|---|---|
| Initial price | 100 |
| Annual drift $\mu$ | 8% |
| Annual volatility $\sigma$ | 20% |
| Length | 3 years, 24/7 market, 365 days/year |
| Ticks | 12 per hour (one every 5 minutes), $\Delta t = 1/105{,}120$ years |
| Seed | 42 |

The program simulates **one** tick path and aggregates that same path:

| Dataset | Observations | How it is built |
|---|---|---|
| Tick | 315,360 | raw simulated prices |
| Hourly | 26,280 | OHLC of 12 ticks |
| Daily | 1,095 | OHLC of 288 ticks |

`main.py` checks the aggregation every time it runs. Every candle must contain
the right number of ticks, daily OHLC must equal hourly OHLC re-aggregated, and
opens/closes must be actual tick prices. It also compares the tick-return
moments with GBM theory. Volume is iid Poisson noise, independent of prices,
so it carries no signal (and it is not used as a feature).

### 2. Target

* `1` = next close > current close
* `0` = next close < current close
* Unchanged next close → row removed (the count is reported in
  `dataset_summary.csv`; with continuous GBM prices it is 0)

### 3. Features (same for all frequencies, past/current data only)

| Feature | Definition (r = log return) |
|---|---|
| `ret_lag1`, `ret_lag2`, `ret_lag3`, `ret_lag5` | $r_t, r_{t-1}, r_{t-2}, r_{t-4}$ (lag 1 = most recent return) |
| `mom_5`, `mom_10` | $\ln(P_t / P_{t-n})$ |
| `vol_5`, `vol_10` | rolling std of $r$ over the last $n$ periods |

The first 10 rows (warm-up) and the last row (no next price) are dropped.

### 4. Training and testing

* Chronological split for each frequency: **60% train / 20% validation / 20% test**, with no shuffling.
* Every model sits in a `Pipeline(StandardScaler → model)`, so the scaling
  statistics are fitted on the training set only.
* The models are trained on the training set. The validation set is used for
  model selection: the model with the best validation balanced accuracy is
  flagged in `selected_on_val`. The test set is used only for the final
  evaluation.
* The hyperparameters are the same for all frequencies (`config.py`):
  * Logistic Regression: C=1
  * Decision Tree: depth 5, min 50 samples per leaf
  * Random Forest: 200 trees, depth 6, min 50 samples per leaf
  * XGBoost: 200 trees, depth 3, learning rate 0.05
  * Neural network, 1 hidden layer: 16 neurons
  * Neural network, 3 hidden layers: 64 → 32 → 16 neurons
  * Both networks (scikit-learn `MLPClassifier`): ReLU, Adam, L2 penalty 0.001,
    batch 256, max 200 epochs. Early stopping is off because it would draw a
    *shuffled* validation sample from the training data.
* Baseline: always predict the majority class **of the training set**.

---

## Results (seed 42)

### Observations and class distribution

| Frequency | Usable obs. | Train | Val | Test | Share UP (train) | Share UP (test) |
|---|---|---|---|---|---|---|
| Tick | 315,349 | 189,209 | 63,070 | 63,070 | 0.499 | 0.498 |
| Hourly | 26,269 | 15,761 | 5,254 | 5,254 | 0.497 | 0.493 |
| Daily | 1,084 | 650 | 217 | 217 | 0.489 | 0.512 |

### Out-of-sample (test set) performance

± is the 95% binomial confidence half-width of accuracy, i.e. how much the
accuracy can vary by chance alone.

| Frequency | Model | Test obs. | Accuracy | Balanced acc. | F1 |
|---|---|---|---|---|---|
| tick | Majority baseline | 63,070 | 0.502 ± 0.004 | 0.500 | 0.000 |
| tick | Logistic Regression | 63,070 | 0.503 ± 0.004 | 0.502 | 0.434 |
| tick | Decision Tree | 63,070 | 0.499 ± 0.004 | 0.498 | 0.471 |
| tick | Random Forest | 63,070 | 0.502 ± 0.004 | 0.502 | 0.415 |
| tick | NN 1 layer | 63,070 | 0.499 ± 0.004 | 0.500 | 0.566 |
| tick | NN 3 layers | 63,070 | 0.500 ± 0.004 | 0.500 | 0.496 |
| tick | XGBoost | 63,070 | 0.502 ± 0.004 | 0.502 | 0.462 |
| hourly | Majority baseline | 5,254 | 0.507 ± 0.013 | 0.500 | 0.000 |
| hourly | Logistic Regression | 5,254 | 0.497 ± 0.013 | 0.494 | 0.349 |
| hourly | Decision Tree | 5,254 | 0.496 ± 0.013 | 0.494 | 0.409 |
| hourly | Random Forest | 5,254 | 0.491 ± 0.013 | 0.489 | 0.383 |
| hourly | NN 1 layer | 5,254 | 0.510 ± 0.013 | 0.509 | 0.462 |
| hourly | NN 3 layers | 5,254 | 0.493 ± 0.013 | 0.493 | 0.487 |
| hourly | XGBoost | 5,254 | 0.494 ± 0.013 | 0.493 | 0.459 |
| daily | Majority baseline | 217 | 0.488 ± 0.067 | 0.500 | 0.000 |
| daily | Logistic Regression | 217 | 0.475 ± 0.066 | 0.476 | 0.457 |
| daily | Decision Tree | 217 | 0.493 ± 0.067 | 0.492 | 0.513 |
| daily | Random Forest | 217 | 0.419 ± 0.066 | 0.420 | 0.417 |
| daily | NN 1 layer | 217 | 0.479 ± 0.067 | 0.479 | 0.479 |
| daily | NN 3 layers | 217 | 0.465 ± 0.066 | 0.467 | 0.437 |
| daily | XGBoost | 217 | 0.461 ± 0.066 | 0.460 | 0.475 |

F1 is computed for the UP class. The majority baseline predicts only DOWN here,
so its F1 is 0 by construction.

### Interpretation

* **No model beats the baseline in a meaningful way at any frequency.** All
  accuracies sit at ~50%, inside or near the chance band. This is the expected
  result: GBM increments are independent, so past returns, momentum and
  volatility carry no information about the next move.
* **The drift is too small to show up.** In theory, P(UP) is about 50.04% per
  tick, 50.1% per hour and 50.6% per day. Sampling noise is larger than that
  bias: in this sample the training set even has slightly more DOWN moves, so
  the majority baseline predicts DOWN.
* **Neural networks don't help either.** Both networks score about 50% on
  tick data. The 1-layer network's 51.0% on hourly data is inside the chance
  band, and the 3-layer network does no better than the 1-layer one. Extra
  layers add capacity, but there is no pattern to capture. On hourly and
  daily data the networks hit the 200-epoch limit without converging: they
  keep fitting noise in the training set, which is overfitting.
* **Fewer observations means noisier results.** The confidence band is
  ±0.4 pp for tick data, ±1.3 pp for hourly and ±6.7 pp for daily. With only
  217 daily test points, single results move a lot by chance. The Random
  Forest's 41.9% daily accuracy is noise and overfitting, not a real
  "anti-signal", and it is one of 21 comparisons.
* **Feature importance is close to uniform** for tick and hourly data. The
  daily Random Forest leans more on the latest returns, but this is fitted
  noise from a small sample, as its poor test score shows.
* **Conclusion:** the pipeline gives honest ~50% results when there is no
  signal, so it does not leak information. This is the benchmark for real
  market data. Any real-data result should be compared with these numbers and
  confidence bands.

---

## Output files (`results/`)

| File | Content |
|---|---|
| `01_gbm_price_path.png` | Simulated GBM path (ticks + daily closes) |
| `02_candlesticks.png` | Sample daily and hourly candlestick charts |
| `03_accuracy_comparison.png` | Test accuracy by model and frequency, with 95% CI |
| `04_results_table.png` | Results table as an image (for slides) |
| `05_rf_feature_importance.png` | Random Forest feature importance per frequency |
| `06_confusion_matrices.png` | Test-set confusion matrices |
| `results_table.csv` | Full test metrics, validation metrics, confusion-matrix counts |
| `dataset_summary.csv` | Observations, dropped rows and class distribution per split |
| `rf_feature_importance.csv` | Feature importance values |

---

## Using real market data later

The ML pipeline only needs a DataFrame with a timestamp index and a `close`
column. To switch to real data, replace the simulation step in `main.py`:

```python
import data
datasets = {
    "tick":   data.load_ohlcv_csv("my_ticks.csv"),
    "hourly": data.load_ohlcv_csv("my_hourly.csv"),
    "daily":  data.load_ohlcv_csv("my_daily.csv"),
}
```

(or aggregate real ticks with `data.ticks_to_ohlc`). Drop the GBM-specific
checks and the price-path plot. Features, models and evaluation stay the same.
