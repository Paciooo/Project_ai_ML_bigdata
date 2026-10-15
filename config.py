"""
Central configuration for the experiment.

Change the values here to re-run the experiment with different simulation
settings, features or model hyperparameters. Nothing else needs editing.
"""

# ---------------------------------------------------------------------------
# Synthetic GBM simulation
# ---------------------------------------------------------------------------
S0 = 100.0              # initial price
MU = 0.08               # annual drift (8%)
SIGMA = 0.20            # annual volatility (20%)
YEARS = 3               # length of the simulation in years
SEEDS = [42, 43, 44, 45, 46]  # one simulated path per seed; the first one
                              # also produces the detailed figures

DAYS_PER_YEAR = 365     # synthetic 24/7 market
HOURS_PER_DAY = 24
TICKS_PER_HOUR = 20     # AVERAGE trades per hour; trade times are random
                        # (Poisson), so ticks arrive irregularly.
                        # Higher = more ticks but slower (60 -> ~4 min per seed)

START_DATE = "2023-01-01"  # arbitrary calendar anchor for the timestamps

# ---------------------------------------------------------------------------
# Features (identical for every frequency)
# ---------------------------------------------------------------------------
RETURN_LAGS = [1, 2, 3, 5]      # lagged log returns
MOMENTUM_WINDOWS = [5, 10]      # cumulative log return over the last n periods
VOLATILITY_WINDOWS = [5, 10]    # rolling std of log returns over n periods

# ---------------------------------------------------------------------------
# Chronological train / validation / test split
# ---------------------------------------------------------------------------
# Fractions of CALENDAR DAYS: the split dates are the same for every frequency
TRAIN_FRAC = 0.60
VAL_FRAC = 0.20                 # the remaining 20% is the test set
SPLIT_GAP = 1                   # observations dropped between train/val and
                                # val/test: the label at t uses the price at t+1

# ---------------------------------------------------------------------------
# Models (same hyperparameters for every frequency)
# ---------------------------------------------------------------------------
USE_XGBOOST = True              # skipped automatically if xgboost is missing

LOGREG_PARAMS = dict(C=1.0, max_iter=1000)
TREE_PARAMS = dict(max_depth=5, min_samples_leaf=50)
FOREST_PARAMS = dict(n_estimators=200, max_depth=6, min_samples_leaf=50, n_jobs=-1)
XGB_PARAMS = dict(n_estimators=200, max_depth=3, learning_rate=0.05,
                  subsample=0.8, n_jobs=-1)

# Small neural networks (multi-layer perceptrons). Both share the same training
# settings; only the architecture differs. No early stopping, because it would
# carve a *shuffled* validation set out of the training data.
MLP_COMMON = dict(activation="relu", alpha=1e-3, learning_rate_init=1e-3,
                  batch_size=256, max_iter=200)
MLP_SHALLOW_LAYERS = (16,)          # 1 hidden layer with 16 neurons
MLP_DEEP_LAYERS = (64, 32, 16)      # 3 hidden layers

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
RESULTS_DIR = "results"

# Models highlighted in the multi-seed chart (simple -> complex)
FOCUS_MODELS = ["Logistic Regression", "Random Forest", "NN 1 layer", "NN 3 layers"]
