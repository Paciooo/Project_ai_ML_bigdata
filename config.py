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
SEED = 42               # random seed (simulation and models)

DAYS_PER_YEAR = 365     # synthetic 24/7 market
HOURS_PER_DAY = 24
TICKS_PER_HOUR = 12     # one synthetic price observation every 5 minutes

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
TRAIN_FRAC = 0.60
VAL_FRAC = 0.20                 # the remaining 20% is the test set

# ---------------------------------------------------------------------------
# Models (same hyperparameters for every frequency)
# ---------------------------------------------------------------------------
USE_XGBOOST = True              # skipped automatically if xgboost is missing

LOGREG_PARAMS = dict(C=1.0, max_iter=1000)
TREE_PARAMS = dict(max_depth=5, min_samples_leaf=50)
FOREST_PARAMS = dict(n_estimators=200, max_depth=6, min_samples_leaf=50, n_jobs=-1)
XGB_PARAMS = dict(n_estimators=200, max_depth=3, learning_rate=0.05,
                  subsample=0.8, n_jobs=-1)

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
RESULTS_DIR = "results"
