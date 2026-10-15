"""
Feature engineering and target construction.

All features at time t use only prices up to and including t (no look-ahead).
The target compares the NEXT close with the current close.
"""

import numpy as np
import pandas as pd


def make_features(df, return_lags, momentum_windows, volatility_windows):
    """Build the feature matrix from a DataFrame with a `close` column.

    ret_lag1 is the most recent log return (t-1 -> t), ret_lag2 the one
    before it (t-2 -> t-1), and so on.
    """
    log_close = np.log(df["close"])
    r = log_close.diff()  # log return over the last period, known at time t

    feats = pd.DataFrame(index=df.index)
    for k in return_lags:
        feats[f"ret_lag{k}"] = r.shift(k - 1)
    for n in momentum_windows:
        feats[f"mom_{n}"] = log_close - log_close.shift(n)
    for n in volatility_windows:
        # Note for tick data: trades are irregularly spaced, so these n returns
        # cover different amounts of clock time. This is volatility per trade,
        # not per unit of time. Kept as is for simplicity; with real data,
        # consider adding the elapsed time between trades and scaling returns
        # by sqrt(elapsed time).
        feats[f"vol_{n}"] = r.rolling(n).std()
    return feats


def make_target(df):
    """1 if next close > current close, 0 if lower, NaN if unchanged / unknown.

    Unchanged prices (and the last observation, whose next price is unknown)
    are returned as NaN and removed later.
    """
    next_close = df["close"].shift(-1)
    target = pd.Series(np.nan, index=df.index)
    target[next_close > df["close"]] = 1
    target[next_close < df["close"]] = 0
    return target


def build_dataset(df, return_lags, momentum_windows, volatility_windows):
    """Return X, y with incomplete rows and unchanged-price rows removed.

    Also returns a small dict with the number of rows dropped and why.
    """
    X = make_features(df, return_lags, momentum_windows, volatility_windows)
    y = make_target(df)

    next_close = df["close"].shift(-1)
    n_unchanged = int((next_close == df["close"]).sum())

    valid = X.notna().all(axis=1) & y.notna()
    info = {"n_raw": len(df), "n_unchanged_dropped": n_unchanged,
            "n_warmup_or_last_dropped": int((~valid).sum()) - n_unchanged}
    return X[valid], y[valid].astype(int), info


def calendar_boundaries(daily_index, train_frac, val_frac):
    """Common split dates for all frequencies, on whole-day boundaries.

    Returns (val_start, test_start): training covers the first `train_frac`
    of the calendar days, validation the next `val_frac`, test the rest.
    """
    n_days = len(daily_index)
    val_start = daily_index[0] + pd.Timedelta(days=round(n_days * train_frac))
    test_start = daily_index[0] + pd.Timedelta(days=round(n_days * (train_frac + val_frac)))
    return val_start, test_start


def calendar_split(X, y, val_start, test_start, gap=1):
    """Split into train / validation / test by calendar date (no shuffling).

    The same dates are used for tick, hourly and daily data, so every
    frequency is trained and tested on exactly the same period.
    `gap` observations are dropped at the start of validation and test: the
    label of the last training row uses the next price, which falls in the
    next period; the gap keeps the label periods apart.
    """
    t = X.index
    train = t < val_start
    val = (t >= val_start) & (t < test_start)
    test = t >= test_start
    return {
        "train": (X[train], y[train]),
        "val": (X[val].iloc[gap:], y[val].iloc[gap:]),
        "test": (X[test].iloc[gap:], y[test].iloc[gap:]),
    }
