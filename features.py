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
    of the days in `daily_index` (trading days for real data), validation the
    next `val_frac`, test the rest.
    """
    validate_fractions(train_frac, val_frac)
    n_days = len(daily_index)
    if n_days < 3:
        raise ValueError(f"need at least 3 days to split, got {n_days}")
    val_start = daily_index[round(n_days * train_frac)]
    test_start = daily_index[round(n_days * (train_frac + val_frac))]
    return val_start, test_start


def calendar_split(X, y, val_start, test_start, gap=1):
    """Split into train / validation / test by calendar date (no shuffling).

    The same dates are used for tick, hourly and daily data, so every
    frequency is trained and tested on exactly the same period.
    The label at t uses the price at t+1, so the last row before a boundary
    has a label that reaches into the next period. The last `gap` rows of
    the training and validation sets are therefore dropped: every label then
    depends only on prices inside its own period.
    """
    t = X.index
    train = t < val_start
    val = (t >= val_start) & (t < test_start)
    test = t >= test_start

    if not isinstance(gap, (int, np.integer)) or gap < 0:
        raise ValueError(f"gap must be a non-negative integer, got {gap!r}")

    def drop_last(mask):
        n = int(mask.sum())
        if gap >= n:  # would leave nothing (and a negative stop would wrap around)
            raise ValueError(f"gap={gap} >= split size {n}: split too small")
        n_keep = n - gap  # not iloc[:-gap]: that is empty when gap=0
        return X[mask].iloc[:n_keep], y[mask].iloc[:n_keep]

    return {
        "train": drop_last(train),
        "val": drop_last(val),
        "test": (X[test], y[test]),
    }


# ---------------------------------------------------------------------------
# Timed datasets (bars from timeframes.py): targets, timing, splits
# ---------------------------------------------------------------------------

TARGETS = {
    # sign(close[t+1] - close[t]): the original target, renamed
    "next_close_direction": "close-to-close of the next bar",
    # sign(close[t+1] - open[t+1]): the next candle's own body, decided at t
    "next_candle_body_direction": "body of the next bar",
    # raw trades: sign(price[i+1] - price[i]); a different task, not a candle
    "next_trade_direction": "next trade vs this trade",
}


class SplitTooSmall(ValueError):
    pass


def validate_fractions(train_frac, val_frac):
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError(f"need 0 < train_frac, val_frac and train_frac + val_frac < 1; "
                         f"got {train_frac}, {val_frac}")


def make_targets(bars, target):
    """Three-class outcome in {-1, 0, +1} (0 = flat), NaN where unknown.

    Computed on the FULL bar sequence before any row is removed, so the
    horizon is always exactly one bar (removing flat rows first would turn
    "next bar" into "next bar that moved").
    """
    c = bars["close"].to_numpy(dtype=float)
    nxt_close = np.append(c[1:], np.nan)
    if target in ("next_close_direction", "next_trade_direction"):
        move = nxt_close - c
    elif target == "next_candle_body_direction":
        move = nxt_close - np.append(bars["open"].to_numpy(dtype=float)[1:], np.nan)
    else:
        raise ValueError(f"unknown target {target!r}; choose from {list(TARGETS)}")
    return np.sign(move)


def make_extra_features(bars, groups):
    """Optional feature groups for the ablation (all known at the bar's close).

    volume: log volume relative to its past 20-bar mean
    range:  (high - low) / close and (close - open) / close of the bar
    rv:     past realized volatility, sqrt of the sum of squared returns over 20 bars
    time:   minutes since the session open, as sin/cos of the session fraction
    """
    f = pd.DataFrame(index=bars.index)
    r = np.log(bars["close"]).diff()
    if "volume" in groups:
        lv = np.log1p(bars["volume"].astype(float))
        f["log_vol_rel20"] = lv - lv.rolling(20).mean()
    if "range" in groups:
        f["range_rel"] = (bars["high"] - bars["low"]) / bars["close"]
        f["body_rel"] = (bars["close"] - bars["open"]) / bars["close"]
    if "rv" in groups:
        f["rv_20"] = np.sqrt((r ** 2).rolling(20).sum())
    if "time" in groups and "session_open" in bars:
        frac = ((bars["bar_end"] - bars["session_open"]).dt.total_seconds() / (23 * 3600)).clip(0, 1)
        f["sess_sin"], f["sess_cos"] = np.sin(2 * np.pi * frac), np.cos(2 * np.pi * frac)
    return f


def build_timed_dataset(bars, freq, target, return_lags, momentum_windows, volatility_windows,
                        extra_groups=(), decision_latency="1s", label_eligible=None):
    """Features, three-class outcome and timing metadata for every bar.

    Returns (X, y3, meta, info). Rows are kept only if all features exist, the
    outcome is defined and eligible (next scheduled interval, not partial,
    not suspect) and no bar in the feature window is suspect. Flat outcomes
    (y3 == 0) are KEPT.

    meta columns: bar_start, bar_end, available_at, decision_at (available_at +
    decision_latency, for inference), label_start/label_end (the next bar's
    interval), signal/label session ids and exit_price. Execution stays
    unavailable until backtest.attach_execution supplies an observed minute
    open after the decision. Body direction is an outcome definition, not
    a claim that the decision occurs before the next candle opens.
    """
    b = bars.reset_index(drop=True)
    X = make_features(b, return_lags, momentum_windows, volatility_windows)
    if extra_groups:
        X = X.join(make_extra_features(b, extra_groups))
    y3 = pd.Series(make_targets(b, target), index=b.index)
    elig = label_eligible if label_eligible is not None else np.ones(len(b), bool)
    lookback = max([*return_lags, *[n + 1 for n in momentum_windows],
                    *[n + 1 for n in volatility_windows], 21 if extra_groups else 0])
    susp = b["suspect"].astype(bool) if "suspect" in b else pd.Series(False, index=b.index)
    window_clean = ~susp.rolling(lookback, min_periods=1).max().astype(bool)
    window_clean &= ~susp.shift(-1, fill_value=False).astype(bool)   # label bar suspect
    latency = pd.Timedelta(decision_latency)
    if pd.isna(latency) or latency < pd.Timedelta(0):
        raise ValueError("decision latency must be nonnegative")
    meta = pd.DataFrame({
        "bar_start": b["bar_start"], "bar_end": b["bar_end"], "available_at": b["available_at"],
        "decision_at": b["available_at"] + latency,
        "label_start": b["bar_start"].shift(-1), "label_end": b["bar_end"].shift(-1),
        "session_id": b["session_id"], "label_session_id": b["session_id"].shift(-1), "close": b["close"],
        "entry_price": np.nan, "exit_price": b["close"].shift(-1),
        "entry_at": pd.NaT, "exit_at": b["bar_end"].shift(-1), "execution_eligible": False})
    keep = (X.notna().all(axis=1) & y3.notna() & pd.Series(elig, index=b.index) & window_clean
            & (meta["label_end"] > meta["decision_at"]))
    info = {"frequency": freq, "target": target, "n_bars": len(b),
            "n_label_ineligible": int((~pd.Series(elig, index=b.index)).sum()),
            "n_quarantine_window": int((~window_clean).sum()),
            "n_rows": int(keep.sum()), "n_flat": int((y3[keep] == 0).sum()),
            "lookback_bars": lookback}
    idx = b.index[keep]
    return X.loc[idx], y3.loc[idx].astype(int), meta.loc[idx], info


def session_boundaries(session_ids, train_frac, val_frac):
    """(val_start, test_start) as session ids: whole sessions per split."""
    validate_fractions(train_frac, val_frac)
    s = pd.DatetimeIndex(sorted(pd.unique(session_ids)))
    if len(s) < 3:
        raise ValueError(f"need at least 3 sessions to split, got {len(s)}")
    return s[round(len(s) * train_frac)], s[round(len(s) * (train_frac + val_frac))]


def purged_split(meta, val_start, test_start, session_open_of, end=None, min_rows=30):
    """Boolean masks (train, val, test) by session, purging labels that cross.

    A row belongs to a split by its session. Rows of train (val) whose label
    ends after the first session of the next split opens are purged, so no
    training label overlaps validation (or test) time. `session_open_of` maps
    a session id to its scheduled open (UTC). Features may use earlier
    history across a boundary: that is information legitimately available.
    `end` (optional session id) excludes sessions >= end from the test set
    (e.g. a reserved confirmatory holdout).
    """
    sid = meta["session_id"]
    tr = np.array(sid < val_start)
    va = np.array((sid >= val_start) & (sid < test_start))
    te = np.array(sid >= test_start)
    if end is not None:
        te &= (sid < end).to_numpy()
    tr &= (meta["label_end"] <= session_open_of(val_start)).to_numpy()
    va &= (meta["label_end"] <= session_open_of(test_start)).to_numpy()
    for name, m in (("train", tr), ("validation", va), ("test", te)):
        if m.sum() < min_rows:
            raise SplitTooSmall(f"{name} split has {int(m.sum())} rows (< {min_rows}): "
                             f"not enough data for this split")
    return tr, va, te


def walk_forward_folds(session_ids, scheme="holdout", train_frac=0.6, val_frac=0.2,
                       n_folds=4, rolling_train_sessions=None):
    """Chronological folds as (train_first, val_start, test_start, test_end) session ids.

    holdout:   one fold, train_frac / val_frac / rest.
    expanding: the last (1 - train_frac - val_frac) of the sessions is cut
               into n_folds test blocks; each fold trains on everything
               before its validation block (validation = val_frac-sized
               block just before the test block).
    rolling:   as expanding, but training keeps only the last
               `rolling_train_sessions` sessions.
    test_end is exclusive (None = to the end).
    """
    validate_fractions(train_frac, val_frac)
    s = pd.DatetimeIndex(sorted(pd.unique(session_ids)))
    n = len(s)
    if scheme == "holdout":
        v, t = round(n * train_frac), round(n * (train_frac + val_frac))
        return [(s[0], s[v], s[t], None)]
    if scheme not in ("expanding", "rolling"):
        raise ValueError(f"unknown scheme {scheme!r}")
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    t0 = round(n * (train_frac + val_frac))
    n_val = max(1, round(n * val_frac / n_folds))
    edges = np.linspace(t0, n, n_folds + 1).round().astype(int)
    folds = []
    for a, b in zip(edges[:-1], edges[1:]):
        v = a - n_val
        first = 0 if scheme == "expanding" else max(0, v - (rolling_train_sessions or v))
        if v <= first or b <= a:
            raise ValueError("fold too small: use fewer folds")
        folds.append((s[first], s[v], s[a], s[b] if b < n else None))
    return folds


def add_context_features(meta, X, higher, prefix, lags=(1, 2, 3)):
    """Join completed higher-frequency candles onto intraday rows, causally.

    Uses a backward as-of join on time: a row decided at `decision_at` gets
    the latest `higher` bar whose `available_at` <= decision_at. An unfinished
    session (or hour) is never used: its available_at is in the future.
    `higher` must come from last-trade candles (no settlement prices, which
    are published after the close).
    """
    r = np.log(higher["close"]).diff()
    h = pd.DataFrame({"available_at": pd.DatetimeIndex(higher["available_at"]).as_unit("ns")})
    for k in lags:
        h[f"{prefix}_ret_lag{k}"] = r.shift(k - 1).to_numpy()
    h = h.dropna().sort_values("available_at")
    left = pd.DataFrame({"decision_at": pd.DatetimeIndex(meta["decision_at"]).as_unit("ns"),
                         "_i": np.arange(len(meta))})
    left = left.sort_values("decision_at")
    j = pd.merge_asof(left, h, left_on="decision_at", right_on="available_at", direction="backward")
    j = j.sort_values("_i")
    out = X.copy()
    for c in h.columns.drop("available_at"):
        out[c] = j[c].to_numpy()
    return out
