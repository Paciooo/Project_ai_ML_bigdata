"""
Data validation and causal quarantine of suspicious prints.

Policy (replaces the old future-dependent spike removal):
  * a minute is SUSPECT when |log(close_t / close_{t-1})| > threshold, using
    only the bar itself and the bar before it. This is prefix-invariant: the
    flag of bar t never changes when later bars arrive, so it is available
    online at the bar's close.
  * suspect bars are NOT deleted or corrected. Every candle containing one
    is flagged; rows whose feature window or label touches a flagged candle
    are excluded from training, evaluation and the backtest, and counted.
  * excluding a row because its LABEL bar is suspect uses the future only to
    decide that the outcome is a data error (as a data vendor's later
    correction would). It never changes features or predictions.
The legacy rule (remove a jump only if the next bar reverts) is kept as
`legacy_remove_reverting_spikes` for a sensitivity comparison; it is
future-dependent and labelled as such.
No independently verified corrections exist for the futures-data export, so
none are applied.
"""

import numpy as np
import pandas as pd


class DataValidationError(ValueError):
    pass


def validate_bars(bars, years=None):
    """Hard checks on OHLCV bars; raises DataValidationError, returns a report.

    Checks: finite positive prices, high/low bound open/close, volume >= 0,
    strictly increasing unique bar starts, requested-year coverage. Also
    reports missing ranges (weekday gaps over 1 day) without failing.
    """
    px = bars[["open", "high", "low", "close"]].to_numpy(dtype=float)
    errors = []
    if not np.isfinite(px).all():
        errors.append(f"{int((~np.isfinite(px)).any(axis=1).sum())} bars with non-finite prices")
    if (px <= 0).any():
        errors.append(f"{int((px <= 0).any(axis=1).sum())} bars with non-positive prices")
    o, h, l, c = px.T
    bad_ohlc = (h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (l > h)
    if bad_ohlc.any():
        errors.append(f"{int(bad_ohlc.sum())} bars violate low <= open/close <= high")
    if (bars["volume"] < 0).any():
        errors.append("negative volume")
    st = pd.DatetimeIndex(bars["bar_start"])
    if st.has_duplicates:
        errors.append(f"{int(st.duplicated().sum())} duplicated bar timestamps")
    if not st.is_monotonic_increasing:
        errors.append("bar timestamps not increasing")
    report = {"n_bars": len(bars), "first": str(st.min()), "last": str(st.max())}
    if years is not None:
        have = set(st.year)
        missing = [y for y in range(years[0], years[1] + 1) if y not in have]
        if missing:
            errors.append(f"requested years without data: {missing}")
    gaps = pd.Series(st[1:] - st[:-1], index=st[1:])
    long = gaps[(gaps > pd.Timedelta(days=1)) & (gaps.index.dayofweek.isin([1, 2, 3, 4]))]
    report["n_missing_ranges_over_1day"] = int(len(long))
    report["missing_ranges_sample"] = [f"{t - g} -> {t}" for t, g in long.head(10).items()]
    if errors:
        raise DataValidationError("; ".join(errors))
    return report


def validate_ticks(ticks):
    """Checks on trades. Equal timestamps and identical records are legitimate
    (several trades in the same second at the same price and size) and are
    neither an error nor deduplicated."""
    errors = []
    p = ticks["price"].to_numpy(dtype=float)
    if not np.isfinite(p).all() or (p <= 0).any():
        errors.append("non-finite or non-positive trade prices")
    if (ticks["size"] < 0).any():
        errors.append("negative trade sizes")
    ts = pd.DatetimeIndex(ticks["ts"])
    if not ts.is_monotonic_increasing:
        errors.append("trade timestamps not non-decreasing")
    if errors:
        raise DataValidationError("; ".join(errors))
    return {"n_trades": len(ticks), "n_identical_records": int(ticks.duplicated().sum()),
            "n_same_second_trades": int(ts.duplicated().sum())}


def flag_suspect(bars, threshold):
    """Causal flag: |log return vs previous bar| > threshold (prefix-invariant)."""
    r = np.log(bars["close"]).diff().abs()
    return (r > threshold).fillna(False).to_numpy()


def suspect_details(bars, threshold):
    """Provenance of every flag of flag_suspect, on the SAME sequence the flag
    was computed on (before any schedule filter): the flagged bar, the log
    return that triggered it, and the predecessor bar it was measured against.
    Diagnostics only: the flags themselves are unchanged."""
    lr = np.log(bars["close"].to_numpy(float))
    flags = flag_suspect(bars, threshold)
    rows = []
    for i in np.flatnonzero(flags):
        t, tp = pd.Timestamp(bars["bar_start"].iloc[i]), pd.Timestamp(bars["bar_start"].iloc[i - 1])
        rows.append({"bar_start_utc": str(t), "close": float(bars["close"].iloc[i]),
                     "flag_log_return_pct": round(100 * float(lr[i] - lr[i - 1]), 4),
                     "prev_bar_start_utc": str(tp), "prev_close": float(bars["close"].iloc[i - 1]),
                     "minutes_since_prev": float((t - tp) / pd.Timedelta("1min")),
                     "threshold_pct": 100 * threshold})
    return rows


def legacy_remove_reverting_spikes(bars, threshold):
    """OLD rule, future-dependent: drop a bar whose jump reverts at the next bar."""
    r = np.log(bars["close"]).diff()
    rn = r.shift(-1)
    spike = (r.abs() > threshold) & (rn.abs() > threshold) & (np.sign(r) != np.sign(rn))
    return bars[~spike.to_numpy()].reset_index(drop=True), int(spike.sum())
