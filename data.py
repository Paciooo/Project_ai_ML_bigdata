"""
Data layer: synthetic GBM tick simulation, OHLC aggregation and a CSV loader.

Every function returns a pandas DataFrame indexed by timestamp with at least a
`close` column. The rest of the pipeline only relies on that, so the synthetic
data can later be replaced by real market data (see `load_ohlcv_csv`).
"""

import numpy as np
import pandas as pd


def simulate_gbm_ticks(s0, mu, sigma, years, seed, ticks_per_hour=20,
                       days_per_year=365, hours_per_day=24,
                       start_date="2023-01-01"):
    """Simulate ONE GBM price path observed at random trade times (ticks).

    Trade times follow a Poisson process: the waiting times between trades are
    independent exponential variables with mean 1 / ticks_per_hour hours. So,
    as in real markets, ticks arrive irregularly.

    Between two trades the price moves with the exact GBM solution over the
    actual elapsed time dt_i:

        S_i = S_{i-1} * exp[(mu - sigma^2 / 2) dt_i + sigma sqrt(dt_i) Z_i],
        Z_i ~ iid N(0, 1).

    The increments are independent, so the tick data contains no predictable
    pattern. Returns a DataFrame with one row per tick: `price`, a synthetic
    `volume` and the elapsed time `dt` (in years) since the previous tick.
    """
    rng = np.random.default_rng(seed)

    hours_per_year = days_per_year * hours_per_day
    horizon = years * hours_per_year  # total length in hours

    # Poisson trade times: cumulative sum of exponential waiting times
    n_draw = int(horizon * ticks_per_hour * 1.1) + 1000  # comfortably enough
    waits = rng.exponential(1.0 / ticks_per_hour, size=n_draw)
    times = np.concatenate([[0.0], np.cumsum(waits)])  # first tick at t=0
    times = times[times < horizon]

    # Exact GBM step over each (random) elapsed time, measured in years
    dt = np.diff(times) / hours_per_year
    z = rng.standard_normal(len(dt))
    log_increments = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    prices = s0 * np.exp(np.concatenate([[0.0], np.cumsum(log_increments)]))

    # Synthetic trade size: iid noise, independent of prices -> carries no signal
    volume = rng.poisson(lam=100, size=len(times)) + 1

    index = pd.Timestamp(start_date) + pd.to_timedelta(times, unit="h")
    return pd.DataFrame({"price": prices, "volume": volume,
                         "dt": np.concatenate([[np.nan], dt])},
                        index=pd.DatetimeIndex(index, name="timestamp"))


def ticks_to_ohlc(ticks, rule):
    """Aggregate tick prices into OHLCV candles (e.g. rule='1h' or '1D')."""
    ohlc = ticks["price"].resample(rule).ohlc()
    ohlc["volume"] = ticks["volume"].resample(rule).sum()
    ohlc["n_ticks"] = ticks["price"].resample(rule).count()
    return ohlc.dropna()


def ticks_as_bars(ticks):
    """Represent raw ticks with the same column layout as candles.

    For a single trade, open = high = low = close = price.
    """
    df = pd.DataFrame(index=ticks.index)
    for col in ["open", "high", "low", "close"]:
        df[col] = ticks["price"]
    df["volume"] = ticks["volume"]
    return df


def build_datasets(ticks):
    """Return the three frequencies, all derived from the same tick path."""
    return {
        "tick": ticks_as_bars(ticks),
        "hourly": ticks_to_ohlc(ticks, "1h"),
        "daily": ticks_to_ohlc(ticks, "1D"),
    }


def validate_aggregation(ticks, datasets, years, days_per_year=365, hours_per_day=24):
    """Sanity checks that the candles really come from the tick path.

    Raises AssertionError if something is wrong, otherwise returns a list of
    human-readable check results.
    """
    hourly, daily = datasets["hourly"], datasets["daily"]
    checks = []

    # 1) Every hour and day of the sample is present and has at least one tick
    assert len(hourly) == years * days_per_year * hours_per_day
    assert len(daily) == years * days_per_year
    assert (hourly["n_ticks"] > 0).all()
    checks.append(f"{len(ticks):,} ticks -> {len(hourly):,} hourly and {len(daily):,} daily candles, "
                  f"none empty")

    # 2) No tick is lost or double-counted in the aggregation
    assert hourly["n_ticks"].sum() == len(ticks) == daily["n_ticks"].sum()
    checks.append(f"ticks per hour: mean {hourly['n_ticks'].mean():.1f}, "
                  f"min {hourly['n_ticks'].min()}, max {hourly['n_ticks'].max()} (irregular, as in real markets)")

    # 3) Daily candles equal the aggregation of hourly candles
    h2d = hourly.resample("1D").agg({"open": "first", "high": "max",
                                     "low": "min", "close": "last"})
    for col in ["open", "high", "low", "close"]:
        assert np.allclose(h2d[col].values, daily[col].values)
    checks.append("daily OHLC == OHLC re-aggregated from hourly candles")

    # 4) Each close is the last trade of its period; high/low bound every candle
    last_trade = ticks["price"].groupby(ticks.index.floor("1D")).last()
    assert np.allclose(daily["close"].values, last_trade.values)
    for df in (hourly, daily):
        assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
        assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
    checks.append("closes are last trades of the period; high/low bound every candle")

    return checks


def gbm_moment_check(ticks, mu, sigma):
    """Check the simulated tick returns against GBM theory.

    Standardising each log return by its own elapsed time gives values that
    should be iid N(0, 1): mean ~0, std ~1, no autocorrelation.
    """
    r = np.log(ticks["price"]).diff()
    dt = ticks["dt"]
    z = ((r - (mu - 0.5 * sigma ** 2) * dt) / (sigma * np.sqrt(dt))).dropna()
    return {"z_mean": z.mean(), "z_std": z.std(),
            "lag1_autocorr": r.dropna().autocorr(lag=1)}


def load_ohlcv_csv(path, time_col="timestamp"):
    """Load real market data later on.

    Expected CSV columns: timestamp, open, high, low, close[, volume].
    For real tick data, set open = high = low = close = trade price.
    The returned DataFrame can be passed directly to the feature pipeline.
    """
    df = pd.read_csv(path, parse_dates=[time_col]).set_index(time_col).sort_index()
    df.columns = [c.lower() for c in df.columns]
    return df
