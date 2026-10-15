"""
Data layer: synthetic GBM simulation, OHLC aggregation and a CSV loader.

Every function returns a pandas DataFrame indexed by timestamp with at least a
`close` column. The rest of the pipeline only relies on that, so the synthetic
data can later be replaced by real market data (see `load_ohlcv_csv`).
"""

import numpy as np
import pandas as pd


def simulate_gbm_ticks(s0, mu, sigma, years, seed, days_per_year=365,
                       hours_per_day=24, ticks_per_hour=12,
                       start_date="2023-01-01"):
    """Simulate ONE high-frequency GBM path using the exact discrete solution

        S_{t+dt} = S_t * exp[(mu - sigma^2 / 2) dt + sigma sqrt(dt) Z_t],
        Z_t ~ iid N(0, 1).

    Returns a DataFrame with one row per tick: `price` and a synthetic `volume`.
    The first tick is S0, so every hour holds exactly `ticks_per_hour` ticks.
    """
    rng = np.random.default_rng(seed)

    ticks_per_year = days_per_year * hours_per_day * ticks_per_hour
    n_ticks = int(years * ticks_per_year)
    dt = 1.0 / ticks_per_year  # time step in years

    # Log increments of the exact GBM solution
    z = rng.standard_normal(n_ticks - 1)
    log_increments = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    log_path = np.concatenate([[0.0], np.cumsum(log_increments)])
    prices = s0 * np.exp(log_path)

    # Synthetic volume: iid noise, independent of prices -> carries no signal
    volume = rng.poisson(lam=100, size=n_ticks)

    freq = pd.Timedelta(hours=1) / ticks_per_hour
    index = pd.date_range(start=start_date, periods=n_ticks, freq=freq, name="timestamp")
    return pd.DataFrame({"price": prices, "volume": volume}, index=index)


def ticks_to_ohlc(ticks, rule):
    """Aggregate tick prices into OHLCV candles (e.g. rule='1h' or '1D')."""
    ohlc = ticks["price"].resample(rule).ohlc()
    ohlc["volume"] = ticks["volume"].resample(rule).sum()
    ohlc["n_ticks"] = ticks["price"].resample(rule).count()
    return ohlc.dropna()


def ticks_as_bars(ticks):
    """Represent raw ticks with the same column layout as candles.

    For a single observation, open = high = low = close = price.
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


def validate_aggregation(ticks, datasets, ticks_per_hour, hours_per_day):
    """Sanity checks that the candles really come from the tick path.

    Raises AssertionError if something is wrong, otherwise returns a list of
    human-readable check results.
    """
    hourly, daily = datasets["hourly"], datasets["daily"]
    checks = []

    # 1) Every candle contains the expected number of ticks
    assert (hourly["n_ticks"] == ticks_per_hour).all()
    assert (daily["n_ticks"] == ticks_per_hour * hours_per_day).all()
    checks.append(f"every hourly candle has {ticks_per_hour} ticks, "
                  f"every daily candle has {ticks_per_hour * hours_per_day}")

    # 2) Total number of observations is consistent
    assert len(hourly) * ticks_per_hour == len(ticks)
    assert len(daily) * ticks_per_hour * hours_per_day == len(ticks)
    checks.append(f"{len(ticks):,} ticks = {len(hourly):,} hours = {len(daily):,} days")

    # 3) Daily candles equal the aggregation of hourly candles
    h2d = hourly.resample("1D").agg({"open": "first", "high": "max",
                                     "low": "min", "close": "last"})
    for col in ["open", "high", "low", "close"]:
        assert np.allclose(h2d[col].values, daily[col].values)
    checks.append("daily OHLC == OHLC re-aggregated from hourly candles")

    # 4) Opens/closes are actual tick prices and high/low bound the candle
    assert np.allclose(daily["close"].values, ticks["price"].iloc[ticks_per_hour * hours_per_day - 1::ticks_per_hour * hours_per_day].values)
    assert np.allclose(hourly["open"].values, ticks["price"].iloc[::ticks_per_hour].values)
    for df in (hourly, daily):
        assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
        assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
    checks.append("opens/closes match tick prices; high/low bound every candle")

    return checks


def gbm_moment_check(ticks, mu, sigma, days_per_year, hours_per_day, ticks_per_hour):
    """Compare simulated tick log-return moments with GBM theory."""
    dt = 1.0 / (days_per_year * hours_per_day * ticks_per_hour)
    r = np.log(ticks["price"]).diff().dropna()
    return {
        "mean_sim": r.mean(), "mean_theory": (mu - 0.5 * sigma ** 2) * dt,
        "std_sim": r.std(), "std_theory": sigma * np.sqrt(dt),
        "lag1_autocorr": r.autocorr(lag=1),
    }


def load_ohlcv_csv(path, time_col="timestamp"):
    """Load real market data later on.

    Expected CSV columns: timestamp, open, high, low, close[, volume].
    The returned DataFrame can be passed directly to the feature pipeline.
    """
    df = pd.read_csv(path, parse_dates=[time_col]).set_index(time_col).sort_index()
    df.columns = [c.lower() for c in df.columns]
    return df
