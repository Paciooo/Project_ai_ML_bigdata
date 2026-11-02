"""
Bars with explicit timing: source conventions, CME sessions and time frames.

Every bar produced here carries
  bar_start, bar_end   interval [start, end) in UTC (tz-aware)
  available_at         first moment the bar's close is known (UTC)
  session_id           CME trading day (date of the session's close, Chicago)
  is_partial           the bar does not cover its full scheduled interval
                       (last 4-hour bin of a session, early-close holidays)
  suspect              the bar contains a quarantined minute (see quality.py)

Source conventions (stated, not inferred by resampling defaults):
  * TradeStation minutes (futures-data): naive Europe/Rome wall clock,
    stamped with the bar END. "10:01" covers [10:00, 10:01) Rome time.
  * IBKR trades: UTC event times (1-second resolution). A trade at exactly
    10:01:00 belongs to [10:01, 10:02): intervals are closed on the left.

CME Globex equity-index schedule (America/Chicago). Sessions open at 17:00 CT
the previous evening. The close and the daily halt changed over time; the
dates below are where the change shows in the NQ minute data (consistent with
CME's schedule changes), so they are an inferred, documented assumption:
  until 2015-09-18      close 16:15 CT, halt 15:15-15:30 CT
  2015-09-21..2021-06-25 close 16:00 CT, halt 15:15-15:30 CT
  from 2021-06-28        close 16:00 CT, no halt
Bars outside the scheduled hours are flagged (`in_schedule`) and dropped by
the caller with a count. Daylight saving time is handled by doing all session
arithmetic in America/Chicago.
"""

import numpy as np
import pandas as pd

CHICAGO = "America/Chicago"
FREQUENCIES = {"1min": "1min", "5min": "5min", "15min": "15min", "1h": "1h", "4h": "4h",
               "session": None}
OPEN_CT = pd.Timedelta(hours=17)                    # session opens 17:00 CT the day before
SCHEDULE = [  # (first session date, close time CT, halt start, halt end)
    (pd.Timestamp("1900-01-01"), pd.Timedelta(hours=16, minutes=15), "15:15", "15:30"),
    (pd.Timestamp("2015-09-21"), pd.Timedelta(hours=16), "15:15", "15:30"),
    (pd.Timestamp("2021-06-28"), pd.Timedelta(hours=16), None, None),
]
EARLY_CLOSE_TOLERANCE = pd.Timedelta(minutes=15)    # session ending earlier = early close


# ---------------------------------------------------------------------------
# Sources -> canonical 1-minute bars
# ---------------------------------------------------------------------------

def _frame(start, end, o, h, l, c, v):
    return pd.DataFrame({"bar_start": start, "bar_end": end, "available_at": end,
                         "open": o, "high": h, "low": l, "close": c, "volume": v})


def from_tradestation_minutes(raw, tz="Europe/Rome"):
    """futures-data minute bars (columns ts, open, high, low, close, volume).

    Returns (bars, report). Wall-clock times that do not exist or are
    ambiguous at a DST change are dropped and counted (CME is closed then).
    """
    end = pd.DatetimeIndex(raw["ts"]).tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    ok = ~end.isna()
    end = end[ok].tz_convert("UTC")
    r = raw[ok]
    bars = _frame(end - pd.Timedelta(minutes=1), end, r["open"].to_numpy(), r["high"].to_numpy(),
                  r["low"].to_numpy(), r["close"].to_numpy(), r["volume"].to_numpy())
    return bars.reset_index(drop=True), {"dst_invalid_timestamps_dropped": int((~ok).sum())}


def from_ib_ticks(ticks):
    """1-minute bars from trades (DataFrame with UTC `ts`, `price`, `size`).

    Left-closed minutes on the event time; trade order inside a minute is kept
    (open = first trade, close = last trade in file order).
    """
    ts = pd.DatetimeIndex(ticks["ts"])
    ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
    start = ts.floor("1min")
    g = pd.DataFrame({"start": start, "p": ticks["price"].to_numpy(),
                      "s": ticks["size"].to_numpy()}).groupby("start", sort=True)
    agg = g.agg(open=("p", "first"), high=("p", "max"), low=("p", "min"), close=("p", "last"),
                volume=("s", "sum"))
    st = pd.DatetimeIndex(agg.index)
    return _frame(st, st + pd.Timedelta(minutes=1), agg["open"].to_numpy(), agg["high"].to_numpy(),
                  agg["low"].to_numpy(), agg["close"].to_numpy(), agg["volume"].to_numpy()
                  ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

def session_calendar(bar_start):
    """Session id, scheduled open/close (UTC) and in-schedule flag per bar start."""
    ct = pd.DatetimeIndex(bar_start).tz_convert(CHICAGO)
    local_day = ct.tz_localize(None).normalize()
    # bars from 17:00 CT belong to the next day's session
    after_open = (ct.hour >= 17)
    sid = local_day + pd.to_timedelta(after_open.astype(int), unit="D")
    sid = pd.DatetimeIndex(sid)
    close_td = pd.Series(pd.Timedelta(0), index=range(len(sid)))
    halt0 = pd.Series(pd.NaT, index=range(len(sid)), dtype="timedelta64[ns]")
    halt1 = halt0.copy()
    for first, close, h0, h1 in SCHEDULE:
        m = np.asarray(sid >= first)
        close_td[m] = close
        halt0[m] = pd.Timedelta(h0 + ":00") if h0 else pd.NaT
        halt1[m] = pd.Timedelta(h1 + ":00") if h1 else pd.NaT
    open_local = (sid - pd.Timedelta(days=1)) + OPEN_CT
    close_local = sid + pd.to_timedelta(close_td.to_numpy())
    tod = ct.tz_localize(None) - local_day                     # time of day of bar start
    in_halt = (tod >= halt0.to_numpy()) & (tod < halt1.to_numpy())
    in_sched = ((ct.tz_localize(None) >= open_local) & (ct.tz_localize(None) < close_local)
                & ~np.asarray(in_halt) & (sid.dayofweek < 5))
    loc = lambda x: pd.DatetimeIndex(x).tz_localize(CHICAGO, ambiguous="NaT",
                                                    nonexistent="shift_forward").tz_convert("UTC")
    return pd.DataFrame({"session_id": sid, "session_open": loc(open_local),
                         "session_close": loc(close_local), "in_schedule": np.asarray(in_sched)})


def add_sessions(bars):
    """Attach session columns; returns (bars in schedule, report)."""
    cal = session_calendar(bars["bar_start"])
    out = pd.concat([bars.reset_index(drop=True), cal], axis=1)
    rep = {"bars_outside_schedule_dropped": int((~out["in_schedule"]).sum())}
    out = out[out["in_schedule"]].drop(columns="in_schedule").reset_index(drop=True)
    return out, rep


# ---------------------------------------------------------------------------
# Time frames
# ---------------------------------------------------------------------------

def aggregate(minutes, freq):
    """Session-anchored candles from 1-minute bars (with session columns).

    Bins start at the session open (17:00 CT) and never cross a session.
    `available_at` is the bin's scheduled end, not the last minute seen:
    an hourly bar [10:00, 11:00) is only known at 11:00. A bin whose
    scheduled interval is shorter than `freq` (the last 4-hour bin, the
    16:00-16:15 bin before 2015) or that is cut by an early close is flagged
    `is_partial`. freq="session": one bar per CME session (last-trade close,
    not the official settlement).
    """
    if freq not in FREQUENCIES:
        raise ValueError(f"unknown frequency {freq!r}; choose from {list(FREQUENCIES)}")
    m = minutes
    actual_end = m.groupby("session_id")["bar_end"].transform("max")
    early = actual_end < m["session_close"] - EARLY_CLOSE_TOLERANCE
    if freq == "session":
        start = m["session_open"]
        sched_end = m["session_close"]
    else:
        f = pd.Timedelta(FREQUENCIES[freq])
        k = (m["bar_start"] - m["session_open"]) // f
        start = m["session_open"] + k * f
        sched_end = np.minimum(start + f, m["session_close"])
    g = pd.DataFrame({"session_id": m["session_id"], "bin": start, "sched_end": sched_end,
                      "open": m["open"], "high": m["high"], "low": m["low"], "close": m["close"],
                      "volume": m["volume"], "suspect": m.get("suspect", False),
                      "early": early, "actual_end": actual_end, "n": 1})
    out = g.groupby(["session_id", "bin"], sort=True).agg(
        sched_end=("sched_end", "first"), open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
        suspect=("suspect", "any"), early=("early", "first"), actual_end=("actual_end", "first"),
        n_minutes=("n", "sum")).reset_index()
    full = pd.Timedelta(FREQUENCIES[freq]) if freq != "session" else None
    short_bin = (out["sched_end"] - out["bin"] < full) if full is not None else False
    cut = out["early"] & (out["sched_end"] > out["actual_end"])
    end = out["sched_end"].where(~cut, out["actual_end"])
    res = pd.DataFrame({
        "bar_start": out["bin"], "bar_end": end, "available_at": end,
        "open": out["open"], "high": out["high"], "low": out["low"], "close": out["close"],
        "volume": out["volume"], "n_minutes": out["n_minutes"], "session_id": out["session_id"],
        "is_partial": np.asarray(short_bin | cut), "suspect": out["suspect"].to_numpy()})
    if freq == "1min":
        res["is_partial"] = False
    return res


def next_label_eligible(bars, freq, max_session_gap_days=5):
    """True where the NEXT row is the next scheduled interval.

    Intraday: the next bar must start exactly where this one ends, in the same
    session (an outage, a halt, a quiet minute or the overnight break makes
    the label ineligible instead of silently stretching the horizon).
    Session: the next session must follow within `max_session_gap_days`
    calendar days (weekends and holidays are fine, missing weeks are not).
    Partial or suspect bars, on either side, are never eligible.
    """
    nxt_start = bars["bar_start"].shift(-1)
    if freq == "session":
        gap = bars["session_id"].shift(-1) - bars["session_id"]
        ok = gap.notna() & (gap <= pd.Timedelta(days=max_session_gap_days))
    else:
        ok = (nxt_start == bars["bar_end"]) & (bars["session_id"].shift(-1) == bars["session_id"])
    bad = bars["is_partial"].astype(bool) | bars["suspect"].astype(bool)
    return (ok & ~bad & ~bad.shift(-1, fill_value=True)).to_numpy()
