"""Timing, sessions, targets, quarantine and splits: leakage/alignment checks."""

import numpy as np
import pandas as pd
import pytest

import data
import features
import quality
import timeframes as tf

LAGS, MOM, VOL = [1, 2, 3, 5], [5, 10], [5, 10]


def minutes(n_sessions=6, bounce=0.0, seed=1):
    m, _ = tf.add_sessions(data.fixture_minutes(n_sessions, seed=seed, bounce=bounce))
    m["suspect"] = False
    return m


# ---------------- source conventions ----------------

def test_tradestation_end_stamped_rome():
    raw = pd.DataFrame({"ts": pd.to_datetime(["2024-07-01 16:31", "2024-01-02 16:31"]),
                        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1})
    b, rep = tf.from_tradestation_minutes(raw)
    # 16:31 Rome summer = 14:31 UTC, end-stamped -> [14:30, 14:31)
    assert b["bar_start"][0] == pd.Timestamp("2024-07-01 14:30", tz="UTC")
    assert b["bar_end"][0] == pd.Timestamp("2024-07-01 14:31", tz="UTC")
    assert b["bar_start"][1] == pd.Timestamp("2024-01-02 15:30", tz="UTC")   # winter: UTC+1
    assert b["available_at"].equals(b["bar_end"])


def test_tradestation_dst_invalid_dropped():
    raw = pd.DataFrame({"ts": pd.to_datetime(["2024-03-31 02:30", "2024-10-27 02:30"]),
                        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1})
    b, rep = tf.from_tradestation_minutes(raw)
    assert len(b) == 0 and rep["dst_invalid_timestamps_dropped"] == 2


def test_ib_tick_minute_boundary_left_closed():
    t = pd.to_datetime(["2026-10-13 14:00:59", "2026-10-13 14:01:00", "2026-10-13 14:01:00"], utc=True)
    ticks = pd.DataFrame({"ts": t, "price": [1.0, 2.0, 3.0], "size": [1, 1, 1]})
    b = tf.from_ib_ticks(ticks)
    assert list(b["close"]) == [1.0, 3.0] and list(b["open"]) == [1.0, 2.0]
    assert b["bar_start"][1] == pd.Timestamp("2026-10-13 14:01", tz="UTC")


# ---------------- sessions and time frames ----------------

def test_session_id_and_dst():
    # Sunday 17:00 CT (both DST states) belongs to Monday's session
    st = pd.DatetimeIndex(["2024-03-10 22:00", "2024-11-03 23:00", "2024-07-01 20:59"], tz="UTC")
    cal = tf.session_calendar(st)
    assert list(cal["session_id"].dt.date.astype(str)) == ["2024-03-11", "2024-11-04", "2024-07-01"]
    assert cal["in_schedule"].all()
    # 16:30 CT is in the daily break; Saturday is closed
    off = tf.session_calendar(pd.DatetimeIndex(["2024-07-01 21:30", "2024-07-06 12:00"], tz="UTC"))
    assert not off["in_schedule"].any()


def test_old_schedule_halt_and_close():
    st = pd.DatetimeIndex(["2012-05-01 20:20", "2012-05-01 21:10", "2024-05-01 20:20"], tz="UTC")
    cal = tf.session_calendar(st)   # 15:20 CT halt (2012), 16:10 CT open in 2012, 15:20 CT open in 2024
    assert list(cal["in_schedule"]) == [False, True, True]


def test_hourly_bar_available_at_end_not_label():
    m = minutes(2)
    h = tf.aggregate(m, "1h")
    assert ((h["available_at"] - h["bar_start"]) <= pd.Timedelta("1h")).all()
    row = h.iloc[5]
    inside = m[(m["bar_start"] >= row["bar_start"]) & (m["bar_start"] < row["bar_end"])]
    assert row["close"] == inside["close"].iloc[-1]
    assert row["available_at"] == row["bar_start"] + pd.Timedelta("1h")
    assert inside["bar_end"].max() <= row["available_at"]


def test_missing_minute_inside_a_full_candle_is_counted_but_not_partial():
    # Documents current behaviour (tools/sample_checks.py measures its incidence): a 5-minute
    # candle with 4 of its 5 one-minute bars keeps its scheduled interval, is NOT flagged
    # partial and stays label-eligible; only n_minutes records the missing minute.
    m = minutes(2)
    full = tf.aggregate(m, "5min")
    first = full.iloc[3]
    inside = m[(m["bar_start"] >= first["bar_start"]) & (m["bar_start"] < first["bar_end"])]
    m2 = m.drop(index=inside.index[2]).reset_index(drop=True)          # remove the middle minute
    b = tf.aggregate(m2, "5min")
    row = b[b["bar_start"] == first["bar_start"]].iloc[0]
    assert first["n_minutes"] == 5 and row["n_minutes"] == 4
    assert not row["is_partial"] and row["bar_end"] - row["bar_start"] == pd.Timedelta("5min")
    ok = tf.next_label_eligible(b, "5min")
    assert ok[b.index[b["bar_start"] == first["bar_start"]][0] - 1]    # the previous candle may still use it as label


def test_4h_last_bin_partial_and_not_eligible():
    h4 = tf.aggregate(minutes(3), "4h")
    last = h4.groupby("session_id").tail(1)
    assert last["is_partial"].all()                          # 13:00-16:00 CT is 3 hours
    assert (last["bar_end"] - last["bar_start"] == pd.Timedelta("3h")).all()
    assert not h4.groupby("session_id").head(5)["is_partial"].any()
    ok = tf.next_label_eligible(h4, "4h")
    assert not ok[h4["is_partial"].to_numpy()].any()


def test_labels_never_cross_session_or_gap():
    m = minutes(3)
    m = m.drop(index=m.index[100:103]).reset_index(drop=True)        # an outage
    b = tf.aggregate(m, "1min")
    ok = tf.next_label_eligible(b, "1min")
    assert not ok[99]                                                # bar before the gap
    last_of_session = b.groupby("session_id").tail(1).index
    assert not ok[last_of_session].any()


# ---------------- targets and features ----------------

def test_target_alignment_and_flats_kept():
    b = tf.aggregate(minutes(2), "1min")
    b.loc[50, "close"] = b.loc[49, "close"]                          # one flat outcome
    y = features.make_targets(b, "next_close_direction")
    c = b["close"].to_numpy()
    assert np.array_equal(y[:-1], np.sign(c[1:] - c[:-1])) and np.isnan(y[-1])
    assert y[49] == 0
    X, y3, meta, info = features.build_timed_dataset(b, "1min", "next_close_direction", LAGS, MOM, VOL,
                                                     label_eligible=tf.next_label_eligible(b, "1min"))
    assert info["n_flat"] >= 1 and 0 in set(y3)
    body = features.make_targets(b, "next_candle_body_direction")
    assert body[10] == np.sign(b["close"][11] - b["open"][11])


def test_features_prefix_invariant_no_lookahead():
    b = tf.aggregate(minutes(2), "1min")
    X1 = features.make_features(b, LAGS, MOM, VOL)
    b2 = b.copy()
    b2.loc[500:, "close"] *= 1.05                                    # change the future
    X2 = features.make_features(b2, LAGS, MOM, VOL)
    pd.testing.assert_frame_equal(X1.iloc[:500], X2.iloc[:500])


def test_timing_metadata_order():
    b = tf.aggregate(minutes(3), "15min")
    X, y3, meta, _ = features.build_timed_dataset(b, "15min", "next_close_direction", LAGS, MOM, VOL,
                                                  label_eligible=tf.next_label_eligible(b, "15min"))
    assert (meta["available_at"] >= meta["bar_end"]).all()
    assert (meta["decision_at"] > meta["available_at"]).all()
    assert (meta["label_start"] >= meta["bar_end"]).all()
    assert (meta["label_end"] > meta["decision_at"]).all()


def test_causal_quarantine_prefix_invariant():
    b = tf.aggregate(minutes(2), "1min")
    b.loc[300, "close"] *= 1.05                                      # bad print
    f1 = quality.flag_suspect(b, 0.02)
    b2 = b.iloc[:301].copy()                                         # the future not yet known
    assert f1[300] and quality.flag_suspect(b2, 0.02)[300]
    assert np.array_equal(f1[:301], quality.flag_suspect(b2, 0.02))
    b["suspect"] = f1
    X, y3, meta, info = features.build_timed_dataset(b, "1min", "next_close_direction", LAGS, MOM, VOL)
    # 299 (label is the bad print) and every row whose 11-bar window holds bar 300
    # or the flagged rebound bar 301 are excluded; rows outside stay
    assert not set(range(299, 312)) & set(X.index)
    assert {298, 312} <= set(X.index)
    assert info["n_quarantine_window"] > 0


def test_validate_bars_rejects_bad_ohlc():
    b = tf.aggregate(minutes(1), "1min")
    b.loc[3, "high"] = b.loc[3, "low"] - 1
    with pytest.raises(quality.DataValidationError):
        quality.validate_bars(b)


def test_identical_ticks_are_kept():
    t = pd.to_datetime(["2026-10-13 14:00:01"] * 3, utc=True)
    rep = quality.validate_ticks(pd.DataFrame({"ts": t, "price": 1.0, "size": 1}))
    assert rep["n_identical_records"] == 2


# ---------------- splits ----------------

def test_legacy_split_rejects_bad_gap():
    X = pd.DataFrame({"a": range(10)}, index=pd.date_range("2024-01-01", periods=10, freq="D"))
    y = pd.Series(range(10), index=X.index)
    with pytest.raises(ValueError):
        features.calendar_split(X, y, X.index[2], X.index[3], gap=5)
    with pytest.raises(ValueError):
        features.calendar_split(X, y, X.index[2], X.index[5], gap=-1)
    with pytest.raises(ValueError):
        features.calendar_boundaries(X.index, 0.9, 0.2)


def test_purged_split_labels_do_not_cross():
    m = minutes(10)
    s = tf.aggregate(m, "session")
    X, y3, meta, _ = features.build_timed_dataset(s.assign(session_open=s["bar_start"]), "session",
                                                  "next_close_direction", [1], [2], [2],
                                                  label_eligible=tf.next_label_eligible(s, "session"))
    opens = m.groupby("session_id")["session_open"].first()
    v, t = features.session_boundaries(m["session_id"], 0.5, 0.2)
    tr, va, te = features.purged_split(meta, v, t, lambda x: opens.loc[x], min_rows=1)
    assert (meta["label_end"][tr] <= opens.loc[v]).all()
    assert (meta["label_end"][va] <= opens.loc[t]).all()
    assert not (tr & va).any() and not (va & te).any()


def test_purged_split_too_small_raises():
    m = minutes(4)
    b = tf.aggregate(m, "session")
    X, y3, meta, _ = features.build_timed_dataset(b, "session", "next_close_direction", [1], [2], [2])
    opens = m.groupby("session_id")["session_open"].first()
    v, t = features.session_boundaries(m["session_id"], 0.5, 0.25)
    with pytest.raises(features.SplitTooSmall):
        features.purged_split(meta, v, t, lambda x: opens.loc[x], min_rows=30)


def test_walk_forward_folds_are_ordered():
    s = pd.bdate_range("2020-01-01", periods=500)
    for scheme in ("expanding", "rolling"):
        folds = features.walk_forward_folds(s, scheme, 0.5, 0.1, n_folds=4, rolling_train_sessions=100)
        assert len(folds) == 4
        for first, v, t, e in folds:
            assert first < v < t and (e is None or t < e)
        assert folds[-1][3] is None
    assert features.walk_forward_folds(s, "rolling", 0.5, 0.1, 4, 100)[0][0] > s[0]


def test_context_join_never_uses_unfinished_session():
    m = minutes(5)
    sess = tf.aggregate(m, "session")
    b = tf.aggregate(m, "1h")
    X, y3, meta, _ = features.build_timed_dataset(b, "1h", "next_close_direction", [1], [2], [2])
    out = features.add_context_features(meta, X, sess, "sess", lags=(1,))
    r = np.log(sess["close"]).diff()
    for i in range(len(meta)):
        d = meta["decision_at"].iloc[i]
        done = sess[sess["available_at"] <= d]
        exp = r[done.index[-1]] if len(done) and not np.isnan(r[done.index[-1]]) else np.nan
        got = out["sess_ret_lag1"].iloc[i]
        assert (np.isnan(exp) and np.isnan(got)) or exp == pytest.approx(got)
        if len(done) and done["session_id"].iloc[-1] == meta["session_id"].iloc[i]:
            # only possible once the session is over (the row's own bar ends at the close)
            assert meta["bar_end"].iloc[i] >= done["available_at"].iloc[-1]
