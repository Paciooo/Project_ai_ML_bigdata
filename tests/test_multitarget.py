import numpy as np
import pandas as pd

import multitarget
import multitarget_report as mr


def _minutes():
    so = pd.Timestamp("2020-01-01 17:00")
    t = so + pd.to_timedelta(np.arange(10), "min")
    o = np.array([100, 101, 102, 101, 100, 100, 102, 103, 103, 104], float)
    c = np.array([101, 102, 101, 100, 100, 102, 103, 103, 104, 105], float)
    return pd.DataFrame({"session_id": pd.Timestamp("2020-01-02"), "session_open": so, "bar_start": t,
                         "open": o, "close": c})


def test_candle_rv_uses_only_returns_inside_the_candle():
    m = _minutes()
    rv = multitarget.candle_rv(m, "5min")
    c, o = m.close.to_numpy(), m.open.to_numpy()
    first = np.log(c[0] / o[0]) ** 2 + sum(np.log(c[i] / c[i - 1]) ** 2 for i in range(1, 5))
    second = np.log(c[5] / o[5]) ** 2 + sum(np.log(c[i] / c[i - 1]) ** 2 for i in range(6, 10))
    assert np.isclose(rv.iloc[0], first) and np.isclose(rv.iloc[1], second)
    # candle 2 does not depend on candle 1's prices: change them, candle 2 is unchanged
    m2 = m.copy()
    m2.loc[:4, ["open", "close"]] *= 1.5
    assert np.isclose(multitarget.candle_rv(m2, "5min").iloc[1], second)


def test_trading_pnl_charges_round_trip_cost_and_multiplier():
    p = np.array([0.6, 0.4, 0.7, 0.2])
    mv = np.array([2.0, -1.0, -0.5, 1.0])
    ls = mr.positions(p, "long_short")
    assert ls.tolist() == [1, -1, 1, -1]
    t = mr.trades(ls, mv, cost=2.5)
    assert t["n_trades"] == 4
    assert np.isclose(t["gross_usd"], 20 * (2 + 1 - 0.5 - 1))
    assert np.isclose(t["net_usd"], t["gross_usd"] - 4 * 2.5)
    assert mr.positions(p, "long_only").tolist() == [1, 0, 1, 0]
    assert mr.positions(p, "short_only").tolist() == [0, -1, 0, -1]
    # drawdown on trade-by-trade equity from 0
    assert np.isclose(t["max_drawdown_usd"], min(0, (np.cumsum([37.5, 17.5, -12.5, -22.5]) -
                                                     np.maximum.accumulate(np.r_[0, np.cumsum([37.5, 17.5, -12.5, -22.5])])[1:]).min()))


def test_block_bootstrap_detects_a_clear_shift_and_not_noise():
    rng = np.random.default_rng(0)
    sess = np.repeat(np.arange(300), 20)
    est, lo, hi, z = mr.block_boot(1 + rng.normal(0, 1, len(sess)), sess)
    assert lo > 0 and z > 5
    est, lo, hi, z = mr.block_boot(rng.normal(0, 1, len(sess)), sess)
    assert lo < 0 < hi


def test_selection_one_per_task_then_two_extra_on_other_timeframes():
    def frame(task_rows):
        return pd.DataFrame([{"split": "val", "rows": "all", **r} for r in task_rows])
    R = frame([{"frequency": "5min", "model": "ridge", "z": 1.0}, {"frequency": "1h", "model": "ridge", "z": 0.5},
               {"frequency": "5min", "model": "zero", "z": 9.0}])
    D = frame([{"frequency": "5min", "model": "logistic", "z": 2.0}, {"frequency": "5min", "model": "xgboost", "z": 1.9},
               {"frequency": "1h", "model": "logistic", "z": 0.1}])
    V = frame([{"frequency": "5min", "model": "xgboost", "z": 30.0}, {"frequency": "1h", "model": "ridge", "z": 20.0},
               {"frequency": "1h", "model": "har_ols", "z": 19.0}, {"frequency": "5min", "model": "persistence", "z": 99}])
    _, sel = mr.select(R, D, V)
    assert len(sel) == 5 and set(sel.task) == {"return", "direction", "volatility"}
    assert "zero" not in set(sel.model) and "persistence" not in set(sel.model)
    # extras: volatility 1h (z 20) and return 1h (z 0.5) beat direction 1h (z 0.1); same-timeframe repeats excluded
    got = set(zip(sel.task, sel.frequency, sel.model))
    assert ("volatility", "1h", "ridge") in got and ("return", "1h", "ridge") in got
    assert ("direction", "5min", "xgboost") not in got


def test_f1_is_zero_not_nan_when_precision_and_recall_are_zero():
    # predictions all DOWN, truth has UP cases: no positive predictions
    s = mr.cls_stats([False, False, False], [True, False, True])
    assert s["precision_up"] == 0.0 and s["recall_up"] == 0.0 and s["f1_up"] == 0.0
    # positive predictions, all wrong: precision 0, recall 0, F1 0
    s = mr.cls_stats([True, False], [False, True])
    assert s["precision_up"] == 0.0 and s["recall_up"] == 0.0 and s["f1_up"] == 0.0
    # usual case agrees with the harmonic mean
    s = mr.cls_stats([True, True, False, False], [True, False, True, False])
    assert np.isclose(s["f1_up"], 0.5) and np.isclose(s["precision_up"], 0.5)
    # no positive case and no positive prediction: F1 0 by convention
    assert mr.cls_stats([False, False], [False, False])["f1_up"] == 0.0


def test_ratio_bootstrap_resamples_both_losses_together():
    rng = np.random.default_rng(1)
    sess = np.repeat(np.arange(400), 10)
    bench = rng.exponential(1.0, len(sess)) * np.repeat(rng.exponential(1.0, 400), 10)   # session-level clustering
    model = 0.7 * bench + 0.05 * rng.exponential(1.0, len(sess))
    est, lo, hi = mr.block_boot_ratio(model, bench, sess)
    assert np.isclose(est, 1 - model.sum() / bench.sum())
    assert lo < est < hi and lo > 0.2 and hi < 0.35
    # identical losses: ratio exactly 0 in every replicate
    est, lo, hi = mr.block_boot_ratio(bench, bench, sess)
    assert est == lo == hi == 0


def _ts_frame(n_sess=100, per=3, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2018-01-01", periods=n_sess + 1)
    org = np.repeat(days[:-1].to_numpy(), per)
    lab = org.copy()
    lab[per - 1::per] = days[1:].to_numpy()                     # last candle of each session: target in the next one
    x = rng.normal(-12, 1, len(org))
    return pd.DataFrame({"split": "val", "timestral_sample": True, "session_id": org, "label_session_id": lab,
                         "label_start": pd.DatetimeIndex(org).tz_localize("UTC"), "vol_raw__timestral": x,
                         "lrv_next": 1.0 + 0.9 * x + rng.normal(0, 0.3, len(org))})


def test_timestral_vol_calibration_is_past_to_future():
    p = _ts_frame()
    f, fac, meta = mr.crossfit_timestral_vol(p)
    org, lab = pd.to_datetime(p.session_id), pd.to_datetime(p.label_session_id)
    # first block has no earlier calibration: not scored
    first = org <= pd.Timestamp(meta[0]["eval_origin_sessions"][1])
    assert not meta[0]["scored"] and np.isnan(f[first.to_numpy()]).all() and np.isfinite(f[~first.to_numpy()]).all()
    for b in meta[1:]:
        # every calibration target is realised before the first origin it is used for
        assert pd.Timestamp(b["calib_target_sessions"][1]) < pd.Timestamp(b["eval_origin_sessions"][0])
    assert np.isfinite(fac[~first.to_numpy()]).all()


def test_changing_future_targets_does_not_change_earlier_timestral_forecasts():
    p = _ts_frame()
    f, fac, meta = mr.crossfit_timestral_vol(p)
    cut = pd.Timestamp(meta[3]["eval_origin_sessions"][0])
    q = p.copy()
    later = pd.to_datetime(q.label_session_id) >= cut
    q.loc[later, "lrv_next"] += 5.0                   # alter every target realised from block 3 on
    g, gac, _ = mr.crossfit_timestral_vol(q)
    early = (pd.to_datetime(p.session_id) < cut).to_numpy()
    assert np.array_equal(f[early], g[early], equal_nan=True) and np.array_equal(fac[early], gac[early], equal_nan=True)
    assert not np.allclose(f[~early & np.isfinite(f)], g[~early & np.isfinite(g)])   # later blocks do use them
