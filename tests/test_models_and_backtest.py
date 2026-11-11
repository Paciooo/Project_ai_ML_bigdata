"""Evaluation, backtest, controls, regimes, CLI and the Timestral interface."""

import numpy as np
import pandas as pd
import pytest

import backtest
import data
import evaluation
import features
import regimes
import timeframes as tf

LAGS, MOM, VOL = [1, 2, 3, 5], [5, 10], [5, 10]


def dataset(bounce, n=30, seed=0, target="next_close_direction"):
    m, _ = tf.add_sessions(data.fixture_minutes(n, seed=seed, bounce=bounce))
    m["suspect"] = False
    b = tf.aggregate(m, "1min")
    return features.build_timed_dataset(b, "1min", target, LAGS, MOM, VOL,
                                        label_eligible=tf.next_label_eligible(b, "1min"))


def fit_eval(X, y3, meta):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    v, t = features.session_boundaries(meta["session_id"], 0.6, 0.2)
    tr, te = (meta["session_id"] < v).to_numpy(), (meta["session_id"] >= t).to_numpy()
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(X[tr & (y3 != 0).to_numpy()], (y3[tr & (y3 != 0).to_numpy()] > 0))
    pred = np.where(m.predict_proba(X[te])[:, 1] >= 0.5, 1, -1)
    return (pred == y3[te].to_numpy()).mean(), meta[te], y3[te]


def test_positive_control_recovers_injected_signal():
    X, y3, meta, _ = dataset(bounce=0.5)
    acc, _, _ = fit_eval(X, y3, meta)
    assert 0.53 < acc < 0.65          # recovered, and not implausibly high (no leak)


def test_null_control_has_no_edge():
    X, y3, meta, _ = dataset(bounce=0.0, seed=3)
    acc, _, _ = fit_eval(X, y3, meta)
    assert abs(acc - 0.5) < 0.02


def test_leak_would_be_caught():
    """If features used the label bar, accuracy would explode: the guard catches it."""
    X, y3, meta, _ = dataset(bounce=0.0, seed=3)
    leaky = X.copy()
    leaky["ret_lag1"] = np.log(meta["exit_price"] / meta["close"]).to_numpy()   # future return
    acc, _, _ = fit_eval(leaky, y3, meta)
    assert acc > 0.9


def test_rules_and_zero_return():
    X = pd.DataFrame({"ret_lag1": [0.1, -0.2, 0.0]})
    assert list(evaluation.rule_predictions(X, "persistence", -1)) == [1, -1, -1]
    assert list(evaluation.rule_predictions(X, "reversal", 1)) == [-1, 1, 1]


def test_direction_metrics_flat_policy():
    m = evaluation.direction_metrics(np.array([1, 1, -1, 1]), np.array([1, 0, -1, -1]),
                                     proba_up=np.array([0.6, 0.6, 0.4, 0.7]))
    assert m["acc_nonflat"] == pytest.approx(2 / 3) and m["acc_all"] == pytest.approx(0.5)
    assert m["coverage_nonflat"] == 0.75 and "brier_nonflat" in m


def test_block_bootstrap_interval_and_verdict():
    rng = np.random.default_rng(0)
    sess = np.repeat(np.arange(200), 50)
    good = rng.random(len(sess)) < 0.56
    base = rng.random(len(sess)) < 0.50
    out = evaluation.paired_difference(good, base, sess)
    assert out["ci_lo_pp_b20"] > 0 and out["verdict"] == "better than baseline"
    few = evaluation.paired_difference(good[:500], base[:500], sess[:500])
    assert few["verdict"].startswith("inconclusive")


def execution_fixture(meta):
    meta = meta.copy()
    start = pd.date_range("2024-01-02 14:00", periods=len(meta), freq="min", tz="UTC")
    meta["decision_at"] = start
    meta["entry_at"] = start + pd.Timedelta(seconds=1)
    meta["exit_at"] = start + pd.Timedelta(minutes=1)
    meta["execution_eligible"] = True
    return meta


def test_backtest_fills_and_costs():
    meta = pd.DataFrame({"session_id": pd.to_datetime(["2024-01-02"] * 3),
                         "close": [100.0, 101.0, 102.0], "entry_price": [100.5, 101.25, 101.0],
                         "exit_price": [101.0, 102.0, 100.0]})
    meta = execution_fixture(meta)
    pos = np.array([1, -1, 0])
    t = backtest.run_backtest(meta, pos, backtest.COST_SCENARIOS["zero_cost"])
    # long from the NEXT open 100.5 to 101.0 (+0.5 pt), short 101.25 -> 102.0 (-0.75 pt), never the signal close
    assert list(t["gross_usd"]) == [10.0, -15.0, 0.0]
    c = backtest.round_trip_cost_usd(**backtest.COST_SCENARIOS["base"])
    assert c == pytest.approx(2 * 2.5 + 0.25 * 20 + 2 * 0.25 * 0.25 * 20)
    tb = backtest.run_backtest(meta, pos, backtest.COST_SCENARIOS["base"])
    assert tb["net_usd"].sum() == pytest.approx(-5.0 - 2 * c)
    s = backtest.summarize(tb, capital_usd=100_000)
    assert s["n_trades"] == 2 and s["turnover_contracts"] == 4 and "net_return_pct_on_capital" in s


def test_flat_outcome_is_traded_not_filtered():
    meta = pd.DataFrame({"session_id": pd.to_datetime(["2024-01-02"]), "entry_price": [100.0],
                         "exit_price": [100.0]})
    t = backtest.run_backtest(execution_fixture(meta), np.array([1]), backtest.COST_SCENARIOS["base"])
    assert t["gross_usd"][0] == 0 and t["net_usd"][0] < 0


def test_threshold_positions():
    pos = backtest.positions(np.array([0.52, 0.49, 0.7, 0.3]), threshold=0.05)
    assert list(pos) == [0, 0, 1, -1]


def test_roll_dates():
    r = backtest.approx_roll_sessions(pd.to_datetime(["2024-03-01"]))
    assert pd.Timestamp("2024-03-07") in r          # third Friday 15 Mar - 8 days


def test_pettitt_and_segmentation():
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(0, 1, 200), rng.normal(2, 1, 200)]
    k, K, p = regimes.pettitt(x)
    assert abs(k - 199) <= 5 and p < 1e-6
    starts = regimes.binary_segmentation(x, 3, min_size=30)
    assert any(abs(s - 200) <= 5 for s in starts[1:])
    assert regimes.binary_segmentation(rng.normal(0, 1, 300), 3, 30) == [0]


def test_regime_router_is_causal():
    m, _ = tf.add_sessions(data.fixture_minutes(80, seed=2, bounce=0.2))
    m["suspect"] = False
    s = tf.aggregate(m, "session")
    router = regimes.RegimeRouter(1, 10).fit(s.iloc[:50])
    p1 = router.predict_sessions(s)
    s2 = s.copy()
    s2.loc[s2.index[60], "close"] *= 1.2           # change session 60's close
    p2 = router.predict_sessions(s2)
    # session 60's own group cannot depend on its own close; only later sessions may change
    assert p1.iloc[:61].equals(p2.iloc[:61])


def test_cli_rejects_unknown_source_and_bad_args():
    import main
    with pytest.raises(SystemExit):
        main.parse_args(["--data-source", "bogus"])
    with pytest.raises(SystemExit):
        main.parse_args(["--data-source", "nq", "--frequency", "tick"])
    a = main.parse_args(["--smoke"])
    assert a.data_source == "fixture" and a.max_train_rows > 0


def test_smoke_run_writes_new_folder(tmp_path, monkeypatch):
    import config
    import main
    monkeypatch.setattr(config, "RESULTS_DIR", str(tmp_path))
    out = main.main(["--smoke", "--frequency", "1min", "--models", "logreg"])
    assert out.parent.parent.name == "smoke" and (out / "run.json").exists()
    preds = pd.read_parquet(out / "predictions_1min.parquet")
    assert {"decision_at", "label_end", "proba_logreg", "y3"} <= set(preds.columns)
    out2 = main.main(["--smoke", "--frequency", "1min", "--models", "logreg"])
    assert out2 != out and (out / "run.json").exists()


def test_missing_futures_data_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="No replacement data"):
        data.load_futures_minutes(tmp_path / "absent", "NQ", 2020, 2021)
