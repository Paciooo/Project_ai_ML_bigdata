import numpy as np
import pandas as pd
import pytest

import strategy


def test_positions_by_direction_and_threshold():
    p = np.array([0.70, 0.52, 0.50, 0.48, 0.30])
    assert strategy.positions(p, "long_short").tolist() == [1, 1, 1, -1, -1]
    assert strategy.positions(p, "long_only").tolist() == [1, 1, 1, 0, 0]
    assert strategy.positions(p, "short_only").tolist() == [0, 0, 0, -1, -1]
    assert strategy.positions(p, "long_short", 0.05).tolist() == [1, 0, 0, 0, -1]
    with pytest.raises(ValueError):
        strategy.positions(p, "both")


def test_long_and_short_pnl_use_nq_multiplier_and_round_trip_cost():
    # move = close(t+1) - open(t+1) in index points
    gross, net = strategy.trade_pnl([1, -1, 0], [2.0, 2.0, 2.0])
    assert gross.tolist() == [40.0, -40.0, 0.0]          # $20 per point
    assert net.tolist() == [35.0, -45.0, 0.0]            # $5 per completed round trip, none when flat


def test_doji_trade_pays_costs_only():
    gross, net = strategy.trade_pnl([1], [0.0])
    assert gross[0] == 0 and net[0] == -strategy.DEFAULT_RT_COST


def test_summary_counts_sides_and_drawdown_from_zero():
    pos = np.array([1, -1, 1, 0])
    move = np.array([-1.0, -1.0, -3.0, 5.0])
    sess = pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"])
    s = strategy.summarize(pos, move, sess, bootstrap=False)
    assert s["n_trades"] == 3 and s["n_long"] == 2 and s["n_short"] == 1
    assert s["gross_usd"] == -20 + 20 - 60
    assert s["costs_usd"] == 15
    assert s["net_usd"] == -75
    assert s["long_net_usd"] == -25 - 65 and s["short_net_usd"] == 15
    # day 1: -25 + 15 = -10, day 2: -65 -> equity 0, -10, -75: drawdown includes the start
    assert s["max_drawdown_usd"] == -75


def test_threshold_choice_uses_only_given_sample():
    p = np.array([0.9] * 300 + [0.51] * 300)
    move = np.array([1.0] * 300 + [-1.0] * 300)
    assert strategy.choose_threshold(p, move, "long_only", (0.0, 0.1, 0.3)) == (0.1, strategy.THRESHOLD_OK)


def test_positions_reject_invalid_probabilities_and_thresholds():
    for bad in ([0.5, np.nan], [0.5, np.inf], [1.2], [-0.1]):
        with pytest.raises(ValueError):
            strategy.positions(np.array(bad), "long_short")
    for th in (-0.01, 0.6, np.nan, np.inf):
        with pytest.raises(ValueError):
            strategy.positions(np.array([0.6]), "long_short", th)
    assert strategy.positions(np.array([0.0, 1.0]), "long_short", 0.5).tolist() == [-1, 1]


def test_threshold_choice_reports_no_threshold_instead_of_zero():
    p = np.full(150, 0.7)
    move = np.ones(150)
    th, status = strategy.choose_threshold(p, move, "long_short", (0.0, 0.1), min_trades=200)
    assert th is None and status == strategy.THRESHOLD_NONE


def test_threshold_choice_counts_only_executable_entries():
    # 300 strong calls, but only 150 executable: no threshold reaches 200 trades
    p = np.r_[np.full(300, 0.9), np.full(300, 0.52)]
    move = np.r_[np.ones(300), -np.ones(300)]
    ex = np.r_[np.ones(150, bool), np.zeros(150, bool), np.ones(300, bool)]
    assert strategy.choose_threshold(p, move, "long_only", (0.0, 0.1), executable=ex) == (0.0, strategy.THRESHOLD_OK)
    ex2 = np.r_[np.ones(150, bool), np.zeros(450, bool)]
    assert strategy.choose_threshold(p, move, "long_only", (0.0, 0.1), executable=ex2)[0] is None
    # non-executable rows cost nothing in the objective: without the mask 0.0 would lose to 0.1
    p3 = np.r_[np.full(300, 0.9), np.full(300, 0.52)]
    ex3 = np.r_[np.ones(300, bool), np.zeros(300, bool)]
    assert strategy.choose_threshold(p3, move, "long_only", (0.0, 0.1), executable=ex3) == (0.0, strategy.THRESHOLD_OK)


def test_threshold_choice_uses_the_given_cost():
    # gross +$6 per trade on 300 trades (0.3 pt): profitable at $5, not at $12.50
    p = np.r_[np.full(300, 0.9), np.full(300, 0.52)]
    move = np.r_[np.full(300, 0.3), np.full(300, 0.3)]
    assert strategy.choose_threshold(p, move, "long_only", (0.0, 0.1), rt_cost=5.0)[0] == 0.0
    assert strategy.choose_threshold(p, move, "long_only", (0.0, 0.1), rt_cost=6.5)[0] == 0.1


def test_delayed_entry_is_flagged_as_not_the_candle_body():
    import study
    ls = pd.to_datetime(["2024-01-02 15:00", "2024-01-02 15:05", "2024-01-02 15:10"], utc=True)
    meta = pd.DataFrame({"label_start": ls, "entry_at": [ls[0], ls[1] + pd.Timedelta("1min"), pd.NaT]})
    y3, move, elig = np.array([1, -1, 1]), np.array([2.0, 0.5, 0.0]), np.array([True, True, False])
    delayed, mismatch, kind = study.traded_return_flags(meta, move, y3, elig)
    assert delayed.tolist() == [False, True, False] and mismatch == 1 and kind == "delayed_entry_to_close"
    meta0 = meta.assign(entry_at=[ls[0], ls[1], pd.NaT])
    assert study.traded_return_flags(meta0, np.array([2.0, -0.5, 0.0]), y3, elig)[2] == "candle_body"


def test_no_executable_entry_is_not_evaluable_not_zero():
    import study
    ls = pd.to_datetime(["2024-01-02 15:00", "2024-01-02 15:01"], utc=True)
    meta = pd.DataFrame({"label_start": ls, "entry_at": pd.Series([pd.NaT, pd.NaT], dtype=ls.dtype)})
    flags = study.traded_return_flags(meta, np.zeros(2), np.array([1, -1]), np.array([False, False]))
    assert flags[2] == strategy.NOT_EVALUABLE
    sess = pd.to_datetime(["2024-01-02", "2024-01-02"])
    s = strategy.summarize(np.zeros(2, int), np.zeros(2), sess, executable=np.array([False, False]))
    assert s["economic_status"] == strategy.NOT_EVALUABLE and np.isnan(s["net_usd"]) and s["n_trades"] == 0
    # opportunities exist but the strategy chooses not to trade: evaluated, zero P&L
    s2 = strategy.summarize(np.zeros(2, int), np.array([1.0, -1.0]), sess, bootstrap=False,
                            executable=np.array([True, True]))
    assert s2["economic_status"] == strategy.EVALUATED and s2["net_usd"] == 0 and s2["n_trades"] == 0
    with pytest.raises(ValueError):
        strategy.summarize(np.array([1, 0]), np.ones(2), sess, executable=np.array([False, True]))


def test_report_requires_walk_forward_status_for_old_runs(tmp_path):
    import json
    import report
    for k in ("model_selection", "accuracy_test", "strategies", "equity_daily", "cost_sensitivity",
              "threshold_sensitivity"):
        (tmp_path / f"{k}.csv").write_text("a\n1\n")
    (tmp_path / "walk_forward.csv").write_text("frequency,fold,threshold_kind,net_usd\nsession,0,validated,1.0\n")
    (tmp_path / "dataset_info.json").write_text("{}")
    (tmp_path / "run.json").write_text(json.dumps({"args": {"cost": 5.0}}))
    with pytest.raises(SystemExit, match="wf_threshold_status"):
        report.load(tmp_path)
