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
    assert strategy.choose_threshold(p, move, "long_only", (0.0, 0.1, 0.3)) == 0.1
