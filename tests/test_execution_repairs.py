"""Regressions for the three confirmed economic-scoring defects."""
import numpy as np
import pandas as pd
import pytest

import backtest


def minute_data():
    start = pd.date_range('2024-01-02 14:00', periods=8, freq='min', tz='UTC')
    return pd.DataFrame(dict(bar_start=start, bar_end=start + pd.Timedelta(minutes=1),
                             open=np.arange(8.) + 100, session_id=pd.Timestamp('2024-01-02'),
                             suspect=False))


def target(end='2024-01-02 14:05', decision='2024-01-02 14:00:01'):
    return pd.DataFrame(dict(decision_at=pd.to_datetime([decision], utc=True),
                             label_start=pd.to_datetime(['2024-01-02 14:00'], utc=True),
                             label_end=pd.to_datetime([end], utc=True),
                             session_id=[pd.Timestamp('2024-01-02')],
                             label_session_id=[pd.Timestamp('2024-01-02')], exit_price=[105.]))


def test_entry_is_first_minute_open_after_decision():
    m = backtest.attach_execution(target(), minute_data())
    assert m.entry_at.iloc[0] == pd.Timestamp('2024-01-02 14:01', tz='UTC')
    assert m.entry_price.iloc[0] == 101
    assert m.execution_eligible.iloc[0]
    assert m.entry_at.iloc[0] >= m.decision_at.iloc[0]


def test_positive_latency_makes_one_minute_economics_unobservable():
    m = backtest.attach_execution(target(end='2024-01-02 14:01'), minute_data())
    assert not m.execution_eligible.iloc[0]
    t = backtest.run_backtest(m, [1], backtest.COST_SCENARIOS['base'])
    s = backtest.summarize(t)
    assert s['n_trades'] == 0 and np.isnan(s['net_usd'])
    assert s['economic_status'].startswith('not_evaluable')


def test_zero_latency_is_explicit_and_can_enter_at_boundary():
    m = backtest.attach_execution(target(end='2024-01-02 14:01', decision='2024-01-02 14:00'), minute_data())
    assert m.entry_price.iloc[0] == 100 and m.execution_eligible.iloc[0]


def test_suspect_entry_is_rejected_not_skipped_to_later_price():
    d = minute_data()
    d.loc[1, 'suspect'] = True
    m = backtest.attach_execution(target(), d)
    assert not m.execution_eligible.iloc[0]


def test_backtest_rejects_entry_before_decision():
    m = backtest.attach_execution(target(), minute_data())
    m['entry_at'] = m.decision_at - pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match='precedes'):
        backtest.run_backtest(m, [1], backtest.COST_SCENARIOS['base'])


def test_missing_timing_is_not_silently_accepted():
    with pytest.raises(ValueError, match='explicit execution'):
        backtest.run_backtest(pd.DataFrame({'entry_price':[1.], 'exit_price':[2.]}), [1],
                              backtest.COST_SCENARIOS['base'])


def test_fold_specific_thresholds_survive_concatenation_and_drive_positions():
    a = pd.DataFrame({'proba_x':[.55], 'threshold_x':[.01], 'pred_x':[1]})
    b = pd.DataFrame({'proba_x':[.55], 'threshold_x':[.1], 'pred_x':[1]})
    p = pd.concat([a,b], ignore_index=True)
    assert list(backtest.frame_positions(p, 'x')) == [1,0]
    with pytest.raises(ValueError, match='restore fold'):
        backtest.frame_positions(p.drop(columns='threshold_x'), 'x')


def test_initial_losses_are_included_in_drawdown():
    t = pd.DataFrame({'session_id':pd.to_datetime(['2024-01-02','2024-01-03']),
                       'pos':[1,1], 'gross_usd':[-1000.,-500.], 'cost_usd':[0.,0.],
                       'net_usd':[-1000.,-500.], 'execution_eligible':[True,True]})
    s = backtest.summarize(t)
    assert s['max_drawdown_usd'] == -1500
    assert s['max_drawdown_eod_usd'] == -1500


def test_entry_cannot_cross_target_session():
    d = minute_data()
    d['session_id'] = pd.Timestamp('2024-01-03')
    assert not backtest.attach_execution(target(), d).execution_eligible.iloc[0]


def test_unobservable_entries_keep_utc_when_transferred_to_saved_frame():
    m = backtest.attach_execution(target(end='2024-01-02 14:01'), minute_data())
    saved = target(end='2024-01-02 14:01').copy()
    for col in ('entry_at','exit_at','entry_price','execution_eligible'):
        saved[col] = m[col].array
    t = backtest.run_backtest(saved, [1], backtest.COST_SCENARIOS['base'])
    assert not t.pos.any()


def test_restore_thresholds_uses_fold_ids_and_rejects_missing_folds():
    from repair_results import restore_thresholds
    p = pd.DataFrame({'fold':[1,0], 'proba_x':[.55,.55], 'pred_x':[1,1]})
    folds = pd.DataFrame({'frequency':['1h','1h'], 'fold':[0,1], 'threshold_x':[.01,.1]})
    fixed = restore_thresholds(p, folds, '1h')
    assert list(backtest.frame_positions(fixed,'x')) == [0,1]
    with pytest.raises(ValueError, match='missing threshold'):
        restore_thresholds(p, folds.iloc[:1], '1h')
