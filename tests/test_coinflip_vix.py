"""Conditional random-policy controls and same-expiry VIX alignment."""
import numpy as np
import pandas as pd
import pytest

from coinflip_vix import (build_front_dataset, chronological_masks, coin_draws,
                         holm_adjust, load_daily, nearest_contract_panel,
                         simulate_pnl)


def futures_fixture():
    dates = pd.bdate_range('2020-01-01', periods=140)
    expiries = pd.to_datetime(['2020-04-01', '2020-06-01', '2020-08-01',
                              '2020-10-01', '2020-12-01'])
    d = pd.DataFrame(index=dates)
    for col in range(1, 4):
        es, ps = [], []
        for j, date in enumerate(dates):
            candidates = expiries[expiries > date]
            e = candidates[col - 1]
            number = list(expiries).index(e)
            es.append(e.strftime('%d/%m/%Y'))
            ps.append(20 + 10 * number + .05 * j + .2 * np.sin(j))
        d[f'VXc{col}.E'] = es
        d['VXc2' if col == 2 else f'VXc{col}.P'] = ps
    return d


def test_closest_expiry_is_chosen_independent_of_column_order():
    d = futures_fixture()
    d[['VXc1.E', 'VXc3.E']] = d[['VXc3.E', 'VXc1.E']].to_numpy()
    d[['VXc1.P', 'VXc3.P']] = d[['VXc3.P', 'VXc1.P']].to_numpy()
    front, _ = nearest_contract_panel(d)
    assert front.expiry.iloc[0] == pd.Timestamp('2020-04-01')
    assert front.price.iloc[0] == 20


def test_missing_nearest_price_does_not_select_different_contract():
    d = futures_fixture()
    d.iloc[0, d.columns.get_loc('VXc1.P')] = np.nan
    front, _ = nearest_contract_panel(d)
    assert front.expiry.iloc[0] == pd.Timestamp('2020-04-01')
    assert np.isnan(front.price.iloc[0])


def test_same_expiry_context_and_targets_exclude_roll_jump():
    d = futures_fixture()
    x, y, meta, context, report = build_front_dataset(d, context=20, end_date='2020-07-10')
    # Each context follows a fixed contract, even when its column used to be second.
    assert np.max(np.abs(np.diff(context, axis=1))) < 1
    assert np.allclose(context[:, -1], meta.price)
    assert np.all(y.to_numpy() == np.sign(meta.next_price - meta.price))
    assert pd.Timestamp('2020-03-31') not in x.index
    assert pd.Timestamp('2020-05-29') not in x.index
    assert (meta.label_date <= pd.Timestamp('2020-07-10')).all()
    assert len(x) == len(y) == len(meta) == len(context)


def test_front_features_and_context_do_not_change_with_future_prices():
    d = futures_fixture()
    changed = d.copy()
    cutoff = pd.Timestamp('2020-05-01')
    for col in ['VXc1.P', 'VXc2', 'VXc3.P']:
        changed.loc[changed.index > cutoff, col] *= 3
    x1, _, _, c1, _ = build_front_dataset(d, 20, '2020-07-10')
    x2, _, _, c2, _ = build_front_dataset(changed, 20, '2020-07-10')
    pd.testing.assert_frame_equal(x1.loc[:cutoff], x2.loc[:cutoff])
    np.testing.assert_array_equal(c1[x1.index <= cutoff], c2[x2.index <= cutoff])


def test_coin_flats_and_deterministic_policy_have_exact_counts():
    y = np.array([1, 1, -1, 0])
    up = coin_draws(y, 1, 100, 1)
    down = coin_draws(y, 0, 100, 1)
    np.testing.assert_array_equal(up['accuracy'], np.full(100, .5))
    np.testing.assert_array_equal(down['accuracy'], np.full(100, .25))
    assert coin_draws(y, .5, 100, 1)['expected_accuracy'] == .375


def test_binomial_simulation_matches_independent_coin_experiment():
    y = np.array([1, 1, -1, -1, 0])
    d = coin_draws(y, .7, 100000, 3)
    brute = np.where(np.random.default_rng(4).random((100000, len(y))) < .7, 1, -1)
    assert abs(d['accuracy'].mean() - (brute == y).mean()) < .003
    assert abs(np.nanmean(d['balanced_accuracy']) - .5) < .003


def test_random_pnl_preserves_activity_and_costs():
    s = simulate_pnl([10., -5.], 2., [1, -1], 1000, p_up=1)
    assert s['n_trades'] == 2
    assert s['model_net_usd'] == 11
    assert s['random_mean_net_usd'] == 1
    assert s['random_lo95_net_usd'] == s['random_hi95_net_usd'] == 1


def test_exact_daily_duplicates_removed_but_conflicts_fail(tmp_path):
    p = tmp_path / 'd.csv'
    p.write_text('Date;C\n01/01/2020;20\n01/01/2020;20\n', encoding='utf-8')
    d, r = load_daily(p, 'Date')
    assert len(d) == 1 and r['exact_duplicates_removed'] == 1
    p.write_text('Date;C\n01/01/2020;20\n01/01/2020;21\n', encoding='utf-8')
    with pytest.raises(ValueError, match='conflicting'):
        load_daily(p, 'Date')


def test_splits_purge_label_crossing_and_holm_adjustment():
    idx = pd.bdate_range('2020-01-01', periods=150)
    m = pd.DataFrame({'label_date': pd.Series(idx, index=idx).shift(-1)}, index=idx)
    masks = chronological_masks(m, idx[50], idx[100])
    assert not masks['train'][49]
    assert not masks['validation'][99]
    np.testing.assert_allclose(holm_adjust([.01, .02, .04]), [.03, .04, .04])
