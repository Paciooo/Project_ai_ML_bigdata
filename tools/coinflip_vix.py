"""Small exploratory extension: random decisions and nearest-expiry VIX futures.

Only one futures expiry is forecast at each origin: the nearest strictly
unexpired expiry in the input file. Historical context tracks THAT expiry
through its former column positions. No other maturity is a predictor.
Targets crossing a front-contract switch are excluded. Prices are daily
vendor observations, not verified executable fills or timestamped settlements.

Coin accuracy draws use exact binomial sufficient statistics: equivalent to
independently tossing one coin per eligible observation, without allocating
10,000 x millions of predictions. These are CONDITIONAL random-policy nulls,
not confidence intervals for future market performance or a selection test.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
import zlib
from pathlib import Path

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import backtest
import evaluation
import features
import runlog

VIX_DIR = Path(r'C:\Users\aless\Desktop\Scuola\BSAMC\VIX\Codice\data')
DEFAULT_NQ_RUNS = [
    ROOT / 'results/nq/runs/20261102T060506Z_892e22_A_holdout_close',
    ROOT / 'results/nq/runs/20261102T060756Z_a695aa_B_holdout_body',
    ROOT / 'results/nq/runs/20261102T061204Z_cdd340_C_walkforward',
]
TIMESTRAL_SIZES = ('5M', '28M', '123M', '209M')


def load_daily(path, date_column):
    d = pd.read_csv(path, sep=';', na_values=['NULL'])
    raw_n = len(d)
    d[date_column] = pd.to_datetime(d[date_column], format='%d/%m/%Y', errors='raise')
    d = d.drop_duplicates()  # whole-record equality, never arbitrary duplicate-date retention
    if d[date_column].duplicated().any():
        raise ValueError(f'{path}: conflicting duplicate dates require source correction')
    d = d.set_index(date_column).sort_index()
    if d.index.hasnans:
        raise ValueError('missing observation dates')
    return d, dict(path=str(Path(path).resolve()), raw_rows=raw_n, unique_rows=len(d),
                   exact_duplicates_removed=raw_n - len(d), first=str(d.index.min().date()),
                   last=str(d.index.max().date()), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def nearest_contract_panel(futures):
    """Build same-expiry histories; choose the nearest expiry before testing price validity."""
    pieces = []
    for i in range(1, 10):
        pc = 'VXc2' if i == 2 else f'VXc{i}.P'
        ec = f'VXc{i}.E'
        if pc not in futures or ec not in futures:
            continue
        q = pd.DataFrame({'date': futures.index,
                          'expiry': pd.to_datetime(futures[ec], format='%d/%m/%Y', errors='coerce').to_numpy(),
                          'price': pd.to_numeric(futures[pc], errors='coerce').to_numpy()})
        pieces.append(q)
    if not pieces:
        raise ValueError('no price/expiry column pairs found')
    long = pd.concat(pieces, ignore_index=True).dropna(subset=['expiry'])
    if long.duplicated(['date', 'expiry']).any():
        raise ValueError('multiple columns identify the same expiry on one date')
    unexpired = long[long.expiry > long.date]
    first = unexpired.sort_values(['date', 'expiry']).drop_duplicates('date').set_index('date')
    front = first.reindex(futures.index)
    # Keep all observed dates. A missing nearest quote must NOT silently select the next contract.
    panel = long.pivot(index='date', columns='expiry', values='price').reindex(futures.index)
    panel = panel.where(np.isfinite(panel) & (panel > 0))
    return front, panel


def price_features(close):
    log = np.log(close)
    ret = log.diff()
    x = pd.DataFrame(index=close.index)
    for k in (1, 2, 3, 5):
        x[f'ret_lag{k}'] = ret.shift(k - 1)
    for n in (5, 10):
        x[f'mom_{n}'] = log - log.shift(n)
        x[f'vol_{n}'] = ret.rolling(n).std()
    x['log_level'] = log
    x['gap_mean20'] = log - log.rolling(20).mean()
    return x


def build_front_dataset(futures, context=64, end_date='2025-05-30'):
    """Same-expiry features and context ending at t; next observed day's SAME front contract label."""
    f = futures.loc[:pd.Timestamp(end_date)]
    front, panel = nearest_contract_panel(f)
    future_date = pd.Series(f.index, index=f.index).shift(-1)
    same_expiry = front.expiry.eq(front.expiry.shift(-1))
    next_price = front.price.shift(-1)
    y = np.sign(next_price - front.price)
    x = pd.DataFrame(index=f.index)
    contexts = {}
    for expiry in front.expiry.dropna().unique():
        expiry = pd.Timestamp(expiry)
        s = panel[expiry]
        xf = price_features(s)
        origins = np.flatnonzero(front.expiry.eq(expiry).to_numpy())
        for pos in origins:
            ctx = s.iloc[max(0, pos - context + 1):pos + 1].to_numpy(float)
            if len(ctx) == context and np.isfinite(ctx).all() and (ctx > 0).all():
                date = f.index[pos]
                x.loc[date, xf.columns] = xf.iloc[pos]
                contexts[date] = ctx
    valid_price = np.isfinite(front.price) & (front.price > 0) & np.isfinite(next_price) & (next_price > 0)
    # Next observation, not an invented calendar-day horizon; reject long data outages.
    gap_ok = (future_date - pd.Series(f.index, index=f.index)) <= pd.Timedelta(days=7)
    keep = same_expiry & valid_price & gap_ok & y.notna() & x.notna().all(axis=1)
    meta = pd.DataFrame({'date': f.index, 'label_date': future_date.to_numpy(),
                         'expiry': front.expiry.to_numpy(), 'price': front.price.to_numpy(),
                         'next_price': next_price.to_numpy()}, index=f.index)
    idx = x.index[keep]
    if len(idx) == 0:
        raise ValueError('no eligible nearest-expiry observations with complete same-contract history')
    return x.loc[idx], y.loc[idx].astype(int), meta.loc[idx], np.stack([contexts[t] for t in idx]), dict(
        raw_unique_dates=len(f), usable_origins=len(idx), context=context,
        front_switch_or_missing_next=int((~same_expiry).sum()),
        first_origin=str(idx.min().date()), last_origin=str(idx.max().date()),
        history='same expiry across historical column positions; no maturity-curve predictors',
        expiry_convention='vendor expiry fields used as identity; actual exchange calendar not verified')


def chronological_masks(meta, val_start, test_start):
    v, t = pd.Timestamp(val_start), pd.Timestamp(test_start)
    if not v < t:
        raise ValueError('validation must precede test')
    masks = dict(train=(meta.index < v) & (meta.label_date < v).to_numpy(),
                 validation=(meta.index >= v) & (meta.index < t) & (meta.label_date < t).to_numpy(),
                 test=meta.index >= t)
    if any(np.sum(m) < 30 for m in masks.values()):
        raise ValueError('at least 30 observations required in each purged split')
    return masks


def coin_draws(y, p_up, simulations, seed):
    """Exact distribution of counts for an independent Bernoulli direction policy on FIXED labels."""
    y = np.asarray(y)
    if len(y) == 0 or not np.isin(y, [-1, 0, 1]).all() or not 0 <= p_up <= 1 or simulations < 1:
        raise ValueError('invalid coin simulation inputs')
    rng = np.random.default_rng(seed)
    n_up, n_down = int((y == 1).sum()), int((y == -1).sum())
    tp = rng.binomial(n_up, p_up, size=simulations)
    tn = rng.binomial(n_down, 1 - p_up, size=simulations)
    balanced = .5 * (tp / n_up + tn / n_down) if n_up and n_down else np.full(simulations, np.nan)
    return dict(accuracy=(tp + tn) / len(y), balanced_accuracy=balanced,
                expected_accuracy=(n_up * p_up + n_down * (1 - p_up)) / len(y))


def monte_carlo_rows(y, pred, dataset, frequency, model, simulations, seed):
    pred, y = np.asarray(pred), np.asarray(y)
    if pred.shape != y.shape or not np.isin(pred, [-1, 1]).all():
        raise ValueError('predictions must be aligned UP/DOWN calls')
    observed = float((pred == y).mean())
    nonflat = y != 0
    bal = evaluation.direction_metrics(pred, y)['balanced_acc_nonflat']
    rows = []
    for null, prob in [('fair_coin', .5), ('matched_expected_up_share', float((pred == 1).mean()))]:
        d = coin_draws(y, prob, simulations, seed)
        lo, hi = np.quantile(d['accuracy'], [.025, .975])
        # Conditional tail frequency, NOT a dependence-robust market-alpha p-value.
        rows.append(dict(dataset=dataset, frequency=frequency, model=model, null=null,
                         n=len(y), n_nonflat=int(nonflat.sum()), simulations=simulations, p_up=prob,
                         model_accuracy=observed, model_balanced_accuracy=bal,
                         random_expected_accuracy=d['expected_accuracy'], random_mean=float(d['accuracy'].mean()),
                         random_lo95=float(lo), random_hi95=float(hi),
                         model_minus_random_mean_pp=100 * (observed - d['accuracy'].mean()),
                         conditional_tail_fraction=float((1 + (d['accuracy'] >= observed).sum()) / (simulations + 1))))
    return rows


def simulate_pnl(payoffs, costs_per_trade, predictions, simulations=10000, seed=42, p_up=.5):
    """Random signs at exactly the same active times and costs; conditional on fixed historical payoffs."""
    payoffs, predictions = np.asarray(payoffs, float), np.asarray(predictions, int)
    if len(payoffs) != len(predictions) or not np.isfinite(payoffs).all():
        raise ValueError('unaligned or invalid payoffs')
    rng = np.random.default_rng(seed)
    net = np.empty(simulations)
    for start in range(0, simulations, 128):
        size = min(128, simulations - start)
        signs = np.where(rng.random((size, len(payoffs))) < p_up, 1., -1.)
        net[start:start + size] = signs @ payoffs - len(payoffs) * costs_per_trade
    obs = float(predictions @ payoffs - len(payoffs) * costs_per_trade)
    return dict(n_trades=len(payoffs), model_net_usd=obs, random_mean_net_usd=float(net.mean()),
                random_lo95_net_usd=float(np.quantile(net, .025)),
                random_hi95_net_usd=float(np.quantile(net, .975)),
                conditional_tail_fraction=float((1 + (net >= obs).sum()) / (simulations + 1)))


def holm_adjust(values):
    """Familywise adjustment for descriptive random-policy tail comparisons."""
    p = np.asarray(values, float)
    order = np.argsort(p)
    ans = np.empty(len(p))
    ans[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return ans


def nq_comparisons(paths, args):
    rows, pnl_rows = [], []
    for run in paths:
        run = Path(run)
        if not run.is_dir():
            raise FileNotFoundError(run)
        folds = pd.read_csv(run / 'folds.csv')
        for pf in sorted(run.glob('predictions_*.parquet')):
            freq = pf.stem.removeprefix('predictions_')
            p = pd.read_parquet(pf)
            name = run.name.split('_', 2)[-1]
            dataset = 'NQ/' + name
            y = p.y3.to_numpy()
            for col in [c for c in p if c.startswith('pred_')]:
                model = col.removeprefix('pred_')
                rows.extend(monte_carlo_rows(y, p[col], dataset, freq, model,
                                             args.simulations, args.seed))
                # Do not reuse the known invalid intraday entry assumption for a profit test.
                timing_ok = (p.label_start >= p.decision_at).to_numpy()
                if not timing_ok.all():
                    pnl_rows.append(dict(dataset=dataset, frequency=freq, model=model,
                                         status='skipped: assumed entry precedes configured decision'))
                    continue
                if f'proba_{model}' in p:
                    key = f'threshold_{model}'
                    ff = folds[folds.frequency == freq].set_index('fold')
                    if key not in ff:
                        raise ValueError(f'missing per-fold thresholds for {model}')
                    th = p.fold.map(ff[key]).to_numpy()
                    if np.isnan(th).any():
                        raise ValueError('missing fold threshold')
                    active = np.abs(p[f'proba_{model}'].to_numpy() - .5) >= th
                else:
                    active = np.ones(len(p), bool)
                if freq == 'session':
                    active &= ~p.session_id.isin(backtest.approx_roll_sessions(p.session_id)).to_numpy()
                if not active.any():
                    continue
                payoffs = (p.exit_price - p.entry_price).to_numpy()[active] * backtest.NQ_MULTIPLIER
                pred = p[col].to_numpy()[active]
                for null, prob in [('fair_coin', .5), ('matched_expected_up_share', float((pred == 1).mean()))]:
                    for scen in ('zero_cost', 'base'):
                        cost = backtest.round_trip_cost_usd(**backtest.COST_SCENARIOS[scen])
                        stat = simulate_pnl(payoffs, cost, pred, args.simulations, args.seed, prob)
                        pnl_rows.append(dict(dataset=dataset, frequency=freq, model=model, null=null,
                                             costs=scen, status='illustrative bar-proxy P&L, not verified fills', **stat))
            print(f'Coin simulations: {dataset} {freq}, {len(p):,} observations', flush=True)
    return rows, pnl_rows


def vix_models(x, y, meta, contexts, args, out):
    masks = chronological_masks(meta, args.val_start, args.test_start)
    tr, va, te = (masks[k] for k in ('train', 'validation', 'test'))
    train = tr & (y.to_numpy() != 0)
    majority = 1 if (y.iloc[np.flatnonzero(train)] > 0).mean() >= .5 else -1
    pred = meta.loc[te].copy()
    pred['y3'] = y.loc[te]
    train_median = meta.loc[tr, 'price'].median()
    rule_values = {
        'majority': lambda mask: np.full(int(mask.sum()), majority),
        'persistence': lambda mask: np.where(x.loc[mask, 'ret_lag1'] > 0, 1,
                                      np.where(x.loc[mask, 'ret_lag1'] < 0, -1, majority)),
        'reversal': lambda mask: np.where(x.loc[mask, 'ret_lag1'] > 0, -1,
                                   np.where(x.loc[mask, 'ret_lag1'] < 0, 1, majority)),
        'train_median_reversion': lambda mask: np.where(meta.loc[mask, 'price'] > train_median, -1, 1),
    }
    val_scores = {}
    for name, fn in rule_values.items():
        pred[f'pred_{name}'] = fn(te)
        val_scores[name] = evaluation.direction_metrics(fn(va), y.loc[va])['balanced_acc_nonflat']
    for name in ('logreg', 'xgboost'):
        if name == 'logreg':
            estimator = LogisticRegression(C=1, max_iter=1000)
        else:
            from xgboost import XGBClassifier
            estimator = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=.05,
                                      subsample=.8, n_jobs=2, random_state=args.seed, tree_method='hist')
        m = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), estimator)
        m.fit(x.loc[train], (y.loc[train] > 0).astype(int))
        for mask, prefix in [(va, 'val'), (te, 'test')]:
            p = m.predict_proba(x.loc[mask])[:, 1]
            if prefix == 'test':
                pred[f'proba_{name}'], pred[f'pred_{name}'] = p, np.where(p >= .5, 1, -1)
            else:
                val_scores[name] = evaluation.direction_metrics(np.where(p >= .5, 1, -1), y.loc[mask])['balanced_acc_nonflat']
        import joblib
        joblib.dump(m, out / f'vix_{name}.joblib', compress=3)
        print(f'VIX {name}: fitted on {train.sum():,}; test {te.sum():,}', flush=True)
    statuses = []
    for size in args.timestral_models:
        from timestral import TimestralForecaster
        model_id = f'mistralai/Timestral-{size}'
        started = time.perf_counter()
        used_cache = None
        try:
            try:
                fc = TimestralForecaster.from_pretrained(model_id, device='cpu', local_files_only=True)
            except (OSError, ValueError):
                used_cache = ROOT / 'data_raw/model_cache'
                for attempt in range(3 if args.allow_download else 1):
                    try:
                        if args.allow_download:
                            # A reset prevents a broken pooled HTTP connection carrying into the next model.
                            from huggingface_hub.utils import close_session
                            close_session()
                        fc = TimestralForecaster.from_pretrained(model_id, device='cpu',
                             cache_dir=used_cache, local_files_only=not args.allow_download)
                        break
                    except (OSError, ValueError):
                        if not args.allow_download or attempt == 2:
                            raise
                        print(f'{model_id}: download failed; retry {attempt + 2}/3', flush=True)
                        time.sleep(2 ** attempt)
            vals = {}
            for mask, split in [(va, 'validation'), (te, 'test')]:
                ctx = contexts[np.asarray(mask)]
                chunks = []
                for k in range(0, len(ctx), args.batch):
                    c = ctx[k:k + args.batch]
                    med = fc.forecast(c, prediction_length=1).median[:, 0, 0].cpu().numpy()
                    if not np.isfinite(med).all():
                        raise ValueError('nonfinite Timestral output')
                    chunks.append(np.where(med > c[:, -1].astype(np.float32), 1, -1))
                vals[split] = np.concatenate(chunks)
            pred[f'pred_timestral_{size}'] = vals['test']
            val_scores[f'timestral_{size}'] = evaluation.direction_metrics(vals['validation'], y.loc[va])['balanced_acc_nonflat']
            from huggingface_hub import try_to_load_from_cache
            config_path = try_to_load_from_cache(model_id, 'config.json', cache_dir=used_cache)
            statuses.append(dict(model=model_id, status='completed',
                                 seconds=round(time.perf_counter() - started, 2),
                                 resolved_config=str(config_path)))
            pred.to_parquet(out / 'vix_predictions.parquet', index=False)
            pd.DataFrame(statuses).to_csv(out / 'timestral_status.csv', index=False)
            print(f'VIX {model_id}: completed', flush=True)
            del fc
            gc.collect()
        except (OSError, ValueError) as exc:
            statuses.append(dict(model=model_id, status='not completed', error=str(exc)))
            print(f'{model_id}: unavailable or failed; saved explicitly as incomplete: {exc}', flush=True)
    local_names = ['logreg', 'xgboost']
    selected = max(local_names, key=val_scores.get)
    pred.to_parquet(out / 'vix_predictions.parquet', index=False)
    metrics, paired, mc = [], [], []
    ytest = pred.y3.to_numpy()
    base = pred.pred_majority.to_numpy() == ytest
    for col in [c for c in pred if c.startswith('pred_')]:
        name = col.removeprefix('pred_')
        proba = pred.get(f'proba_{name}')
        metrics.append(dict(model=name, selected_local_on_validation=name == selected,
                            validation_balanced_accuracy=val_scores[name],
                            **evaluation.direction_metrics(pred[col], ytest, proba)))
        for reference in ('majority', 'train_median_reversion'):
            ref = pred[f'pred_{reference}'].to_numpy() == ytest
            paired.append(dict(model=name, reference=reference,
                               **evaluation.paired_difference(pred[col].to_numpy() == ytest, ref, pred.date)))
        mc.extend(monte_carlo_rows(ytest, pred[col], 'VIX nearest-expiry futures', 'daily', name,
                                   args.simulations, args.seed))
    pd.DataFrame(metrics).to_csv(out / 'vix_metrics.csv', index=False)
    pd.DataFrame(paired).to_csv(out / 'vix_paired_comparisons.csv', index=False)
    pd.DataFrame(statuses).to_csv(out / 'timestral_status.csv', index=False)
    split_report = {k: dict(n=int(mask.sum()), first=str(meta.index[mask].min().date()),
                            last=str(meta.index[mask].max().date())) for k, mask in masks.items()}
    return mc, pd.DataFrame(metrics), pd.DataFrame(paired), statuses, split_report


def charts(mc, metrics, paired, args, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for ax, dataset, freq, title in [
        (axes[0], 'NQ/A_holdout_close', '1min', 'NQ one-minute: random-policy distribution'),
        (axes[1], 'VIX nearest-expiry futures', 'daily', 'VIX nearest expiry: random-policy distribution')]:
        rows = mc[(mc.dataset == dataset) & (mc.frequency == freq) & (mc['null'] == 'fair_coin')]
        if rows.empty:
            ax.set_visible(False)
            continue
        # An independent sample of the same exact binomial accuracy law for visualization.
        r = rows.iloc[0]
        rng = np.random.default_rng(args.seed)
        samples = rng.binomial(int(r.n_nonflat), .5, size=args.simulations) / r.n * 100
        ax.hist(samples, bins=45, color='#b8c5d6', edgecolor='white', label=f'{args.simulations:,} fair-coin runs')
        colors = dict(majority='#333333', logreg='#167d8d', xgboost='#d57628', timestral_209M='#7f4ca5')
        for model, color in colors.items():
            rr = rows[rows.model == model]
            if not rr.empty:
                val = float(rr.iloc[0].model_accuracy * 100)
                ax.axvline(val, color=color, linewidth=2, linestyle='--' if model == 'majority' else '-',
                           label=f'{model}: {val:.2f}%')
        ax.set(title=title, xlabel='Accuracy on all eligible outcomes (%)', ylabel='Random runs')
        ax.legend(fontsize=8, loc='upper left')
    fig.suptitle('Coin-flip Monte Carlo on fixed historical outcomes', fontsize=13)
    fig.savefig(out / 'coinflip_comparison.png', dpi=180)
    plt.close(fig)
    rows = paired[(paired.reference == 'majority') & (paired.model != 'majority')].copy()
    fig, ax = plt.subplots(figsize=(10, max(4, .48 * len(rows) + 1.8)), constrained_layout=True)
    yy = np.arange(len(rows))
    ax.errorbar(rows.diff_pp, yy, xerr=np.stack([np.maximum(0, rows.diff_pp - rows.ci_lo_pp_b20),
                                               np.maximum(0, rows.ci_hi_pp_b20 - rows.diff_pp)]),
                fmt='o', color='#167d8d', capsize=4)
    ax.axvline(0, color='#555555', linewidth=1)
    ax.set_yticks(yy, rows.model)
    ax.invert_yaxis()
    ax.set(title='VIX nearest-expiry direction: improvement over training majority',
           xlabel='Accuracy difference (percentage points), 95% interval with 20-day blocks')
    fig.savefig(out / 'vix_model_comparison.png', dpi=180)
    plt.close(fig)


def write_summary(out, mc, metrics, paired, status, reports, args):
    lines = ['# Coin-flip Monte Carlo and nearest-expiry VIX futures', '',
             'Exploratory research extension. No source CSV or previous run was modified.', '',
             f'Random-policy trials: {args.simulations:,}. Target: next observed daily price change in the',
             'nearest unexpired contract, with front-contract changes excluded. All models use the same',
             f'test origins and {args.context} observations of the SAME expiry as historical context.',
             'Other maturities are not separate predictors. The spot file is audited as a reference only.', '',
             '## VIX directional results', '',
             '| Model | Accuracy, all outcomes | Balanced accuracy, non-flat | vs majority, pp | 20-day interval, pp |',
             '|---|---:|---:|---:|---|']
    for _, r in metrics.iterrows():
        q = paired[(paired.model == r.model) & (paired.reference == 'majority')].iloc[0]
        lines.append(f'| {r.model} | {100*r.acc_all:.2f}% | {100*r.balanced_acc_nonflat:.2f}% | '
                     f'{q.diff_pp:+.2f} | [{q.ci_lo_pp_b20:+.2f}, {q.ci_hi_pp_b20:+.2f}] |')
    selected = metrics.loc[metrics.selected_local_on_validation, 'model'].iloc[0]
    lines += ['', f'Local model selected on validation only: **{selected}**. No hyperparameter search was performed.',
              '', '## Reading the random-decision comparison', '',
              '- Fair coin: independently UP or DOWN with probability 0.5.',
              '- Matched coin: the same expected UP share as the evaluated policy. This is a conditional diagnostic, not a fitted live strategy.',
              '- Flat outcomes count as misses for both the models and coins. Expected fair-coin accuracy is therefore half the non-flat coverage, not automatically 50%.',
              '- The simulations condition on the observed outcomes. Their 95% bands describe luck from random decisions on this history, not uncertainty over future market paths.',
              '- A model can beat a fair coin while failing to beat the training-majority rule. Use both comparisons.',
              '- conditional_tail_fraction and its Holm adjustment are conditional random-policy diagnostics, not proof of market alpha. Minimum resolution is 1/(simulations+1).',
              '- Session/day-block bootstrap comparisons against majority and a simple training-median reversion rule are saved separately.', '',
              '## Trading interpretation', '',
              'No VIX dollar backtest is produced: these files contain index OHLC and futures daily price observations,',
              'without verified price publication times, executable quotes, or account costs. Spot VIX is a calculated',
              '30-day implied-volatility index and cannot itself be bought. Nearest-expiry VIX futures price direction',
              'is a different target from spot VIX direction, S&P 500 direction, or realized S&P 500 volatility.', '',
              'NQ random-sign profit diagnostics are restricted to saved rows whose assumed entry is no earlier than',
              'the recorded decision. Intraday entries failing that check are explicitly skipped. Daily/session checks',
              'use exactly each model\'s active timestamps, recorded per-fold thresholds, roll exclusions and the same',
              'zero/base cost assumptions. They remain illustrative bar-price simulations, not verified fills.', '',
              '## Data and availability limits', '']
    for name, report in reports.items():
        lines.append(f'- {name}: `{json.dumps(report, default=str)}`')
    lines += ['', '- Exact duplicate full rows are removed before joins. Conflicting duplicate dates cause a failure.',
              '- The chosen expiry is determined using that day\'s vendor expiry fields, not future prices.',
              '- Roll transitions are excluded; the vendor expiry-date convention has not been validated against an exchange contract master.',
              '- Data after the end-date cutoff is not used for fitting, validation, contexts, or target outcomes.',
              '- Daily publication timestamps are absent. Interpret forecasts as made after the reported daily values become available; no intraday timing claim is made.',
              '- Comparing different instruments does not establish that one is intrinsically easier: eligibility, class balance and sample size differ.', '',
              '## Timestral completion', '']
    lines += [f'- {s["model"]}: {s["status"]}' for s in status]
    lines += ['', '## Presentation scope', '',
              'Use one slide for the coin-versus-model chart and one for nearest-expiry VIX results.',
              'Replace two detailed NQ result slides rather than exceeding ten slides.', '',
              'Sources: https://www.cboe.com/tradable_products/vix/faqs and supplied local files.', '']
    (out / 'SUMMARY.md').write_text('\n'.join(lines), encoding='utf-8')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spot', type=Path, default=VIX_DIR / 'spot_LSEG_VIX.csv')
    p.add_argument('--futures', type=Path, default=VIX_DIR / 'futures_LSEG_VIX.csv')
    p.add_argument('--nq-runs', nargs='*', type=Path, default=DEFAULT_NQ_RUNS)
    p.add_argument('--simulations', type=int, default=10000)
    p.add_argument('--context', type=int, default=64)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val-start', default='2017-08-25')
    p.add_argument('--test-start', default='2021-07-15')
    p.add_argument('--end-date', default='2025-05-30')
    p.add_argument('--timestral-models', nargs='*', choices=TIMESTRAL_SIZES, default=[])
    p.add_argument('--allow-download', action='store_true', help='download missing Timestral weights to data_raw/model_cache')
    p.add_argument('--reuse-nq-monte-carlo', type=Path,
                   help='reuse the NQ Monte Carlo CSVs from a previous extension run')
    a = p.parse_args(argv)
    if a.simulations < 100 or a.context < 20 or a.batch < 1:
        p.error('need simulations >= 100, context >= 20 and positive batch')
    if not pd.Timestamp(a.val_start) < pd.Timestamp(a.test_start) < pd.Timestamp(a.end_date):
        p.error('need validation < test < end-date')
    return a


def main(argv=None):
    args = parse_args(argv)
    out = runlog.new_run_dir('results', 'extensions', 'coinflip_vix_front')
    print(f'Run folder: {out}', flush=True)
    spot, sr = load_daily(args.spot, 'Date')
    fut, fr = load_daily(args.futures, 'Data')
    x, y, meta, contexts, dr = build_front_dataset(fut, args.context, args.end_date)
    if args.reuse_nq_monte_carlo:
        previous_args = json.loads((args.reuse_nq_monte_carlo / 'run.json').read_text())['args']
        if previous_args['seed'] != args.seed:
            raise ValueError('reused random seed differs from this run')
        previous = pd.read_csv(args.reuse_nq_monte_carlo / 'coinflip_accuracy.csv')
        previous = previous[previous.dataset.str.startswith('NQ/')]
        if not previous.empty and not previous.simulations.eq(args.simulations).all():
            raise ValueError('reused simulations count differs from this run')
        rows = previous.drop(columns=['conditional_tail_holm'], errors='ignore').to_dict('records')
        pnl = pd.read_csv(args.reuse_nq_monte_carlo / 'coinflip_pnl.csv').to_dict('records')
        print('Reused existing NQ random-policy comparisons; no NQ retraining or resimulation', flush=True)
    else:
        rows, pnl = nq_comparisons(args.nq_runs, args)
    vr, metrics, paired, status, splits = vix_models(x, y, meta, contexts, args, out)
    mc = pd.DataFrame(rows + vr)
    mc['conditional_tail_holm'] = holm_adjust(mc.conditional_tail_fraction)
    mc.to_csv(out / 'coinflip_accuracy.csv', index=False)
    pd.DataFrame(pnl).to_csv(out / 'coinflip_pnl.csv', index=False)
    reports = dict(spot=sr, futures=fr, front_dataset=dr, splits=splits)
    write_summary(out, mc, metrics, paired, status, reports, args)
    charts(mc, metrics, paired, args, out)
    runlog.write_run_json(out, dict(args=vars(args), data_reports=reports, timestral=status,
                          target='same nearest-expiry futures next observed daily quoted price direction; roll changes excluded',
                          status='complete' if all(s['status'] == 'completed' for s in status) else 'incomplete models',
                          coin_null='conditional independent decisions on fixed historical outcomes, not future-market inference'))
    print(metrics[['model', 'acc_all', 'balanced_acc_nonflat']].to_string(index=False), flush=True)
    print(f'Saved to {out}', flush=True)
    return out


if __name__ == '__main__':
    main()
