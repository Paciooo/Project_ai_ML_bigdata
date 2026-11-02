"""Repair economic scoring and verify local evidence without refitting classifiers.

Requires the original local Parquet history, saved fitted models and run folders.
Preserves pre-repair files under each run's audit_backups/<repair id>/, then
updates execution metadata, fold thresholds and small summary files. Source
labels, prediction probabilities and classifier outputs must remain identical.
Raw inputs and model weights are never modified. Small verified snapshots go
to reports/ for remote reviewers; large data/weights remain local.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
import backtest
import config
import evaluation
import experiment
import features
import plots
import runlog
from coinflip_vix import (DEFAULT_NQ_RUNS, VIX_DIR, build_front_dataset, chronological_masks,
                          load_daily, nq_comparisons)

TIMESTRAL_NQ = ROOT / 'results/nq/runs/20261102T061421Z_ef43c1_timestral_Timestral-5M'
VIX_RUN = ROOT / 'results/extensions/runs/20261102T065506Z_d1d861_coinflip_vix_front'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def capture(path, backup, record):
    path = Path(path)
    if path.exists():
        dest = backup / path.name
        if not dest.exists():
            shutil.copy2(path, dest)
        record[path.name] = dict(before_sha256=digest(dest), backup=str(dest.relative_to(ROOT)))


def parse_optional_date(value):
    return None if pd.isna(value) or str(value) in ('None', 'NaT', '') else pd.Timestamp(value)


def restore_thresholds(preds, folds, frequency):
    """Strict row mapping; a missing fold never defaults to a zero threshold."""
    out = preds.copy()
    fs = folds[folds.frequency == frequency].set_index('fold')
    if fs.index.duplicated().any():
        raise ValueError('duplicate fold metadata')
    for col in (c for c in out if c.startswith('proba_')):
        model = col.removeprefix('proba_')
        key = f'threshold_{model}'
        if key not in fs:
            raise ValueError(f'missing {key}')
        out[key] = out.fold.map(fs[key]).to_numpy()
        if out[key].isna().any():
            raise ValueError(f'missing threshold for a prediction fold: {key}')
    return out


def test_probabilities(model, x, p, column):
    vals = model.predict_proba(x)[:, 1]
    if not np.allclose(vals, p[column].to_numpy(), rtol=1e-6, atol=1e-7):
        raise ValueError(f'saved model no longer reproduces {column}')


def verify_timestral_nq(freq, bars, forecaster):
    file = TIMESTRAL_NQ / f'timestral_predictions_{freq}.parquet'
    if not file.exists():
        return None
    p = pd.read_parquet(file)
    positions = pd.DatetimeIndex(bars.bar_start).get_indexer(p.bar_start)
    ctx_len = 512
    if (positions < ctx_len - 1).any():
        raise ValueError('unmatched Timestral origin')
    close = bars.close.to_numpy(float)
    values = []
    for start in range(0, len(p), 64):
        ctx = np.stack([close[j-ctx_len+1:j+1] for j in positions[start:start+64]])
        median = forecaster.forecast(ctx, prediction_length=1).median[:, 0, 0].cpu().numpy()
        values.extend(np.where(median > ctx[:, -1].astype(np.float32), 1, -1))
    if not np.array_equal(values, p.pred_timestral.to_numpy()):
        raise ValueError(f'Timestral NQ {freq}: cached model does not reproduce saved directions')
    summary = pd.read_csv(TIMESTRAL_NQ / 'timestral_results.csv')
    checks = []
    for col in (c for c in p if c.startswith('pred_')):
        name = col.removeprefix('pred_')
        key = 'mistralai/Timestral-5M' if name == 'timestral' else name
        row = summary[(summary.frequency == freq) & (summary.model == key)].iloc[0]
        measured = evaluation.direction_metrics(p[col].astype(int), p.y3)
        for field in ('acc_all', 'acc_nonflat'):
            if not np.isclose(row[field], measured[field], atol=5.1e-5):
                raise ValueError(f'Timestral summary mismatch: {freq}/{key}/{field}')
        checks.append(dict(model=key, n=len(p), acc_all=measured['acc_all']))
    return dict(frequency=freq, model_inference_reproduced=True, metrics=checks,
                predictions_sha256=digest(file))


def copy_small_report(run, name):
    dest = ROOT / 'reports' / name
    dest.mkdir(parents=True, exist_ok=True)
    for pat in ('*.csv', '*.png', 'run.json', 'rescore.json'):
        for src in run.glob(pat):
            shutil.copy2(src, dest / src.name)
    return dest


def repair_nq(args, repair_id, report):
    config.FUTURES_DATA_DIR = str(args.futures_data)
    minutes, _, data_report, _ = experiment.load_minutes('nq')
    session_opens = minutes.groupby('session_id').session_open.first()
    from timestral import TimestralForecaster
    forecaster = TimestralForecaster.from_pretrained('mistralai/Timestral-5M', local_files_only=True)
    states = []
    for path in args.runs:
        run = Path(path)
        source_record = json.loads((run / 'run.json').read_text())
        a = SimpleNamespace(**source_record['args'])
        backup = run / 'audit_backups' / repair_id
        backup.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for file in ('folds.csv', 'backtest.csv', 'metrics.csv', 'paired_vs_majority.csv',
                     'by_year.csv', 'timeframe_comparison.png'):
            capture(run / file, backup, hashes)
        folds = pd.read_csv(run / 'folds.csv')
        old_metrics = pd.read_csv(run / 'metrics.csv')
        old_paired = pd.read_csv(run / 'paired_vs_majority.csv')
        all_m, all_p, all_y, all_b, freq_checks = [], [], [], [], []
        for freq in a.frequency:
            pfile = run / f'predictions_{freq}.parquet'
            if not pfile.exists():
                continue
            print(f'Repair {run.name}: {freq}', flush=True)
            bars, eligible = experiment.bars_for(minutes, None, freq)
            x, y, meta, _ = features.build_timed_dataset(
                bars, freq, a.target, config.RETURN_LAGS, config.MOMENTUM_WINDOWS,
                config.VOLATILITY_WINDOWS, extra_groups=a.extra_features,
                decision_latency=a.latency, label_eligible=eligible)
            meta = backtest.attach_execution(meta, minutes)
            original = pd.read_parquet(pfile)
            lookup = pd.DatetimeIndex(meta.bar_start).get_indexer(original.bar_start)
            if (lookup < 0).any():
                raise ValueError(f'{freq}: a saved origin no longer matches source data')
            aligned = meta.iloc[lookup]
            if not np.array_equal(original.y3, y.iloc[lookup]):
                raise ValueError('reconstructed source labels differ from saved labels')
            fixed = original.copy()
            for col in ('label_session_id', 'entry_at', 'exit_at', 'entry_price', 'exit_price',
                        'execution_eligible', 'execution_reason'):
                fixed[col] = aligned[col].array  # preserve UTC dtype even when every entry is NaT
            model_checks = []
            for row_index, fold in folds[folds.frequency == freq].iterrows():
                v, t = pd.Timestamp(fold.val_start), pd.Timestamp(fold.test_start)
                end = parse_optional_date(fold.test_end)
                _, val, _ = features.purged_split(meta, v, t, lambda day: session_opens.loc[day],
                                                 end=end, min_rows=a.min_rows)
                cols = fold.features.split(',')
                test = fixed.fold.eq(fold.fold).to_numpy()
                for name in a.models:
                    model_file = run / 'models' / f'{freq}_{name}_fold{int(fold.fold)}.joblib'
                    model = joblib.load(model_file)
                    test_probabilities(model, x.iloc[lookup[test]][cols], fixed.loc[test], f'proba_{name}')
                    probability = model.predict_proba(x.loc[val, cols])[:, 1]
                    folds.loc[row_index, f'threshold_{name}'] = backtest.choose_threshold(
                        meta.loc[val], probability, backtest.COST_SCENARIOS['base'])
                    model_checks.append(dict(fold=int(fold.fold), model=name,
                                              model_sha256=digest(model_file), test_probabilities_reproduced=True))
                    del model
            fixed = restore_thresholds(fixed, folds, freq)
            preserved = [c for c in original if c.startswith(('pred_', 'proba_'))] + ['y3']
            if not original[preserved].equals(fixed[preserved]):
                raise ValueError('repair changed classifier predictions')
            m, paired, yearly, bt = experiment.score(fixed, freq, a.models, a)
            for _, row in m.iterrows():
                old = old_metrics[(old_metrics.frequency == freq) & (old_metrics.model == row.model)].iloc[0]
                if not np.isclose(row.acc_all, old.acc_all, atol=5.1e-6):
                    raise ValueError('classification accuracy changed')
            all_m.append(m); all_p.append(paired); all_y.append(yearly); all_b.append(bt)
            capture(pfile, backup, hashes)
            tmp = pfile.with_suffix('.repair.parquet')
            fixed.to_parquet(tmp, index=False)
            tmp.replace(pfile)
            hashes[pfile.name]['after_sha256'] = digest(pfile)
            if run == Path(args.runs[0]):
                check = verify_timestral_nq(freq, bars, forecaster)
                if check:
                    report['timestral_nq'].append(check)
            freq_checks.append(dict(frequency=freq, n=len(fixed), predictions_unchanged=True,
                                    source_labels_rebuilt=True, models=model_checks,
                                    n_execution_eligible=int(fixed.execution_eligible.sum())))
            del bars, x, y, meta, original, fixed
            gc.collect()
        M, P, Y, B = [pd.concat(rows, ignore_index=True) for rows in (all_m, all_p, all_y, all_b)]
        M.to_csv(run / 'metrics.csv', index=False, float_format='%.5f')
        P.to_csv(run / 'paired_vs_majority.csv', index=False, float_format='%.4f')
        Y.to_csv(run / 'by_year.csv', index=False, float_format='%.5f')
        B.to_csv(run / 'backtest.csv', index=False, float_format='%.6f')
        folds.to_csv(run / 'folds.csv', index=False)
        plots.plot_paired_differences(P, B, str(run))
        provenance = dict(repair_id=repair_id, corrected_at=datetime.now(timezone.utc).isoformat(),
                          original_run_json_unchanged=True, original_run_json_sha256=digest(run / 'run.json'),
                          original_model_fit_preserved=True, classifier_predictions_unchanged=True,
                          validation_thresholds_reselected_using_saved_models=True,
                          economic_policy='first clean observed minute open >= decision, inside target interval',
                          label='exploratory corrected execution simulation; bar-price proxies, assumed costs',
                          checks=freq_checks, artifacts=hashes,
                          repair_source_hashes=runlog.source_hashes())
        (run / 'rescore.json').write_text(json.dumps(provenance, indent=2, default=str), encoding='utf-8')
        name = {'A_holdout_close':'local_run_A_holdout_close', 'B_holdout_body':'local_run_B_holdout_body',
                'C_walkforward':'local_run_C_walkforward'}[run.name.split('_', 2)[-1]]
        dest = copy_small_report(run, name)
        states.append(dict(name=name, local_folder=str(run), github_report=str(dest.relative_to(ROOT)), **provenance))
        print(f'Updated {dest}', flush=True)
    copy_small_report(TIMESTRAL_NQ, 'local_timestral_NQ_5M')
    report['nq'] = states
    report['nq_source'] = dict(folder=str(args.futures_data), **data_report)
    del minutes, forecaster
    gc.collect()


def verify_vix(report):
    a = json.loads((VIX_RUN / 'run.json').read_text())['args']
    _, spot_audit = load_daily(Path(a['spot']), 'Date')
    f, futures_audit = load_daily(Path(a['futures']), 'Data')
    x, y, meta, context, info = build_front_dataset(f, a['context'], a['end_date'])
    masks = chronological_masks(meta, a['val_start'], a['test_start'])
    p = pd.read_parquet(VIX_RUN / 'vix_predictions.parquet')
    take = pd.DatetimeIndex(meta.date).get_indexer(p.date)
    if (take < 0).any() or not np.array_equal(y.iloc[take], p.y3):
        raise ValueError('VIX source reconstruction does not match labels')
    if not np.array_equal(meta.expiry.iloc[take], p.expiry):
        raise ValueError('VIX nearest-expiry selection mismatch')
    model_checks = []
    for model in ('logreg', 'xgboost'):
        saved = VIX_RUN / f'vix_{model}.joblib'
        m = joblib.load(saved)
        test_probabilities(m, x.iloc[take], p, f'proba_{model}')
        model_checks.append(dict(model=model, model_sha256=digest(saved), predictions_reproduced=True))
    from timestral import TimestralForecaster
    statuses = pd.read_csv(VIX_RUN / 'timestral_status.csv')
    for _, row in statuses.iterrows():
        checkpoint = Path(row.resolved_config).parent
        weights = checkpoint / 'model.safetensors'
        fc = TimestralForecaster.from_pretrained(checkpoint, local_files_only=True)
        predictions = []
        ctx_all = context[take]
        for start in range(0, len(p), 32):
            ctx = ctx_all[start:start+32]
            median = fc.forecast(ctx, prediction_length=1).median[:, 0, 0].cpu().numpy()
            predictions.extend(np.where(median > ctx[:, -1].astype(np.float32), 1, -1))
        key = 'timestral_' + row.model.split('-')[-1]
        if not np.array_equal(predictions, p[f'pred_{key}']):
            raise ValueError(f'VIX {key}: checkpoint no longer reproduces directions')
        model_checks.append(dict(model=row.model, weights_bytes=weights.stat().st_size,
                                 weights_sha256=digest(weights), config_sha256=digest(checkpoint / 'config.json'),
                                 predictions_reproduced=True))
        del fc
        gc.collect()
        print(f'Verified raw VIX labels and cached {key}', flush=True)
    saved_metrics = pd.read_csv(VIX_RUN / 'vix_metrics.csv')
    for _, row in saved_metrics.iterrows():
        measured = evaluation.direction_metrics(p[f'pred_{row.model}'], p.y3)
        if not np.isclose(row.acc_all, measured['acc_all'], atol=1e-12):
            raise ValueError('VIX summary not reproduced')
    report['vix'] = dict(spot=spot_audit, futures=futures_audit, front_dataset=info,
                          n_test=len(p), source_labels_rebuilt=True, models=model_checks,
                          predictions_sha256=digest(VIX_RUN / 'vix_predictions.parquet'))
    # Public audit evidence contains outcomes/predictions and identity/timing, not licensed price histories.
    cols = [c for c in p if c in ('date', 'label_date', 'expiry', 'y3') or c.startswith(('pred_', 'proba_'))]
    p[cols].to_csv(ROOT / 'reports/coinflip_vix_front/prediction_audit.csv', index=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--futures-data', type=Path, default=Path(r'C:\Users\aless\Desktop\py\futures-data'))
    p.add_argument('--runs', nargs='+', type=Path, default=DEFAULT_NQ_RUNS)
    args = p.parse_args(argv)
    repair_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    report = dict(repair_id=repair_id, classifier_retraining=False, timestral_nq=[])
    repair_nq(args, repair_id, report)
    verify_vix(report)
    # Refresh only economics. The already verified Monte Carlo classification draws stay unchanged.
    sim_args = SimpleNamespace(simulations=10000, seed=42)
    _, pnl = nq_comparisons(args.runs, sim_args)
    pd.DataFrame(pnl).to_csv(VIX_RUN / 'coinflip_pnl.csv', index=False)
    shutil.copy2(VIX_RUN / 'coinflip_pnl.csv', ROOT / 'reports/coinflip_vix_front/coinflip_pnl.csv')
    dest = ROOT / 'reports/verification'
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'local_verification.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    print(f'Complete: {dest / "local_verification.json"}', flush=True)


if __name__ == '__main__':
    main()
