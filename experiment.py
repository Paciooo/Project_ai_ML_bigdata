"""
Timed bar experiment: next-candle direction on several time frames.

Pipeline for each frequency:
  1. 1-minute bars with explicit timing and CME sessions (timeframes.py),
     causal quarantine of suspect prints (quality.py);
  2. session-anchored candles (1min, 5min, 15min, 1h, 4h, session) or raw
     trades ("tick", a separate task: next_trade_direction);
  3. features, three-class outcome (flats kept), eligibility, timing;
  4. chronological folds (holdout or walk-forward) on whole sessions, with
     labels purged at each boundary; models fitted on training rows only,
     no-trade threshold chosen on validation only;
  5. per-origin predictions, classification metrics (non-flat and all
     outcomes), paired block-bootstrap vs the majority baseline, results by
     year, and an offline cost-aware backtest.
Every run gets its own folder; nothing is overwritten.
"""

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import backtest
import config
import data
import evaluation
import features
import quality
import regimes
import runlog
import timeframes

RULES = ["majority", "persistence", "reversal"]
MODELS = ["logreg", "xgboost", "random_forest", "mlp1", "mlp3", "mlp_ens"]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class MLPEnsemble:
    """Average of a few small MLPs with different seeds."""

    def __init__(self, seed, n=3):
        self.members = [make_model("mlp1", seed + i) for i in range(n)]

    def fit(self, X, y):
        for m in self.members:
            m.fit(X, y)
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.members], axis=0)


def make_model(name, seed):
    if name == "logreg":
        m = LogisticRegression(**config.LOGREG_PARAMS)
    elif name == "xgboost":
        from xgboost import XGBClassifier
        m = XGBClassifier(random_state=seed, tree_method="hist", **config.XGB_PARAMS)
    elif name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(random_state=seed, **config.FOREST_PARAMS)
    elif name == "mlp1":
        m = MLPClassifier(hidden_layer_sizes=config.MLP_SHALLOW_LAYERS, random_state=seed, **config.MLP_COMMON)
    elif name == "mlp3":
        m = MLPClassifier(hidden_layer_sizes=config.MLP_DEEP_LAYERS, random_state=seed, **config.MLP_COMMON)
    elif name == "mlp_ens":
        return MLPEnsemble(seed)
    else:
        raise ValueError(f"unknown model {name!r}; choose from {MODELS}")
    return make_pipeline(StandardScaler(), m)


def fit_quiet(m, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return m.fit(X, y)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_minutes(source, years=None, spike_policy="quarantine", smoke=False, fixture_signal=0.5):
    """Canonical 1-minute bars with sessions + suspect flags, and a report.

    Returns (minutes, ticks or None, report, contract metadata).
    """
    rep = {"source": source, "spike_policy": spike_policy}
    ticks = None
    if source == "fixture":
        raw = data.fixture_minutes(n_sessions=40 if smoke else 120, seed=0, bounce=fixture_signal)
        bars = raw
        contract = {"series": f"synthetic fixture (bounce={fixture_signal})", "tradable": False}
    elif source == "nq":
        y0, y1 = years or config.FUTURES_YEARS
        d = config.futures_data_dir()
        raw = data.load_futures_minutes(d, config.FUTURES_SYMBOL, y0, y1).reset_index()
        bars, r = timeframes.from_tradestation_minutes(raw)
        rep.update(r, futures_data_dir=str(d), years=[y0, y1])
        contract = {"series": "TradeStation continuous NQ, additive back-adjusted",
                    "tradable_prices": False, "roll_dates": "not in data (estimated, see backtest.py)",
                    "timestamps": "Europe/Rome wall clock, bar END"}
    elif source == "nq_ticks":
        d = config.ticks_dir()
        ticks = data.load_tick_files(d).reset_index()
        rep["ticks"] = quality.validate_ticks(ticks)
        rep["manifest"] = data.tick_manifest_status(d)
        bars = timeframes.from_ib_ticks(ticks)
        contract = {"series": "IBKR single contract trades", "tradable_prices": True,
                    "contract": rep["manifest"].get("contract"), "timestamps": "UTC event time"}
    else:
        raise ValueError(f"unknown data source {source!r}")

    if spike_policy == "legacy_remove":     # future-dependent, sensitivity only
        bars, n = quality.legacy_remove_reverting_spikes(bars, config.SPIKE_THRESHOLD)
        rep["legacy_spikes_removed"] = n
        bars["suspect"] = False
    elif spike_policy == "quarantine":
        bars["suspect"] = quality.flag_suspect(bars, config.SPIKE_THRESHOLD)
    elif spike_policy == "none":
        bars["suspect"] = False
    else:
        raise ValueError(f"unknown spike policy {spike_policy!r}")
    rep["n_suspect_minutes"] = int(bars["suspect"].sum())
    rep["validation"] = quality.validate_bars(bars, years if source == "nq" else None)
    minutes, r = timeframes.add_sessions(bars)
    rep.update(r)
    return minutes, ticks, rep, contract


def tick_rows(ticks):
    """Raw trades as rows (task next_trade_direction). Times have 1-second
    resolution, so a trade stamped hh:mm:ss is known by hh:mm:ss + 1 s."""
    ts = pd.DatetimeIndex(ticks["ts"])
    ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
    p = ticks["price"].to_numpy(float)
    rows = pd.DataFrame({"bar_start": ts, "bar_end": ts + pd.Timedelta(seconds=1),
                         "open": p, "high": p, "low": p, "close": p,
                         "volume": ticks["size"].to_numpy()})
    rows["available_at"] = rows["bar_end"]
    cal = timeframes.session_calendar(rows["bar_start"])
    rows = pd.concat([rows, cal], axis=1)
    rows = rows[rows["in_schedule"]].drop(columns="in_schedule").reset_index(drop=True)
    rows["is_partial"], rows["suspect"] = False, False
    same = (rows["session_id"].shift(-1) == rows["session_id"]).to_numpy()
    return rows, same


def bars_for(minutes, ticks, freq):
    if freq == "tick":
        if ticks is None:
            raise ValueError("frequency 'tick' needs trade data (--data-source nq_ticks)")
        return tick_rows(ticks)
    bars = timeframes.aggregate(minutes, freq)
    opens = minutes.groupby("session_id")["session_open"].first()
    bars["session_open"] = bars["session_id"].map(opens)
    return bars, timeframes.next_label_eligible(bars, freq)


# ---------------------------------------------------------------------------
# One frequency
# ---------------------------------------------------------------------------

def run_frequency(minutes, ticks, freq, args, folds, session_open_of, holdout_start, log):
    target = "next_trade_direction" if freq == "tick" else args.target
    bars, elig = bars_for(minutes, ticks, freq)
    lags = list(range(1, 6)) if args.regimes else config.RETURN_LAGS
    X, y3, meta, info = features.build_timed_dataset(
        bars, freq, target, lags, config.MOMENTUM_WINDOWS, config.VOLATILITY_WINDOWS,
        extra_groups=args.extra_features, decision_latency=args.latency, label_eligible=elig)
    if args.context_features and freq not in ("session", "tick"):
        sess = timeframes.aggregate(minutes, "session")
        X = features.add_context_features(meta, X, sess, "sess")
        ok = X.notna().all(axis=1)
        X, y3, meta = X[ok], y3[ok], meta[ok]
    if args.regimes:
        sessions = timeframes.aggregate(minutes, "session")
    log(f"[{freq}] {info['n_rows']:,} eligible rows of {info['n_bars']:,} bars "
        f"(flat outcomes {info['n_flat']:,}, ineligible labels {info['n_label_ineligible']:,}, "
        f"quarantine {info['n_quarantine_window']:,})")

    costs = backtest.COST_SCENARIOS["base"]
    pred_frames, fold_log = [], []
    for k, (first, val_start, test_start, test_end) in enumerate(folds):
        end = test_end if test_end is not None else holdout_start
        tr, va, te = features.purged_split(meta, val_start, test_start, session_open_of, end=end,
                                           min_rows=args.min_rows)
        tr &= (meta["session_id"] >= first).to_numpy()
        tr_nf = tr & (y3.to_numpy() != 0)
        idx_tr = np.flatnonzero(tr_nf)
        if args.max_train_rows and len(idx_tr) > args.max_train_rows:
            idx_tr = idx_tr[-args.max_train_rows:]           # most recent training rows
        Xtr, ytr = X.iloc[idx_tr], (y3.iloc[idx_tr] > 0).astype(int)
        majority = 1 if ytr.mean() >= 0.5 else -1
        cols = list(X.columns)
        if args.regimes:
            cols = select_lags(X, y3, tr_nf, va, args.seed)
        out = meta[te].copy()
        out["fold"], out["y3"] = k, y3[te].to_numpy()
        thresholds = {}
        for rule in RULES:
            out[f"pred_{rule}"] = evaluation.rule_predictions(X[te], rule, majority)
        for name in args.models:
            t0 = time.time()
            m = fit_quiet(make_model(name, args.seed), Xtr[cols], ytr)
            pv, pt = m.predict_proba(X[va][cols])[:, 1], m.predict_proba(X[te][cols])[:, 1]
            thresholds[name] = backtest.choose_threshold(meta[va], pv, costs)
            if getattr(args, "out_dir", None) is not None and name != "random_forest":
                import joblib   # fitted scaler + model, one file per fold (forests skipped: large)
                (args.out_dir / "models").mkdir(exist_ok=True)
                joblib.dump(m, args.out_dir / "models" / f"{freq}_{name}_fold{k}.joblib", compress=3)
            out[f"proba_{name}"], out[f"pred_{name}"] = pt, np.where(pt >= 0.5, 1, -1)
            log(f"  [{freq}] fold {k} {name}: fitted on {len(Xtr):,} rows in {time.time() - t0:.0f} s; "
                f"validation threshold {thresholds[name]}")
        if args.regimes:
            regime_models(out, X, y3, meta, sessions, tr_nf, va, te, idx_tr, cols, val_start,
                          args, costs, thresholds, freq, k, log, fold_log)
        out.attrs["thresholds"] = thresholds
        fold_log.append({"frequency": freq, "fold": k, "train_first": first, "val_start": val_start,
                         "test_start": test_start, "test_end": end, "n_train_fit": len(Xtr),
                         "n_val": int(va.sum()), "n_test": int(te.sum()), "majority": majority,
                         "features": ",".join(cols),
                         **{f"threshold_{n}": v for n, v in thresholds.items()}})
        pred_frames.append(out)
    preds = pd.concat(pred_frames)
    return preds, info, fold_log


def select_lags(X, y3, tr_nf, va, seed):
    """Paper-inspired lag count: 3 vs 5 lagged returns, chosen on validation."""
    base = [c for c in X.columns if not c.startswith("ret_lag")]
    best, best_acc = None, -1
    yv = y3[va].to_numpy()
    for n in (3, 5):
        cols = [f"ret_lag{i}" for i in range(1, n + 1)] + base
        m = fit_quiet(make_model("logreg", seed), X[tr_nf][cols], (y3[tr_nf] > 0).astype(int))
        p = np.where(m.predict_proba(X[va][cols])[:, 1] >= 0.5, 1, -1)
        acc = (p[yv != 0] == yv[yv != 0]).mean()
        if acc > best_acc:
            best, best_acc = cols, acc
    return best


def regime_models(out, X, y3, meta, sessions, tr_nf, va, te, idx_tr, cols, val_start, args, costs,
                  thresholds, freq, k, log, fold_log):
    """Pooled vs regime-conditioned models on identical rows (adds columns to out)."""
    s_train = sessions[sessions["session_id"] < val_start]
    yv = y3[va].to_numpy()
    best = None
    for n_cp in (1, 3):
        for min_size in (20, 60):
            router = regimes.RegimeRouter(n_cp, min_size, seed=args.seed).fit(s_train)
            g_pred = router.predict_sessions(sessions)
            g_tr = meta["session_id"].iloc[idx_tr].map(router.train_group).fillna(-1).astype(int).to_numpy()
            rm = regimes.RegimeModel(lambda: make_model("logreg", args.seed), args.regime_min_rows)
            rm.fit(X.iloc[idx_tr][cols].to_numpy(), (y3.iloc[idx_tr] > 0).astype(int).to_numpy(), g_tr)
            gv = meta[va]["session_id"].map(g_pred).fillna(-1).astype(int).to_numpy()
            p = np.where(rm.predict_proba_up(X[va][cols].to_numpy(), gv) >= 0.5, 1, -1)
            acc = (p[yv != 0] == yv[yv != 0]).mean()
            if best is None or acc > best[0]:
                best = (acc, n_cp, min_size, router, g_pred, g_tr)
    _, n_cp, min_size, router, g_pred, g_tr = best
    log(f"  [{freq}] fold {k} regimes: {n_cp} change-points, min {min_size} sessions "
        f"(chosen on validation); change sessions {[str(c.date()) for c in router.change_sessions]}")
    gv = meta[va]["session_id"].map(g_pred).fillna(-1).astype(int).to_numpy()
    gt = meta[te]["session_id"].map(g_pred).fillna(-1).astype(int).to_numpy()
    out["regime"] = gt
    Xtr, ytr = X.iloc[idx_tr][cols].to_numpy(), (y3.iloc[idx_tr] > 0).astype(int).to_numpy()
    for name in ("logreg", "xgboost", "mlp_ens"):
        rm = regimes.RegimeModel(lambda n=name: make_model(n, args.seed), args.regime_min_rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            rm.fit(Xtr, ytr, g_tr)
        pv = rm.predict_proba_up(X[va][cols].to_numpy(), gv)
        pt = rm.predict_proba_up(X[te][cols].to_numpy(), gt)
        key = f"regime_{name}"
        thresholds[key] = backtest.choose_threshold(meta[va], pv, costs)
        out[f"proba_{key}"], out[f"pred_{key}"] = pt, np.where(pt >= 0.5, 1, -1)
        log(f"  [{freq}] fold {k} {key}: {len(rm.per)} group models + pooled fallback")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score(preds, freq, models, args):
    """Metrics, paired bootstrap, per-year results and backtests for one frequency."""
    y3, sess = preds["y3"].to_numpy(), preds["session_id"].to_numpy()
    names = RULES + [c[5:] for c in preds.columns if c.startswith("pred_") and c[5:] not in RULES]
    base_correct = preds["pred_majority"].to_numpy() == y3
    metrics, paired, by_year, bt = [], [], [], []
    rolls = backtest.approx_roll_sessions(preds["session_id"]) if freq == "session" else None
    exclude = preds["session_id"].isin(rolls).to_numpy() if rolls is not None else None
    for name in names:
        pred = preds[f"pred_{name}"].to_numpy()
        proba = preds[f"proba_{name}"].to_numpy() if f"proba_{name}" in preds else None
        m = {"frequency": freq, "model": name, **evaluation.direction_metrics(pred, y3, proba)}
        metrics.append(m)
        correct = pred == y3
        if name != "majority":
            paired.append({"frequency": freq, "model": name, "metric": "acc_all_vs_majority",
                           **evaluation.paired_difference(correct, base_correct, sess)})
        yr = pd.DatetimeIndex(preds["session_id"]).year
        for y, g in pd.DataFrame({"y": yr, "c": correct, "b": base_correct, "nf": y3 != 0}).groupby("y"):
            by_year.append({"frequency": freq, "model": name, "year": y, "n": len(g),
                            "acc_all": g["c"].mean(), "acc_all_majority": g["b"].mean(),
                            "acc_nonflat": g.loc[g["nf"], "c"].mean()})
        for scen, costs in backtest.COST_SCENARIOS.items():
            th = preds.attrs.get("thresholds", {}).get(name, 0.0) if proba is not None else 0.0
            pos = backtest.positions(proba, th) if proba is not None else pred
            trades = backtest.run_backtest(preds, pos, costs, exclude=exclude)
            s = backtest.summarize(trades, capital_usd=config.CAPITAL_USD)
            est, lo, hi = evaluation.block_bootstrap(trades["net_usd"], trades["session_id"],
                                                     block_len=5, stat="sum_per_group")
            bt.append({"frequency": freq, "model": name, "costs": scen, "threshold": th, **s,
                       "mean_daily_net_ci_lo_b5": lo, "mean_daily_net_ci_hi_b5": hi})
    bt.append({"frequency": freq, "model": "no_trade", "costs": "any", "n_trades": 0, "net_usd": 0.0})
    return (pd.DataFrame(metrics), pd.DataFrame(paired), pd.DataFrame(by_year), pd.DataFrame(bt))


def run(args, log=print):
    source = "fixture" if args.smoke and args.data_source != "fixture" and args.smoke_fixture else args.data_source
    minutes, ticks, rep, contract = load_minutes(source, args.years, args.spike_policy, args.smoke,
                                                 args.fixture_signal)
    out_dir = runlog.new_run_dir(config.RESULTS_DIR, source if not args.smoke else "smoke",
                                 tag=args.tag or ("smoke" if args.smoke else ""))
    log(f"run folder: {out_dir}")
    args.out_dir = out_dir
    sessions = pd.DatetimeIndex(sorted(minutes["session_id"].unique()))
    holdout_start = pd.Timestamp(config.CONFIRMATORY_HOLDOUT_START) if not args.confirmatory else None
    usable = sessions[sessions < holdout_start] if holdout_start is not None else sessions
    folds = features.walk_forward_folds(usable, args.scheme, config.TRAIN_FRAC, config.VAL_FRAC,
                                        args.folds, args.rolling_train_sessions)
    opens = minutes.groupby("session_id")["session_open"].first()
    session_open_of = lambda s: opens.loc[s]
    for f in folds:
        log("fold: train from {} | validation from {} | test {} -> {}".format(
            *(str(pd.Timestamp(x).date()) if x is not None else "end" for x in f)))

    runlog.write_run_json(out_dir, {
        "args": {k: v for k, v in vars(args).items() if k != "out_dir"}, "config": runlog.config_snapshot(config), "data_report": rep,
        "contract": contract, "folds": [[str(x) for x in f] for f in folds],
        "conventions": {
            "sessions": "CME Globex, America/Chicago, open 17:00 CT; schedule in timeframes.SCHEDULE",
            "bars": "session-anchored, [start, end), available at scheduled end",
            "daily": "last-trade session close, not settlement",
            "flat_policy": "flats kept; 'nonflat' = legacy conditional benchmark; 'all' counts flats as misses",
            "decision_latency": args.latency,
            "exploratory": not args.confirmatory,
            "retraining": "every model refitted from scratch on each fold's training rows; "
                          "thresholds and regime settings chosen on that fold's validation rows",
        }})
    all_m, all_p, all_y, all_b, infos, flog = [], [], [], [], [], []
    for freq in args.frequency:
        try:
            preds, info, fl = run_frequency(minutes, ticks, freq, args, folds, session_open_of,
                                            holdout_start, log)
        except features.SplitTooSmall as e:   # reported, not hidden
            log(f"[{freq}] SKIPPED: {e}")
            infos.append({"frequency": freq, "skipped": str(e)})
            continue
        infos.append(info)
        flog += fl
        keep = [c for c in preds.columns if c not in ("close",)]
        preds[keep].to_parquet(out_dir / f"predictions_{freq}.parquet", index=False)
        m, p, y, b = score(preds, freq, args.models, args)
        all_m.append(m), all_p.append(p), all_y.append(y), all_b.append(b)
        for name in args.models:
            if f"proba_{name}" in preds:
                evaluation.calibration_table(preds[f"proba_{name}"], preds["y3"]).to_csv(
                    out_dir / f"calibration_{freq}_{name}.csv", index=False)
    if not all_m:
        raise SystemExit("no frequency had enough data")
    M, P, Y, B = (pd.concat(x, ignore_index=True) for x in (all_m, all_p, all_y, all_b))
    M.to_csv(out_dir / "metrics.csv", index=False, float_format="%.5f")
    P.to_csv(out_dir / "paired_vs_majority.csv", index=False, float_format="%.4f")
    Y.to_csv(out_dir / "by_year.csv", index=False, float_format="%.5f")
    B.to_csv(out_dir / "backtest.csv", index=False, float_format="%.2f")
    pd.DataFrame(infos).to_csv(out_dir / "dataset_info.csv", index=False)
    import plots
    plots.plot_paired_differences(P, B, str(out_dir))
    pd.DataFrame(flog).to_csv(out_dir / "folds.csv", index=False)
    for _, r in M.iterrows():
        runlog.append_trial(config.RESULTS_DIR, source if not args.smoke else "smoke", {
            "run": out_dir.name, "frequency": r["frequency"], "model": r["model"], "target": args.target,
            "scheme": args.scheme, "features": "+".join(["base", *args.extra_features,
                                                        *(["context"] if args.context_features else [])]),
            "acc_all": r["acc_all"], "acc_nonflat": r["acc_nonflat"],
            "status": "confirmatory" if args.confirmatory else "exploratory"})
    show = P[["frequency", "model", "diff_pp", "ci_lo_pp_b20", "ci_hi_pp_b20", "n_sessions", "verdict"]]
    log("\nAccuracy (all eligible outcomes) minus majority baseline, pp, 95% block-bootstrap (20 sessions):")
    log(show.to_string(index=False, float_format=lambda v: f"{v:+.2f}"))
    nb = B[B["costs"].isin(["zero_cost", "base"])][["frequency", "model", "costs", "n_trades", "gross_usd",
                                                    "net_usd", "mean_daily_net_ci_lo_b5",
                                                    "mean_daily_net_ci_hi_b5", "max_drawdown_usd"]]
    log("\nBacktest, one NQ contract per signal (USD; costs are assumptions):")
    log(nb.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))
    log(f"\nSaved to {out_dir}")
    return out_dir
