"""
NQ next-candle study: model optimisation, trading strategies, equity curves.

Task (fixed before any result was seen):
  decision  at the close of candle t (latency 0 s by default: idealised),
            using candles <= t only;
  target    direction of candle t+1's BODY: up if close(t+1) > open(t+1),
            down if lower, doji (equal) kept as 0 = a miss for any call;
  trade     enter at open(t+1), exit at close(t+1), one NQ contract,
            $5 per round trip ($2.50 open + $2.50 close), zero slippage.
So the predicted outcome is exactly the traded return.

Protocol (same chronological boundaries as runs A-C, sessions < 2025-06-02):
  train      2006-01 .. 2017-08-24   models fitted here (non-flat rows)
  validation 2017-08-25 .. 2021-07-14 hyper-parameters, early stopping,
                                     strategy direction and thresholds
  test       2021-07-15 .. 2025-05-30 scored once, after all choices are fixed
Labels crossing a split boundary are purged. The test period was already
inspected with the ORIGINAL models in earlier runs, so it is untouched by
this study's choices but not historically untouched: findings stay exploratory.

Usage:
  python study.py                         # full study (1min 5min 15min 1h)
  python study.py --freqs 15min --years 2016 2019 --quick   # fast check
"""

import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import backtest
import config
import evaluation
import experiment
import features
import runlog
import strategy

TARGET = "next_candle_body_direction"
EXTRA_GROUPS = ("volume", "range", "rv", "time")
THRESHOLDS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10)
COST_GRID = (0.0, 2.5, 5.0, 12.5, 23.0)   # $ per round trip; 12.5/23 add spread+slippage


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def custom_features(b):
    """Candle shape, RSI and moving-average gaps; all known at bar t's close."""
    o, h, l, c = (b[k].astype(float) for k in ("open", "high", "low", "close"))
    f = pd.DataFrame(index=b.index)
    f["upper_wick_rel"] = (h - np.maximum(o, c)) / c
    f["lower_wick_rel"] = (np.minimum(o, c) - l) / c
    d = c.diff()
    gain, loss = d.clip(lower=0).rolling(14).mean(), (-d.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    f["rsi14_c"] = (rsi.where(loss > 0, 100.0).where(gain.notna()) - 50) / 50
    f["ma_gap_20"] = c / c.rolling(20).mean() - 1
    f["ma_gap_50"] = c / c.rolling(50).mean() - 1
    return f


BASE_COLS = None   # filled from features.make_features column names


def build(minutes, freq, latency):
    global BASE_COLS
    bars, elig = experiment.bars_for(minutes, None, freq)
    X, y3, meta, info = features.build_timed_dataset(
        bars, freq, TARGET, config.RETURN_LAGS, config.MOMENTUM_WINDOWS, config.VOLATILITY_WINDOWS,
        extra_groups=EXTRA_GROUPS, decision_latency=latency, label_eligible=elig)
    BASE_COLS = list(features.make_features(bars.head(60).reset_index(drop=True), config.RETURN_LAGS,
                                            config.MOMENTUM_WINDOWS, config.VOLATILITY_WINDOWS).columns)
    b = bars.reset_index(drop=True)
    cf = custom_features(b)
    clean50 = ~b["suspect"].astype(bool).rolling(50, min_periods=1).max().astype(bool)
    X = X.join(cf.loc[X.index])
    ok = X.notna().all(axis=1) & clean50.loc[X.index]
    X, y3, meta = X[ok], y3[ok], meta[ok]
    meta = backtest.attach_execution(meta, minutes)
    move = (meta["exit_price"] - meta["entry_price"]).to_numpy(float)
    elig_ex = meta["execution_eligible"].to_numpy(bool)
    mismatch = int((np.sign(move[elig_ex]) != y3.to_numpy()[elig_ex]).sum())
    info.update(n_rows_final=int(len(X)), n_execution_eligible=int(elig_ex.sum()),
                target_vs_traded_sign_mismatch=mismatch)
    return X.astype(np.float32), y3, meta, np.where(elig_ex, move, 0.0), elig_ex, info


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TorchMLP:
    """Feed-forward network with optional dropout / batch norm, early stopping on validation loss."""

    def __init__(self, layers, dropout=0.0, bn=False, weight_decay=1e-4, lr=1e-3, batch=4096,
                 max_epochs=40, patience=4, seed=42, optimizer="adamw"):
        self.kw = dict(layers=tuple(layers), dropout=dropout, bn=bn, weight_decay=weight_decay, lr=lr,
                       batch=batch, max_epochs=max_epochs, patience=patience, optimizer=optimizer)
        self.seed = seed

    def _net(self, n_in):
        import torch.nn as nn
        mods, prev = [], n_in
        for width in self.kw["layers"]:
            mods.append(nn.Linear(prev, width))
            if self.kw["bn"]:
                mods.append(nn.BatchNorm1d(width))
            mods.append(nn.ReLU())
            if self.kw["dropout"]:
                mods.append(nn.Dropout(self.kw["dropout"]))
            prev = width
        mods.append(nn.Linear(prev, 1))
        return nn.Sequential(*mods)

    def fit(self, X, y, Xv, yv):
        import torch
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.scaler = StandardScaler().fit(X)
        xt = torch.tensor(self.scaler.transform(X), dtype=torch.float32)
        yt = torch.tensor(np.asarray(y, np.float32))
        xv = torch.tensor(self.scaler.transform(Xv), dtype=torch.float32)
        yvt = torch.tensor(np.asarray(yv, np.float32))
        self.net = self._net(xt.shape[1])
        opt_cls = torch.optim.AdamW if self.kw["optimizer"] == "adamw" else torch.optim.Adam
        opt = opt_cls(self.net.parameters(), lr=self.kw["lr"], weight_decay=self.kw["weight_decay"])
        lossf = torch.nn.BCEWithLogitsLoss()
        g = torch.Generator().manual_seed(self.seed)
        best, best_state, bad = np.inf, None, 0
        self.history = []
        for epoch in range(self.kw["max_epochs"]):
            self.net.train()
            perm = torch.randperm(len(xt), generator=g)   # shuffling INSIDE the training set only
            for i in range(0, len(xt), self.kw["batch"]):
                idx = perm[i:i + self.kw["batch"]]
                opt.zero_grad()
                loss = lossf(self.net(xt[idx]).squeeze(1), yt[idx])
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                vl = float(lossf(self.net(xv).squeeze(1), yvt))
            self.history.append(vl)
            if vl < best - 1e-6:
                best, bad = vl, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.kw["patience"]:
                    break
        self.net.load_state_dict(best_state)
        self.epochs = len(self.history)
        return self

    def predict_proba(self, X):
        import torch
        self.net.eval()
        out = []
        with torch.no_grad():
            xs = torch.tensor(self.scaler.transform(X), dtype=torch.float32)
            for i in range(0, len(xs), 200_000):
                out.append(torch.sigmoid(self.net(xs[i:i + 200_000]).squeeze(1)).numpy())
        p = np.concatenate(out)
        return np.c_[1 - p, p]


class XGBEarly:
    def __init__(self, seed=42, **params):
        self.params, self.seed = params, seed

    def fit(self, X, y, Xv, yv):
        from xgboost import XGBClassifier
        early = self.params.pop("early", False)
        p = dict(tree_method="hist", n_jobs=-1, random_state=self.seed, **self.params)
        if early:
            p.update(n_estimators=1500, early_stopping_rounds=50, eval_metric="logloss")
            self.m = XGBClassifier(**p).fit(X, y, eval_set=[(Xv, yv)], verbose=False)
            self.best_iteration = int(self.m.best_iteration)
        else:
            self.m = XGBClassifier(**p).fit(X, y)
        self.params["early"] = early
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)


class SK:
    """sklearn estimator with train-only scaling; validation set unused."""

    def __init__(self, est, scale=True):
        self.m = make_pipeline(StandardScaler(), est) if scale else est

    def fit(self, X, y, Xv=None, yv=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.m.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)


def candidates(quick, seed):
    """(family, name, feature set, factory, is_original). The ORIGINAL entry of
    each family reproduces the repository's configuration (config.py)."""
    xgb0 = dict(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8)
    c = [
        ("logreg", "logreg_original", "base", lambda: SK(LogisticRegression(C=1.0, max_iter=1000)), True),
        ("logreg", "logreg_C0.01_ext", "ext", lambda: SK(LogisticRegression(C=0.01, max_iter=1000)), False),
        ("logreg", "logreg_C1_ext", "ext", lambda: SK(LogisticRegression(C=1.0, max_iter=1000)), False),
        ("xgboost", "xgb_original", "base", lambda: XGBEarly(seed, **xgb0), True),
        ("xgboost", "xgb_d3_early_ext", "ext", lambda: XGBEarly(seed, max_depth=3, learning_rate=0.05,
                                                                 subsample=0.8, colsample_bytree=0.8, early=True), False),
        ("xgboost", "xgb_d5_early_ext", "ext", lambda: XGBEarly(seed, max_depth=5, learning_rate=0.05,
                                                                 subsample=0.8, colsample_bytree=0.8,
                                                                 min_child_weight=100, early=True), False),
        ("random_forest", "rf_original", "base", lambda: SK(RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=50, n_jobs=-1, random_state=seed,
            max_samples=0.25), scale=False), True),
        ("random_forest", "rf_d10_ext", "ext", lambda: SK(RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=200, n_jobs=-1, random_state=seed,
            max_samples=0.25), scale=False), False),
        # repository's deep MLP (64-32-16, ReLU, L2 1e-3, lr 1e-3), trained with early stopping
        ("mlp", "mlp_original_64-32-16", "base", lambda: TorchMLP((64, 32, 16), weight_decay=1e-3,
                                                                  optimizer="adam", seed=seed), True),
        ("mlp", "mlp_64-32_ext", "ext", lambda: TorchMLP((64, 32), seed=seed), False),
        ("mlp", "mlp_128-64-32_ext", "ext", lambda: TorchMLP((128, 64, 32), seed=seed), False),
        ("mlp", "mlp_256-128-64-32_ext", "ext", lambda: TorchMLP((256, 128, 64, 32), seed=seed), False),
        ("mlp", "mlp_128-64-32_drop0.2_bn_ext", "ext",
         lambda: TorchMLP((128, 64, 32), dropout=0.2, bn=True, seed=seed), False),
        ("mlp", "mlp_256-128-64-32_drop0.2_bn_ext", "ext",
         lambda: TorchMLP((256, 128, 64, 32), dropout=0.2, bn=True, seed=seed), False),
        ("mlp", "mlp_128-64-32_lr3e-4_b1024_ext", "ext",
         lambda: TorchMLP((128, 64, 32), lr=3e-4, batch=1024, seed=seed), False),
    ]
    if quick:
        keep = {"logreg_original", "logreg_C1_ext", "xgb_original", "xgb_d3_early_ext", "rf_original",
                "mlp_original_64-32-16", "mlp_128-64-32_ext"}
        c = [x for x in c if x[1] in keep]
    return c


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def cls_metrics(pred, y3, proba=None):
    pred, y3 = np.asarray(pred), np.asarray(y3)
    m = evaluation.direction_metrics(pred, y3, proba)
    nf = y3 != 0
    yb, pb = (y3[nf] > 0).astype(int), (pred[nf] > 0).astype(int)
    n = len(y3)
    m.update(
        acc_all_ci95=1.96 * np.sqrt(m["acc_all"] * (1 - m["acc_all"]) / n) if n else np.nan,
        precision_up=precision_score(yb, pb, zero_division=0),
        recall_up=recall_score(yb, pb, zero_division=0),
        f1_up=f1_score(yb, pb, zero_division=0),
        acc_on_bullish=float((pred[y3 > 0] > 0).mean()) if (y3 > 0).any() else np.nan,
        acc_on_bearish=float((pred[y3 < 0] < 0).mean()) if (y3 < 0).any() else np.nan,
        n_bullish=int((y3 > 0).sum()), n_bearish=int((y3 < 0).sum()), n_doji=int((y3 == 0).sum()))
    return m


def rule_proba(X, rule, majority):
    if rule == "always_long":
        return np.ones(len(X))
    if rule == "always_short":
        return np.zeros(len(X))
    if rule == "reversal":
        s = -np.sign(X["ret_lag1"].to_numpy())
        return np.where(s > 0, 1.0, np.where(s < 0, 0.0, 1.0 if majority > 0 else 0.0))
    raise ValueError(rule)


# ---------------------------------------------------------------------------
# One frequency
# ---------------------------------------------------------------------------

def fit_predict(factory, cols, X, y3, tr_idx, va_mask, pred_masks):
    m = factory()
    Xtr, ytr = X.iloc[tr_idx][cols].to_numpy(), (y3.iloc[tr_idx] > 0).astype(int).to_numpy()
    va_nf = va_mask & (y3.to_numpy() != 0)
    Xv, yv = X[va_nf][cols].to_numpy(), (y3[va_nf] > 0).astype(int).to_numpy()
    t0 = time.time()
    m.fit(Xtr, ytr, Xv, yv)
    fit_s = time.time() - t0
    return m, fit_s, {k: m.predict_proba(X[msk][cols].to_numpy())[:, 1] for k, msk in pred_masks.items()}


def run_freq(minutes, freq, args, session_open_of, holdout_start, out, log):
    X, y3, meta, move, elig_ex, info = build(minutes, freq, args.latency)
    log(f"[{freq}] rows {len(X):,}; execution-eligible {info['n_execution_eligible']:,}; "
        f"target/trade sign mismatch {info['target_vs_traded_sign_mismatch']}")
    sessions_all = pd.DatetimeIndex(sorted(minutes["session_id"].unique()))
    usable = sessions_all[sessions_all < holdout_start]
    (first, v0, t0, _), = features.walk_forward_folds(usable, "holdout", config.TRAIN_FRAC, config.VAL_FRAC)
    tr, va, te = features.purged_split(meta, v0, t0, session_open_of, end=holdout_start)
    nfm = y3.to_numpy() != 0
    tr_idx = np.flatnonzero(tr & nfm)
    if args.max_train_rows and len(tr_idx) > args.max_train_rows:
        tr_idx = tr_idx[-args.max_train_rows:]
    majority = 1 if (y3.iloc[tr_idx] > 0).mean() >= 0.5 else -1
    ext_cols = list(X.columns)
    cols_of = {"base": BASE_COLS, "ext": ext_cols}
    sess = meta["label_session_id"].to_numpy()
    split_info = {"train": [str(meta.session_id[tr].min().date()), str(meta.session_id[tr].max().date()), int(tr.sum())],
                  "train_fit_rows": int(len(tr_idx)),
                  "validation": [str(meta.session_id[va].min().date()), str(meta.session_id[va].max().date()), int(va.sum())],
                  "test": [str(meta.session_id[te].min().date()), str(meta.session_id[te].max().date()), int(te.sum())],
                  "majority_train": majority, "features_base": BASE_COLS, "features_ext": ext_cols}
    log(f"[{freq}] split {json.dumps(split_info)[:300]}")

    probas, model_rows = {}, []
    for rule in ("always_long", "always_short", "reversal"):
        probas[rule] = {k: rule_proba(X[m], rule, majority) for k, m in (("train", tr), ("val", va), ("test", te))}
    for family, name, fs, factory, original in candidates(args.quick, args.seed):
        m, fit_s, pr = fit_predict(factory, cols_of[fs], X, y3, tr_idx, va,
                                   {"train": tr, "val": va, "test": te})
        probas[name] = pr
        extra = {}
        if isinstance(m, TorchMLP):
            extra = {"epochs": m.epochs, "best_val_logloss": float(min(m.history))}
        if isinstance(m, XGBEarly) and hasattr(m, "best_iteration"):
            extra = {"best_iteration": m.best_iteration}
        vm = cls_metrics(np.where(pr["val"] >= 0.5, 1, -1), y3[va], pr["val"])
        model_rows.append({"frequency": freq, "family": family, "model": name, "features": fs,
                           "original": original, "fit_seconds": round(fit_s, 1),
                           "val_balanced_acc": vm["balanced_acc_nonflat"], "val_acc_all": vm["acc_all"],
                           "val_log_loss": vm.get("log_loss_nonflat"), **extra})
        log(f"  [{freq}] {name}: fit {fit_s:.0f}s, val bal.acc {vm['balanced_acc_nonflat']:.4f} {extra}")
    sel = pd.DataFrame(model_rows)
    # optimised = best validation balanced accuracy within each family (may be the original)
    sel["selected_optimised"] = False
    for fam, g in sel.groupby("family"):
        sel.loc[g["val_balanced_acc"].idxmax(), "selected_optimised"] = True

    # ---- classification metrics on the test set
    acc_rows = []
    base_correct = np.where(probas["always_long" if majority > 0 else "always_short"]["test"] >= .5, 1, -1) == y3[te].to_numpy()
    for name, pr in probas.items():
        pred = np.where(pr["test"] >= 0.5, 1, -1)
        cm = cls_metrics(pred, y3[te], None if name in ("always_long", "always_short", "reversal") else pr["test"])
        pdiff = evaluation.paired_difference(pred == y3[te].to_numpy(), base_correct, meta.session_id[te].to_numpy())
        tr_acc = float((np.where(pr["train"] >= .5, 1, -1) == y3[tr].to_numpy()).mean())
        acc_rows.append({"frequency": freq, "model": name, "train_acc_all": tr_acc, **cm,
                         "vs_majority_pp": pdiff["diff_pp"], "vs_majority_ci_lo": pdiff["ci_lo_pp_b20"],
                         "vs_majority_ci_hi": pdiff["ci_hi_pp_b20"]})

    # ---- strategies: thresholds chosen on validation, scored on train/val/test
    strat_rows, eq_rows, cost_rows = [], [], []
    for name, pr in probas.items():
        for st in strategy.STRATEGIES:
            if name == "always_long" and st != "long_only" or name == "always_short" and st != "short_only":
                continue
            is_rule = name in ("always_long", "always_short", "reversal")
            th_opt = 0.0 if is_rule else strategy.choose_threshold(
                pr["val"], move[va], st, THRESHOLDS)
            for th_kind, th in (("none", 0.0), ("validated", th_opt)):
                if th_kind == "validated" and (is_rule or th == 0.0):
                    continue
                for split, msk in (("train", tr), ("val", va), ("test", te)):
                    pos = strategy.positions(pr[split], st, th) * elig_ex[msk]
                    s = strategy.summarize(pos, move[msk], sess[msk], args.cost, bootstrap=split != "train")
                    strat_rows.append({"frequency": freq, "model": name, "strategy": st, "threshold_kind": th_kind,
                                       "threshold": th, "split": split, **s})
                    if split == "test":
                        e = strategy.daily_equity(pos, move[msk], sess[msk], args.cost)
                        eq_rows.append(pd.DataFrame({"frequency": freq, "model": name, "strategy": st,
                                                     "threshold_kind": th_kind, "session": e.index,
                                                     "cum_net_usd": e.to_numpy()}))
                        for c in COST_GRID:
                            g, n_ = strategy.trade_pnl(pos, move[msk], c)
                            cost_rows.append({"frequency": freq, "model": name, "strategy": st,
                                              "threshold_kind": th_kind, "rt_cost": c,
                                              "net_usd": float(n_[pos != 0].sum()), "n_trades": int((pos != 0).sum())})
    # ---- threshold sensitivity on test for every model (reported, never used for choice)
    thr_rows = []
    for name, pr in probas.items():
        if name in ("always_long", "always_short", "reversal"):
            continue
        for st in strategy.STRATEGIES:
            for th in THRESHOLDS:
                for split, msk in (("val", va), ("test", te)):
                    pos = strategy.positions(pr[split], st, th) * elig_ex[msk]
                    g, n_ = strategy.trade_pnl(pos, move[msk], args.cost)
                    thr_rows.append({"frequency": freq, "model": name, "strategy": st, "threshold": th,
                                     "split": split, "n_trades": int((pos != 0).sum()),
                                     "net_usd": float(n_[pos != 0].sum())})

    # save compact test predictions for charts / audits
    keep = pd.DataFrame({"session_id": meta.session_id[te].to_numpy(), "label_session_id": sess[te],
                         "label_start": meta.label_start[te].to_numpy(), "y3": y3[te].to_numpy(),
                         "move_points": move[te], "execution_eligible": elig_ex[te]})
    for name, pr in probas.items():
        keep[f"p_{name}"] = pr["test"]          # float64: thresholds must reproduce exactly
    keep.to_parquet(out / f"test_predictions_{freq}.parquet", index=False)
    probas_val = {n: p["val"] for n, p in probas.items()}
    return dict(sel=sel, acc=pd.DataFrame(acc_rows), strat=pd.DataFrame(strat_rows),
                eq=pd.concat(eq_rows, ignore_index=True), cost=pd.DataFrame(cost_rows),
                thr=pd.DataFrame(thr_rows), info={**info, "splits": split_info},
                data=(X, y3, meta, move, elig_ex, sess, majority, cols_of), probas_val=probas_val)


def walk_forward(minutes, freq, res, model_name, strat_name, args, session_open_of, holdout_start, log):
    """Yearly expanding refits of one fixed configuration; thresholds re-chosen on each fold's validation."""
    X, y3, meta, move, elig_ex, sess, _, cols_of = res["data"]
    spec = {c[1]: c for c in candidates(False, args.seed)}[model_name]
    usable = pd.DatetimeIndex(sorted(minutes["session_id"].unique()))
    usable = usable[usable < holdout_start]
    rows = []
    for k, (first, v0, t0, t1) in enumerate(features.walk_forward_folds(usable, "expanding", config.TRAIN_FRAC,
                                                                        config.VAL_FRAC, n_folds=4)):
        tr, va, te = features.purged_split(meta, v0, t0, session_open_of, end=t1 if t1 is not None else holdout_start)
        tr_idx = np.flatnonzero(tr & (y3.to_numpy() != 0))
        if args.max_train_rows and len(tr_idx) > args.max_train_rows:
            tr_idx = tr_idx[-args.max_train_rows:]
        _, _, pr = fit_predict(spec[3], cols_of[spec[2]], X, y3, tr_idx, va, {"val": va, "test": te})
        th = strategy.choose_threshold(pr["val"], move[va], strat_name, THRESHOLDS)
        for th_kind, t in (("none", 0.0), ("validated", th)):
            pos = strategy.positions(pr["test"], strat_name, t) * elig_ex[te]
            s = strategy.summarize(pos, move[te], sess[te], args.cost)
            acc = float((np.where(pr["test"] >= .5, 1, -1) == y3[te].to_numpy()).mean())
            rows.append({"frequency": freq, "model": model_name, "strategy": strat_name, "fold": k,
                         "test_start": str(t0.date()), "test_end": str(t1.date()) if t1 is not None else "end",
                         "threshold_kind": th_kind, "threshold": t, "acc_all": acc, **s})
        log(f"  [{freq}] walk-forward fold {k}: {rows[-2]['net_usd']:,.0f} USD (no threshold)")
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--freqs", nargs="+", default=["1min", "5min", "15min", "1h"])
    p.add_argument("--years", type=int, nargs=2, default=None)
    p.add_argument("--latency", default="0s", help="decision delay after the bar close (0s = idealised)")
    p.add_argument("--cost", type=float, default=strategy.DEFAULT_RT_COST, help="USD per round trip")
    p.add_argument("--max-train-rows", type=int, default=2_000_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true", help="few configurations (pipeline check)")
    p.add_argument("--tag", default="")
    args = p.parse_args(argv)
    import torch
    torch.set_num_threads(12)

    out = runlog.new_run_dir(config.RESULTS_DIR, "nq_study", tag=args.tag)
    logf = open(out / "console.log", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(f"run folder: {out}")
    minutes, _, rep, contract = experiment.load_minutes("nq", tuple(args.years) if args.years else None)
    opens = minutes.groupby("session_id")["session_open"].first()
    session_open_of = lambda s: opens.loc[s]
    holdout_start = pd.Timestamp(config.CONFIRMATORY_HOLDOUT_START)
    runlog.write_run_json(out, {"args": vars(args), "data_report": rep, "contract": contract,
                                "instrument": {"symbol": "NQ", "multiplier_usd_per_point": strategy.NQ_MULTIPLIER,
                                               "tick": strategy.NQ_TICK, "tick_value_usd": strategy.NQ_TICK * strategy.NQ_MULTIPLIER},
                                "costs": {"round_trip_usd": args.cost, "slippage": "zero (idealised limit fills, 100% fill rate)"},
                                "target": TARGET, "entry": "open of candle t+1", "exit": "close of candle t+1"})
    parts = {k: [] for k in ("sel", "acc", "strat", "eq", "cost", "thr", "wf")}
    infos = {}
    for freq in args.freqs:
        res = run_freq(minutes, freq, args, session_open_of, holdout_start, out, log)
        for k in ("sel", "acc", "strat", "eq", "cost", "thr"):
            parts[k].append(res[k])
        infos[freq] = res["info"]
        # walk-forward for the configuration chosen on validation (best validation net P&L)
        v = res["strat"]
        v = v[(v.split == "val") & ~v.model.isin(["always_long", "always_short", "reversal"])]
        best = v.sort_values("net_usd", ascending=False).iloc[0]
        log(f"[{freq}] validation-best: {best.model} / {best.strategy} / {best.threshold_kind} "
            f"({best.net_usd:,.0f} USD on validation)")
        parts["wf"].append(walk_forward(minutes, freq, res, best.model, best.strategy, args,
                                        session_open_of, holdout_start, log))
        for k, frame in parts.items():           # save after every frequency (resumable reading)
            if frame:
                pd.concat(frame, ignore_index=True).to_csv(out / f"{ {'sel': 'model_selection', 'acc': 'accuracy_test', 'strat': 'strategies', 'eq': 'equity_daily', 'cost': 'cost_sensitivity', 'thr': 'threshold_sensitivity', 'wf': 'walk_forward'}[k]}.csv", index=False)
        (out / "dataset_info.json").write_text(json.dumps(infos, indent=2, default=str), encoding="utf-8")
        del res
    log(f"done: {out}")


if __name__ == "__main__":
    main()
