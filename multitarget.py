"""
NASDAQ (NQ futures) multi-target study: what is predictable one candle ahead?

Instrument: NQ (E-mini Nasdaq-100 futures, CME), back-adjusted continuous
1-minute TradeStation series 2006-2025 (Europe/Rome, end-stamped), $20 per
index point. No QQQ, MNQ, cash index or VIX data is used.

Three tasks, all for candle t+1, decided at the close of candle t with
candles <= t only (latency 0, idealised):
  return      r(t+1) = log(C(t+1) / C(t)), in basis points (regression)
  direction   UP if C(t+1) > O(t+1), DOWN if lower; a doji (equal) is kept
              and counts as a miss for any call (classification)
  volatility  RV(t+1) = sum of squared 1-minute log returns INSIDE candle
              t+1 (realized variance; the first term is log(C/O) of its first
              minute, so nothing from candle t enters). Regression on log RV;
              direction UP if RV(t+1) > RV(t), DOWN if lower or equal.
Intraday labels never cross a session (timeframes.next_label_eligible).

Splits (same chronological boundaries as study.py, sessions < 2025-06-02):
  train 2006-01 .. 2017-08-24, validation 2017-08-25 .. 2021-07-14,
  test 2021-07-15 .. 2025-05-30; labels crossing a boundary are purged.
Hyper-parameters, early stopping and the configuration selection use the
validation block only. The test period was inspected in earlier phases of
the project (direction task), so test results are exploratory.

Timestral (Mistral AI's pretrained time-series model, local weights,
zero-shot, no API): forecasts quantiles of the next close from the last 512
closes of the same timeframe. Median -> return and direction; the q10-q90
spread -> a variance forecast, mapped to log RV by a linear calibration
fitted on the validation sample. Scored on a fixed random sample of
validation and test origins (it is not a classifier or regressor trained here).

Outputs (results/multitarget/runs/<id>/): predictions_<freq>.parquet (val +
test rows, every model), models.csv, dataset_info.json, console.log, run.json.
Tables, selection and charts: multitarget_report.py.

Usage:
  python multitarget.py                               # 5min 15min 30min 1h session
  python multitarget.py --freqs 1h --years 2015 2019 --quick --no-timestral   # pipeline check
"""

import argparse
import json
import os
import time
import warnings

os.environ.setdefault("HF_HUB_OFFLINE", "1")          # Timestral weights must already be cached
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import config  # noqa: E402
import experiment  # noqa: E402
import features  # noqa: E402
import runlog  # noqa: E402
import study  # noqa: E402

TIMESTRAL_ID = "mistralai/Timestral-5M"
CONTEXT = 512
MLP_LAYERS = {"mlp_64-32-16": (64, 32, 16), "mlp_64-32": (64, 32), "mlp_128-64-32": (128, 64, 32),
              "mlp_256-128-64-32": (256, 128, 64, 32)}
HAR_COLS = ["lrv_1", "lrv_5", "lrv_22"]
Z90 = 1.2815515655446004


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def candle_rv(minutes, freq):
    """Realized variance of every candle from its own 1-minute log returns."""
    m = minutes
    if freq == "session":
        b = m["session_open"]
    else:
        f = pd.Timedelta(freq)
        b = m["session_open"] + ((m["bar_start"] - m["session_open"]) // f) * f
    key = [m["session_id"], b]
    prev = m["close"].groupby(key).shift(1)
    r = np.log(m["close"] / prev.fillna(m["open"]))
    rv = (r ** 2).groupby(key).sum()
    rv.index.names = ["session_id", "bar_start"]
    return rv


def majority_direction(y3_train):
    """Majority of UP vs DOWN among NON-doji training candles (dojis are a miss for any call)."""
    y = np.asarray(y3_train)
    return int(1 if (y[y != 0] > 0).mean() >= 0.5 else -1)


def build(minutes, freq):
    X, y3, meta, move, elig, info = study.build(minutes, freq, "0s")
    bars, _ = experiment.bars_for(minutes, None, freq)
    b = bars.reset_index(drop=True)
    rv = candle_rv(minutes, freq).reindex(pd.MultiIndex.from_arrays([b.session_id, b.bar_start])).to_numpy()
    floor = (0.25 / b["close"].to_numpy(float)) ** 2 / 4          # a quarter of a one-tick move
    rv = np.maximum(np.nan_to_num(rv, nan=0.0), floor)
    rvs = pd.Series(rv)
    f = pd.DataFrame(index=b.index)
    f["lrv_1"] = np.log(rvs)
    f["lrv_5"] = np.log(rvs.rolling(5).mean())
    f["lrv_22"] = np.log(rvs.rolling(22).mean())
    if freq != "session":
        # mean RV of the NEXT candle's time-of-day slot over the 5 previous sessions (known at t)
        off = (b["bar_start"] - b["session_open"]).rename("off")
        seas = rvs.groupby(off).transform(lambda s: s.shift(1).rolling(5).mean())
        f["lrv_seas_next"] = np.log(seas.shift(-1))
    close = b["close"].to_numpy(float)
    idx = X.index.to_numpy()
    tg = pd.DataFrame({"ret_bps": 1e4 * np.log(close[idx + 1] / close[idx]),
                       "lrv_next": np.log(rv[idx + 1]), "lrv_now": np.log(rv[idx])}, index=X.index)
    X = X.join(f.loc[X.index].astype(np.float32))
    ok = X.notna().all(axis=1) & np.isfinite(tg).all(axis=1)
    keep = ok.to_numpy()
    X, y3, meta, tg = X[ok], y3[ok], meta[ok], tg[ok]
    info.update(n_rows_multitarget=int(keep.sum()), rv_definition="sum of squared 1-min log returns inside the candle")
    return X, y3, meta, move[keep], elig[keep], tg, b


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Net:
    """Feed-forward network (ReLU, AdamW, early stopping on validation loss); mse or bce loss."""

    def __init__(self, layers, loss, lr=1e-3, wd=1e-4, batch=4096, max_epochs=40, patience=4, seed=42):
        self.layers, self.loss, self.lr, self.wd = layers, loss, lr, wd
        self.batch, self.max_epochs, self.patience, self.seed = batch, max_epochs, patience, seed

    def fit(self, X, y, Xv, yv):
        import torch
        import torch.nn as nn
        torch.manual_seed(self.seed)
        self.sx = StandardScaler().fit(X)
        y, yv = np.asarray(y, np.float64), np.asarray(yv, np.float64)
        self.mu, self.sd = (0.0, 1.0) if self.loss == "bce" else (y.mean(), y.std() or 1.0)
        t = lambda a: torch.tensor(np.asarray(a, np.float32))
        xt, xv = t(self.sx.transform(X)), t(self.sx.transform(Xv))
        yt, yvt = t((y - self.mu) / self.sd), t((yv - self.mu) / self.sd)
        mods, prev = [], xt.shape[1]
        for w in self.layers:
            mods += [nn.Linear(prev, w), nn.ReLU()]
            prev = w
        self.net = nn.Sequential(*mods, nn.Linear(prev, 1))
        lossf = nn.BCEWithLogitsLoss() if self.loss == "bce" else nn.MSELoss()
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        batch = self.batch if len(xt) >= 20_000 else 128
        g = torch.Generator().manual_seed(self.seed)
        best, state, bad, self.history = np.inf, None, 0, []
        for _ in range(self.max_epochs):
            self.net.train()
            perm = torch.randperm(len(xt), generator=g)
            for i in range(0, len(xt), batch):
                j = perm[i:i + batch]
                opt.zero_grad()
                lossf(self.net(xt[j]).squeeze(1), yt[j]).backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                vl = float(lossf(self.net(xv).squeeze(1), yvt))
            self.history.append(vl)
            if vl < best - 1e-7:
                best, bad, state = vl, 0, {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.load_state_dict(state)
        self.info = {"epochs": len(self.history), "best_val_loss": best}
        return self

    def _raw(self, X):
        import torch
        self.net.eval()
        with torch.no_grad():
            xs = torch.tensor(self.sx.transform(X), dtype=torch.float32)
            return np.concatenate([self.net(xs[i:i + 200_000]).squeeze(1).numpy() for i in range(0, len(xs), 200_000)])

    def predict(self, X):
        z = self._raw(X).astype(np.float64)
        return 1 / (1 + np.exp(-z)) if self.loss == "bce" else self.mu + self.sd * z


class Tuned:
    """Picks one hyper-parameter value on validation (lowest MSE / log loss)."""

    def __init__(self, makers, kind):
        self.makers, self.kind = makers, kind

    def fit(self, X, y, Xv, yv):
        best = np.inf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for label, mk in self.makers.items():
                m = mk().fit(X, y)
                if self.kind == "clf":
                    p = np.clip(m.predict_proba(Xv)[:, 1], 1e-6, 1 - 1e-6)
                    loss = -np.mean(yv * np.log(p) + (1 - yv) * np.log(1 - p))
                else:
                    loss = np.mean((m.predict(Xv) - yv) ** 2)
                if loss < best:
                    best, self.m, self.info = loss, m, {"chosen": label}
        return self

    def predict(self, X):
        return self.m.predict_proba(X)[:, 1] if self.kind == "clf" else self.m.predict(X)


class Plain:
    def __init__(self, est, clf=False, cols=None):
        self.est, self.clf, self.info = est, clf, {}

    def fit(self, X, y, Xv=None, yv=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.est.fit(X, y)
        return self

    def predict(self, X):
        return self.est.predict_proba(X)[:, 1] if self.clf else self.est.predict(X)


class XGB:
    def __init__(self, clf, seed, small):
        self.clf, self.seed, self.small, self.info = clf, seed, small, {}

    def fit(self, X, y, Xv, yv):
        from xgboost import XGBClassifier, XGBRegressor
        p = dict(tree_method="hist", n_jobs=-1, random_state=self.seed, n_estimators=2000, learning_rate=0.03,
                 max_depth=4, subsample=0.8, colsample_bytree=0.8, min_child_weight=5 if self.small else 100,
                 early_stopping_rounds=50, eval_metric="logloss" if self.clf else "rmse")
        self.m = (XGBClassifier if self.clf else XGBRegressor)(**p).fit(X, y, eval_set=[(Xv, yv)], verbose=False)
        self.info = {"best_iteration": int(self.m.best_iteration)}
        return self

    def predict(self, X):
        return self.m.predict_proba(X)[:, 1] if self.clf else self.m.predict(X)


def model_specs(task, seed, small, quick):
    """(name, family, feature set, factory). Feature sets: 'all' or 'har'."""
    leaf = 20 if small else 200
    clf = task == "direction"
    sc = lambda est: make_pipeline(StandardScaler(), est)
    if clf:
        lin = ("logistic", "linear", "all", lambda: Tuned({f"C={c}": (lambda c=c: sc(LogisticRegression(C=c, max_iter=2000)))
                                                          for c in (0.01, 1.0)}, "clf"))
    else:
        lin = ("ridge", "linear", "all", lambda: Tuned({f"alpha={a}": (lambda a=a: sc(Ridge(alpha=a)))
                                                       for a in (0.1, 10.0, 1000.0)}, "reg"))
    rf_cls = RandomForestClassifier if clf else RandomForestRegressor
    specs = [lin,
             ("random_forest", "tree ensemble", "all", lambda: Plain(rf_cls(
                 n_estimators=200, max_depth=8, min_samples_leaf=leaf, max_samples=0.25, n_jobs=-1,
                 random_state=seed), clf=clf)),
             ("xgboost", "tree ensemble", "all", lambda: XGB(clf, seed, small))]
    for name, layers in MLP_LAYERS.items():
        specs.append((name, "deep learning", "all",
                      lambda layers=layers: Net(layers, "bce" if clf else "mse", seed=seed)))
    if task == "volatility":
        specs.insert(0, ("har_ols", "statistical", "har", lambda: Plain(LinearRegression())))
    if quick:
        specs = [s for s in specs if s[0] in ("logistic", "ridge", "har_ols", "xgboost", "mlp_64-32")]
    return specs


# ---------------------------------------------------------------------------
# Timestral (Mistral AI)
# ---------------------------------------------------------------------------

def timestral_forecasts(fc, close, pos, batch=256):
    med, q10, q90 = [], [], []
    for i in range(0, len(pos), batch):
        p = pos[i:i + batch]
        ctx = np.stack([close[j - CONTEXT + 1:j + 1] for j in p]).astype(np.float32)
        out = fc.forecast(ctx, prediction_length=1)
        lv = np.asarray(out.quantile_levels, float)
        q = out.quantiles[:, 0, 0, :].cpu().numpy().astype(np.float64)
        med.append(out.median[:, 0, 0].cpu().numpy().astype(np.float64))
        q10.append(q[:, int(np.argmin(np.abs(lv - 0.1)))])
        q90.append(q[:, int(np.argmin(np.abs(lv - 0.9)))])
    return np.concatenate(med), np.concatenate(q10), np.concatenate(q90)


# ---------------------------------------------------------------------------
# One timeframe
# ---------------------------------------------------------------------------

def run_freq(minutes, freq, args, session_open_of, holdout_start, fc, out, log):
    t_start = time.time()
    X, y3, meta, move, elig, tg, bars = build(minutes, freq)
    usable = pd.DatetimeIndex(sorted(minutes["session_id"].unique()))
    usable = usable[usable < holdout_start]
    (_, v0, t0, _), = features.walk_forward_folds(usable, "holdout", config.TRAIN_FRAC, config.VAL_FRAC)
    tr, va, te = features.purged_split(meta, v0, t0, session_open_of, end=holdout_start)
    tr_idx = np.flatnonzero(tr)
    if args.max_train_rows and len(tr_idx) > args.max_train_rows:
        tr_idx = tr_idx[-args.max_train_rows:]
    small = len(tr_idx) < 20_000
    cols_all = list(X.columns)
    cols = {"all": cols_all, "har": HAR_COLS + (["lrv_seas_next"] if "lrv_seas_next" in cols_all else [])}
    info = {"frequency": freq, "horizon": "one candle (t+1)", "n_rows": int(len(X)),
            "train": [str(meta.session_id[tr].min().date()), str(meta.session_id[tr].max().date()), int(tr.sum())],
            "train_fit_rows": int(len(tr_idx)),
            "validation": [str(meta.session_id[va].min().date()), str(meta.session_id[va].max().date()), int(va.sum())],
            "test": [str(meta.session_id[te].min().date()), str(meta.session_id[te].max().date()), int(te.sum())],
            "data_period": [str(meta.session_id.min().date()), str(meta.session_id.max().date())],
            "features": cols_all, "har_features": cols["har"]}
    log(f"[{freq}] rows {len(X):,}; train {info['train']} val {info['validation']} test {info['test']}")

    ret, lrv, lrv_now = tg.ret_bps.to_numpy(), tg.lrv_next.to_numpy(), tg.lrv_now.to_numpy()
    lo, hi = np.quantile(ret[tr_idx], [0.001, 0.999])        # winsorise the training return target only
    nf = y3.to_numpy() != 0
    targets = {"return": (np.clip(ret, lo, hi), ret),
               "direction": ((y3.to_numpy() > 0).astype(int), None),
               "volatility": (lrv, lrv)}
    keep = pd.DataFrame({"split": np.where(va, "val", np.where(te, "test", "train")),
                         "session_id": meta.session_id.to_numpy(), "label_session_id": meta.label_session_id.to_numpy(),
                         "label_start": meta.label_start.to_numpy(), "entry_at": meta.entry_at.to_numpy(),
                         "y3": y3.to_numpy(), "move_points": move, "execution_eligible": elig,
                         "ret_bps": ret, "lrv_next": lrv, "lrv_now": lrv_now})
    rows_out = va | te
    keep = keep[rows_out].reset_index(drop=True)
    model_rows, consts = [], {"majority_direction": majority_direction(y3.to_numpy()[tr_idx]),
                              "majority_vol_up": int((lrv[tr_idx] > lrv_now[tr_idx]).mean() >= 0.5),
                              "train_mean_ret_bps": float(ret[tr_idx].mean()),
                              "ret_winsor_bps": [float(lo), float(hi)]}
    # simple benchmarks
    keep["vol__persistence"] = lrv_now[rows_out]
    keep["vol__rolling_mean_22"] = X["lrv_22"].to_numpy()[rows_out]
    keep["ret__zero"] = 0.0
    keep["ret__train_mean"] = consts["train_mean_ret_bps"]
    resid = {"vol__persistence": lrv[tr_idx] - lrv_now[tr_idx], "vol__rolling_mean_22": lrv[tr_idx] - X["lrv_22"].to_numpy()[tr_idx]}
    for task in ("return", "direction", "volatility"):
        ytr_all, _ = targets[task]
        fit_rows = tr_idx[nf[tr_idx]] if task == "direction" else tr_idx
        val_rows = np.flatnonzero(va & nf) if task == "direction" else np.flatnonzero(va)
        for name, family, fs, factory in model_specs(task, args.seed, small, args.quick):
            Xm = X[cols[fs]].to_numpy(np.float32)
            t1 = time.time()
            m = factory().fit(Xm[fit_rows], ytr_all[fit_rows], Xm[val_rows], ytr_all[val_rows])
            pr = m.predict(Xm[rows_out])
            prefix = {"return": "ret", "direction": "dir", "volatility": "vol"}[task]
            keep[f"{prefix}__{name}"] = pr
            if task == "volatility":
                resid[f"vol__{name}"] = lrv[tr_idx] - m.predict(Xm[tr_idx])
            row = {"frequency": freq, "task": task, "model": name, "family": family, "features": fs,
                   "fit_rows": int(len(fit_rows)), "fit_seconds": round(time.time() - t1, 1), **getattr(m, "info", {})}
            model_rows.append(row)
            log(f"  [{freq}] {task:10s} {name:18s} {row['fit_seconds']:6.1f}s {getattr(m, 'info', {})}")
    # level bias correction for variance forecasts (QLIKE): E[RV] = c * exp(log forecast), c from train
    consts["vol_level_factor"] = {k: float(np.mean(np.exp(v))) for k, v in resid.items()}

    # Timestral on a fixed random sample of validation and test origins
    keep["timestral_sample"] = False
    if fc is not None:
        bar_pos = X.index.to_numpy()[rows_out]
        close = bars["close"].to_numpy(np.float64)
        rng = np.random.default_rng(args.seed + sum(map(ord, freq)))
        med = np.full(len(keep), np.nan)
        q10, q90 = med.copy(), med.copy()
        for split in ("val", "test"):
            cand = np.flatnonzero((keep.split == split).to_numpy() & (bar_pos >= CONTEXT - 1))
            pick = np.sort(rng.choice(cand, size=min(args.timestral_n, len(cand)), replace=False))
            t1 = time.time()
            m_, a_, b_ = timestral_forecasts(fc, close, bar_pos[pick])
            med[pick], q10[pick], q90[pick] = m_, a_, b_
            keep.loc[pick, "timestral_sample"] = True
            log(f"  [{freq}] timestral {split}: {len(pick):,} origins, {time.time() - t1:.0f}s")
        last = close[bar_pos]
        keep["ret__timestral"] = 1e4 * np.log(med / last)
        keep["dir__timestral"] = np.where(med > last, 1.0, np.where(med < last, 0.0, 0.5))
        sig2 = ((np.log(q90) - np.log(q10)) / (2 * Z90)) ** 2
        keep["vol_raw__timestral"] = np.log(np.maximum(sig2, 1e-14))
        cal = (keep.split == "val").to_numpy() & keep.timestral_sample.to_numpy()
        A = np.c_[np.ones(cal.sum()), keep.vol_raw__timestral[cal]]
        coef, *_ = np.linalg.lstsq(A, keep.lrv_next[cal], rcond=None)
        keep["vol__timestral"] = coef[0] + coef[1] * keep.vol_raw__timestral
        r_cal = keep.lrv_next[cal] - keep.vol__timestral[cal]
        consts["timestral_vol_calibration"] = {"intercept": float(coef[0]), "slope": float(coef[1]),
                                               "fitted_on": "validation sample"}
        consts["vol_level_factor"]["vol__timestral"] = float(np.mean(np.exp(r_cal)))
        for task in ("return", "direction", "volatility"):
            model_rows.append({"frequency": freq, "task": task, "model": "timestral", "family": "Mistral AI (pretrained)",
                               "features": f"last {CONTEXT} closes", "fit_rows": 0, "fit_seconds": 0.0,
                               "model_id": TIMESTRAL_ID, "zero_shot": True})
    keep.to_parquet(out / f"predictions_{freq}.parquet", index=False)
    info.update(consts=consts, seconds=round(time.time() - t_start))
    return pd.DataFrame(model_rows), info


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--freqs", nargs="+", default=["5min", "15min", "30min", "1h", "session"])
    p.add_argument("--years", type=int, nargs=2, default=None)
    p.add_argument("--max-train-rows", type=int, default=2_000_000)
    p.add_argument("--timestral-n", type=int, default=20_000, help="origins per split")
    p.add_argument("--no-timestral", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="")
    args = p.parse_args(argv)
    import torch
    torch.set_num_threads(12)
    out = runlog.new_run_dir(config.RESULTS_DIR, "multitarget", tag=args.tag)
    logf = open(out / "console.log", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    fc = None
    if not args.no_timestral:
        from timestral import TimestralForecaster
        fc = TimestralForecaster.from_pretrained(TIMESTRAL_ID, device="cpu")
    log(f"run folder: {out}; timestral: {TIMESTRAL_ID if fc else 'off'}")
    minutes, _, rep, contract = experiment.load_minutes("nq", tuple(args.years) if args.years else None)
    opens = minutes.groupby("session_id")["session_open"].first()
    holdout_start = pd.Timestamp(config.CONFIRMATORY_HOLDOUT_START)
    runlog.write_run_json(out, {"args": vars(args), "data_report": rep, "contract": contract,
                                "instrument": {"symbol": "NQ", "multiplier_usd_per_point": 20.0},
                                "timestral": {"model_id": TIMESTRAL_ID if fc else None, "context": CONTEXT,
                                              "zero_shot": True, "api": "none (local weights)"}})
    models, infos = [], {}
    for freq in args.freqs:
        mr, info = run_freq(minutes, freq, args, lambda s: opens.loc[s], holdout_start, fc, out, log)
        models.append(mr)
        infos[freq] = info
        pd.concat(models, ignore_index=True).to_csv(out / "models.csv", index=False)
        (out / "dataset_info.json").write_text(json.dumps(infos, indent=2, default=str), encoding="utf-8")
    log(f"done: {out}")


if __name__ == "__main__":
    main()
