"""
Tables, selection of the five configurations, charts and deck data for a
multitarget.py run. Reads only the saved predictions (no refitting).

Selection rule (fixed before looking at test results; uses VALIDATION only):
  evidence score z = block-bootstrap z-statistic (20-session blocks) of the
  paired validation improvement over the task's naive benchmark:
    return      squared-error reduction vs the zero-return forecast
    direction   accuracy gain vs the training-majority class
    volatility  squared-error reduction of log RV vs persistence (RV(t+1) = RV(t));
                QLIKE is reported but not used: on validation >99% of the persistence
                QLIKE comes from a few limit-locked candles with ~zero RV (Mar 2020).
                This choice was made AFTER a first report had printed test numbers.
  1. each task gets its highest-z model-timeframe configuration;
  2. the two remaining slots go to the two tasks whose best configuration
     on a DIFFERENT timeframe has the highest z (at most two per task).
Benchmarks are never selected. Timestral is scored on its own sample.
The selected five are then scored once on the test block.

Trading (direction task, test block): UP -> long, DOWN -> short at the next
candle's open, exit at its close, 1 NQ contract ($20/point), $2.50 per round
trip, zero slippage, 100% fills (idealised). Drawdown on trade-by-trade equity.

Usage: python multitarget_report.py results/multitarget/runs/<id> [out_dir]
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MULT, COST, COST_ALT = 20.0, 2.5, 5.0
FREQ_LABEL = {"5min": "5 min", "15min": "15 min", "30min": "30 min", "1h": "1 hour", "session": "1 day"}
BENCH = {"return": ("zero", "train_mean"), "direction": ("majority",),
         "volatility": ("persistence", "rolling_mean_22")}
NAIVE = {"return": "zero", "direction": "majority", "volatility": "persistence"}
MODEL_LABEL = {"zero": "Zero return", "train_mean": "Training mean", "majority": "Majority class",
               "persistence": "Persistence", "rolling_mean_22": "Rolling mean (22)", "har_ols": "HAR-OLS",
               "ridge": "Ridge", "logistic": "Logistic", "random_forest": "Random forest", "xgboost": "XGBoost",
               "mlp_64-32-16": "DNN 64-32-16", "mlp_64-32": "DNN 64-32", "mlp_128-64-32": "DNN 128-64-32",
               "mlp_256-128-64-32": "DNN 256-128-64-32", "timestral": "Timestral-5M (Mistral)"}


def block_boot(d, sess, block=20, reps=2000, seed=0):
    """Mean of d, 95% moving-block (sessions) bootstrap interval and z = mean / bootstrap s.e."""
    d = np.asarray(d, float)
    s = pd.Series(d).groupby(np.asarray(sess)).agg(["sum", "count"]).sort_index()
    S, N = s["sum"].to_numpy(), s["count"].to_numpy(float)
    k = len(S)
    est = S.sum() / N.sum()
    if k < 2 * block:
        return est, np.nan, np.nan, np.nan
    cs, cn = np.r_[0, np.cumsum(S)], np.r_[0, np.cumsum(N)]
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(k / block))
    st = rng.integers(0, k - block + 1, size=(reps, nb))
    bs = (cs[st + block] - cs[st]).sum(1) / (cn[st + block] - cn[st]).sum(1)
    se = bs.std(ddof=1)
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return est, lo, hi, est / se if se > 0 else np.nan


def cls_stats(pred_up, true_up, valid=None):
    """Accuracy etc. for boolean predictions; `valid` excludes rows (doji) from balanced metrics."""
    pred_up, true_up = np.asarray(pred_up, bool), np.asarray(true_up, bool)
    v = np.ones(len(pred_up), bool) if valid is None else np.asarray(valid, bool)
    p, t = pred_up[v], true_up[v]
    tp, fp = int((p & t).sum()), int((p & ~t).sum())
    tn, fn = int((~p & ~t).sum()), int((~p & t).sum())
    rec_up = tp / (tp + fn) if tp + fn else np.nan
    rec_dn = tn / (tn + fp) if tn + fp else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    return {"balanced_acc": np.nanmean([rec_up, rec_dn]), "precision_up": prec, "recall_up": rec_up,
            "f1_up": 2 * prec * rec_up / (prec + rec_up) if prec and rec_up and prec + rec_up else np.nan,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "pred_up_share": float(p.mean()) if len(p) else np.nan}


def trades(pos, move, cost=COST):
    pnl = pos * move * MULT - cost * (pos != 0)
    t = pos != 0
    tn = pnl[t]
    eq = np.r_[0.0, np.cumsum(tn)]
    win, loss = tn[tn > 0].sum(), tn[tn <= 0].sum()
    return {"n_trades": int(t.sum()), "net_usd": float(tn.sum()), "gross_usd": float((pos * move * MULT)[t].sum()),
            "win_rate": float((tn > 0).mean()) if t.any() else np.nan,
            "profit_factor": float(win / -loss) if loss < 0 else np.nan,
            "max_drawdown_usd": float((eq - np.maximum.accumulate(eq)).min()),
            "avg_net_per_trade_usd": float(tn.mean()) if t.any() else np.nan}


def positions(p_up, strat):
    side = np.where(p_up >= 0.5, 1, -1)
    if strat == "long_only":
        return np.where(side > 0, 1, 0)
    if strat == "short_only":
        return np.where(side < 0, -1, 0)
    return side


def evaluate(run):
    info = json.loads((run / "dataset_info.json").read_text())
    R, D, V, T = [], [], [], []
    for freq, fi in info.items():
        p = pd.read_parquet(run / f"predictions_{freq}.parquet")
        c = fi["consts"]
        for split in ("val", "test"):
            s = p[p.split == split]
            for rows in ("all", "timestral_sample"):
                q = s if rows == "all" else s[s.timestral_sample]
                if q.empty:
                    continue
                sess = q.label_session_id.to_numpy()
                tag = {"frequency": freq, "split": split, "rows": rows, "n": len(q)}
                # ---- return
                r = q.ret_bps.to_numpy()
                e0 = r ** 2
                for col in [k for k in q if k.startswith("ret__")]:
                    m = col[5:]
                    if (m == "timestral") != (rows == "timestral_sample") and not (rows == "timestral_sample" and m in BENCH["return"]):
                        continue
                    f = q[col].to_numpy()
                    e = (r - f) ** 2
                    est, lo, hi, z = block_boot(e0 - e, sess)
                    nz = r != 0
                    R.append({**tag, "model": m, "mae_bps": np.mean(np.abs(r - f)), "rmse_bps": np.sqrt(e.mean()),
                              "rmse_zero_bps": np.sqrt(e0.mean()), "r2_oos_pct": 100 * (1 - e.sum() / e0.sum()),
                              "r2_ci_lo_pct": 100 * lo / e0.mean() if lo == lo else np.nan,
                              "r2_ci_hi_pct": 100 * hi / e0.mean() if hi == hi else np.nan, "z": z,
                              "dir_acc": float((np.sign(f[nz]) == np.sign(r[nz])).mean()),
                              "corr": float(np.corrcoef(f, r)[0, 1]) if np.std(f) > 0 else np.nan})
                # ---- direction
                y = q.y3.to_numpy()
                maj_ok = (y == c["majority_direction"])
                for col in [k for k in q if k.startswith("dir__")] + ["dir__majority"]:
                    m = col[5:]
                    if (m == "timestral") != (rows == "timestral_sample") and not (rows == "timestral_sample" and m == "majority"):
                        continue
                    pu = np.full(len(q), c["majority_direction"] > 0) if m == "majority" else q[col].to_numpy() >= 0.5
                    ok = np.where(pu, 1, -1) == y
                    est, lo, hi, z = block_boot(ok.astype(float) - maj_ok, sess)
                    D.append({**tag, "model": m, "accuracy": ok.mean(), "majority_acc": maj_ok.mean(),
                              "acc_nonflat": ok[y != 0].mean(), "diff_pp": 100 * est, "ci_lo_pp": 100 * lo,
                              "ci_hi_pp": 100 * hi, "z": z, "n_doji": int((y == 0).sum()),
                              **cls_stats(pu, y > 0, y != 0)})
                    if split == "test" and rows == "all" and m != "timestral":
                        mv = q.move_points.to_numpy() * q.execution_eligible.to_numpy()
                        pp = np.where(pu, 1.0, 0.0)
                        for st in ("long_short", "long_only", "short_only"):
                            pos = positions(pp, st) * q.execution_eligible.to_numpy()
                            T.append({"frequency": freq, "model": m, "strategy": st, **trades(pos, mv),
                                      "net_usd_at_5": trades(pos, mv, COST_ALT)["net_usd"]})
                # ---- volatility
                lv, ln = q.lrv_next.to_numpy(), q.lrv_now.to_numpy()
                rv = np.exp(lv)
                fac = c["vol_level_factor"]
                ql = lambda F: rv / F - np.log(rv / F) - 1
                ql0 = ql(np.exp(ln))                       # persistence in levels
                up = lv > ln
                maj_up_ok = (up == bool(c["majority_vol_up"]))
                e_p = (lv - ln) ** 2
                for col in [k for k in q if k.startswith("vol__")]:
                    m = col[5:]
                    if (m == "timestral") != (rows == "timestral_sample") and not (rows == "timestral_sample" and m in BENCH["volatility"]):
                        continue
                    f = q[col].to_numpy()
                    F = np.exp(ln) if m == "persistence" else fac[col] * np.exp(f)
                    qm = ql(F)
                    est, lo, hi, z = block_boot(ql0 - qm, sess)
                    pu = f > ln
                    ok = pu == up
                    de, dlo, dhi, dz = block_boot(ok.astype(float) - maj_up_ok, sess)
                    e = (lv - f) ** 2
                    # selection evidence: squared error of log RV vs persistence (QLIKE vs persistence is
                    # dominated on validation by a few limit-locked candles with ~zero RV, March 2020)
                    le, llo, lhi, lz = block_boot(e_p - e, sess)
                    V.append({**tag, "model": m, "mae_log": np.mean(np.abs(lv - f)), "rmse_log": np.sqrt(e.mean()),
                              "rmse_log_persistence": np.sqrt(e_p.mean()),
                              "r2_log_vs_persistence_pct": 100 * (1 - e.sum() / e_p.sum()),
                              "r2_log_ci_lo_pct": 100 * llo / e_p.mean(), "r2_log_ci_hi_pct": 100 * lhi / e_p.mean(),
                              "z": lz, "qlike": qm.mean(), "qlike_persistence": ql0.mean(),
                              "qlike_impr_pct": 100 * est / ql0.mean(), "qlike_ci_lo_pct": 100 * lo / ql0.mean(),
                              "qlike_ci_hi_pct": 100 * hi / ql0.mean(), "qlike_z": z,
                              "corr_log": float(np.corrcoef(f, lv)[0, 1]) if np.std(f) > 0 else np.nan,
                              "voldir_acc": ok.mean(), "voldir_majority_acc": maj_up_ok.mean(),
                              "voldir_diff_pp": 100 * de, "voldir_ci_lo_pp": 100 * dlo, "voldir_ci_hi_pp": 100 * dhi,
                              "voldir_z": dz, "share_up": up.mean(), **{f"voldir_{k}": v for k, v in cls_stats(pu, up).items()}})
    return info, pd.DataFrame(R), pd.DataFrame(D), pd.DataFrame(V), pd.DataFrame(T)


def select(R, D, V):
    rows = []
    for task, df in (("return", R), ("direction", D), ("volatility", V)):
        v = df[(df.split == "val") & ~df.model.isin(BENCH[task])]
        v = v[(v.rows == "all") | (v.model == "timestral")]
        for _, r in v.iterrows():
            rows.append({"task": task, "frequency": r.frequency, "model": r.model, "val_z": r.z, "rows": r.rows})
    c = pd.DataFrame(rows).dropna(subset=["val_z"]).sort_values("val_z", ascending=False)
    pick, used = [], {}
    for task in ("return", "direction", "volatility"):
        b = c[c.task == task].iloc[0]
        pick.append({**b.to_dict(), "slot": "best of task"})
        used[task] = [b.frequency]
    second = []
    for task in ("return", "direction", "volatility"):
        o = c[(c.task == task) & ~c.frequency.isin(used[task])]
        if len(o):
            second.append({**o.iloc[0].to_dict(), "slot": "extra (different timeframe)"})
    second = sorted(second, key=lambda r: -r["val_z"])[:2]
    sel = pd.DataFrame(pick + second)
    order = {"return": 0, "direction": 1, "volatility": 2}
    sel = sel.assign(o=sel.task.map(order)).sort_values(["o", "val_z"], ascending=[True, False]).drop(columns="o")
    sel.insert(0, "id", range(1, len(sel) + 1))
    return c, sel.reset_index(drop=True)


def test_row(task, r, R, D, V, T):
    rows = "timestral_sample" if r.model == "timestral" else "all"
    if task == "return":
        x = R[(R.split == "test") & (R.rows == rows) & (R.frequency == r.frequency) & (R.model == r.model)].iloc[0]
        return {"key_metric": f"RMSE {x.rmse_bps:.2f} bp", "baseline": f"Zero return: RMSE {x.rmse_zero_bps:.2f} bp",
                "improvement": f"R² {x.r2_oos_pct:+.2f}% [{x.r2_ci_lo_pct:+.2f}, {x.r2_ci_hi_pct:+.2f}]",
                "improvement_value": x.r2_oos_pct, "ci_lo": x.r2_ci_lo_pct, "ci_hi": x.r2_ci_hi_pct,
                "secondary": f"sign accuracy {100 * x.dir_acc:.1f}%", **{f"t_{k}": v for k, v in x.items()}}
    if task == "direction":
        x = D[(D.split == "test") & (D.rows == rows) & (D.frequency == r.frequency) & (D.model == r.model)].iloc[0]
        return {"key_metric": f"Accuracy {100 * x.accuracy:.2f}%", "baseline": f"Majority: {100 * x.majority_acc:.2f}%",
                "improvement": f"{x.diff_pp:+.2f} pp [{x.ci_lo_pp:+.2f}, {x.ci_hi_pp:+.2f}]",
                "improvement_value": x.diff_pp, "ci_lo": x.ci_lo_pp, "ci_hi": x.ci_hi_pp,
                "secondary": f"balanced accuracy {100 * x.balanced_acc:.2f}%", **{f"t_{k}": v for k, v in x.items()}}
    x = V[(V.split == "test") & (V.rows == rows) & (V.frequency == r.frequency) & (V.model == r.model)].iloc[0]
    h = V[(V.split == "test") & (V.rows == "all") & (V.frequency == r.frequency) & (V.model == "har_ols")]
    har = {"har_qlike": h.qlike.iloc[0], "har_voldir_acc": h.voldir_acc.iloc[0], "har_rmse_log": h.rmse_log.iloc[0],
           "har_r2_log_pct": h.r2_log_vs_persistence_pct.iloc[0],
           "har_qlike_impr_pct": h.qlike_impr_pct.iloc[0]} if len(h) and rows == "all" else {}
    return {**har, "key_metric": f"log-RV RMSE {x.rmse_log:.3f}; RV UP/DOWN {100 * x.voldir_acc:.1f}%",
            "baseline": f"Persistence {x.rmse_log_persistence:.3f}; majority {100 * x.voldir_majority_acc:.1f}%",
            "improvement": f"R² {x.r2_log_vs_persistence_pct:+.1f}% [{x.r2_log_ci_lo_pct:+.1f}, {x.r2_log_ci_hi_pct:+.1f}]; "
                           f"direction {x.voldir_diff_pp:+.1f} pp",
            "improvement_value": x.r2_log_vs_persistence_pct, "ci_lo": x.r2_log_ci_lo_pct, "ci_hi": x.r2_log_ci_hi_pct,
            "secondary": f"vol direction {100 * x.voldir_acc:.1f}% vs {100 * x.voldir_majority_acc:.1f}%",
            **{f"t_{k}": v for k, v in x.items()}}


def charts(run, sel, out):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    files = {}
    for _, r in sel.iterrows():
        p = pd.read_parquet(run / f"predictions_{r.frequency}.parquet")
        q = p[p.split == "test"]
        if r.model == "timestral":
            q = q[q.timestral_sample]
        name = f"cfg{r.id}"
        if r.task == "return":
            f, y = q[f"ret__{r.model}"].to_numpy(), q.ret_bps.to_numpy()
            b = pd.qcut(pd.Series(f).rank(method="first"), 10, labels=False)
            g = pd.DataFrame({"f": f, "y": y, "b": b}).groupby("b").agg(f=("f", "mean"), y=("y", "mean"),
                                                                       se=("y", lambda s: s.std() / np.sqrt(len(s))))
            fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=200)
            ax.errorbar(g.f, g.y, yerr=1.96 * g.se, fmt="o", color="#1f5fa8", ecolor="#9db6d6", capsize=3, label="realized (mean ± 95%)")
            lim = [min(g.f.min(), g.y.min()), max(g.f.max(), g.y.max())]
            ax.plot(lim, lim, ls="--", color="#888", lw=1, label="perfect forecast")
            ax.axhline(0, color="#ccc", lw=0.8)
            ax.set_xlabel("forecast return, decile mean (bp)")
            ax.set_ylabel("realized next return (bp)")
            ax.legend(frameon=False, fontsize=9)
        elif r.task == "direction":
            fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
            pu = q[f"dir__{r.model}"].to_numpy()
            mv = q.move_points.to_numpy() * q.execution_eligible.to_numpy()
            ts = pd.to_datetime(q.entry_at)
            for st, col, lab in (("long_short", "#1f5fa8", "Long & Short"), ("long_only", "#2e8b57", "Long only"),
                                 ("short_only", "#c0392b", "Short only")):
                pos = positions(pu, st) * q.execution_eligible.to_numpy()
                pnl = pos * mv * MULT - COST * (pos != 0)
                ax.plot(ts, np.cumsum(pnl) / 1e3, color=col, lw=1.4, label=lab)
            ax.axhline(0, color="#999", lw=0.8)
            ax.set_ylabel("cumulative net P&L ($ thousand)")
            ax.legend(frameon=False, fontsize=9)
            fig.autofmt_xdate()
        else:
            f, y = q[f"vol__{r.model}"].to_numpy(), q.lrv_next.to_numpy()
            b = pd.qcut(pd.Series(f).rank(method="first"), 20, labels=False)
            g = pd.DataFrame({"f": f, "y": y, "b": b}).groupby("b").agg(f=("f", "mean"), y=("y", "mean"),
                                                                       lo=("y", lambda s: s.quantile(0.25)),
                                                                       hi=("y", lambda s: s.quantile(0.75)))
            fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=200)
            ax.fill_between(g.f, g.lo, g.hi, color="#d6e4f0", label="realized, 25-75%")
            ax.plot(g.f, g.y, "o-", color="#1f5fa8", ms=4, label="realized, mean")
            lim = [g.f.min(), g.f.max()]
            ax.plot(lim, lim, ls="--", color="#888", lw=1, label="perfect forecast")
            ax.set_xlabel("forecast log realized variance (20 bins)")
            ax.set_ylabel("realized log RV, next candle")
            ax.legend(frameon=False, fontsize=9)
        ax.set_title(f"{MODEL_LABEL[r.model]}, {FREQ_LABEL[r.frequency]} (test 2021-07 to 2025-05)", fontsize=10)
        fig.tight_layout()
        fp = out / f"chart_{name}_{r.task}.png"
        fig.savefig(fp)
        plt.close(fig)
        files[int(r.id)] = fp.name
    return files


def main(run, out=None):
    run = Path(run)
    out = Path(out) if out else run / "report"
    out.mkdir(parents=True, exist_ok=True)
    info, R, D, V, T = evaluate(run)
    cand, sel = select(R, D, V)
    tests = [test_row(r.task, r, R, D, V, T) for _, r in sel.iterrows()]
    five = pd.concat([sel, pd.DataFrame(tests)], axis=1)
    R.to_csv(out / "results_return.csv", index=False)
    D.to_csv(out / "results_direction.csv", index=False)
    V.to_csv(out / "results_volatility.csv", index=False)
    T.to_csv(out / "trading_direction_test.csv", index=False)
    cand.to_csv(out / "selection_validation_ranking.csv", index=False)
    five.to_csv(out / "selected_five.csv", index=False)
    files = charts(run, sel, out)
    models = pd.read_csv(run / "models.csv")
    tr = {}
    for _, r in sel[sel.task == "direction"].iterrows():
        tr[int(r.id)] = T[(T.frequency == r.frequency) & (T.model == r.model)].to_dict("records")
    tested = pd.concat([R[["frequency", "model"]].assign(task="return"), D[["frequency", "model"]].assign(task="direction"),
                        V[["frequency", "model"]].assign(task="volatility")]).drop_duplicates()
    tested.to_csv(out / "configurations_tested.csv", index=False)
    deck = {"run_id": run.name, "dataset": info, "n_models_tested": int(tested.model.nunique()),
            "n_configs": int(len(tested)), "selected": five.to_dict("records"), "charts": files,
            "trading": tr, "models": models.to_dict("records"),
            # Timestral (Mistral) on its own test sample, every timeframe: inventory-level outcome, not a selection
            "timestral_test": {f: {"return_r2_pct": R[(R.split == "test") & (R.frequency == f) & (R.model == "timestral")].r2_oos_pct.max(),
                                   "direction_diff_pp": D[(D.split == "test") & (D.frequency == f) & (D.model == "timestral")].diff_pp.max(),
                                   "vol_r2_log_pct": V[(V.split == "test") & (V.frequency == f) & (V.model == "timestral")].r2_log_vs_persistence_pct.max()}
                               for f in info},
            "costs": {"round_trip_usd": COST, "alt_round_trip_usd": COST_ALT, "slippage": "zero",
                      "fills": "100% at the next candle's open (idealised)"}}
    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (bool, np.bool_)):
            return bool(o)
        if isinstance(o, (int, np.integer)):
            return int(o)
        if isinstance(o, (float, np.floating)):
            return None if not np.isfinite(o) else float(o)
        return o if o is None or isinstance(o, str) else str(o)

    (out / "deck_data.json").write_text(json.dumps(clean(deck), indent=1), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(five[["id", "task", "model", "frequency", "val_z", "slot", "key_metric", "baseline", "improvement"]].to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
