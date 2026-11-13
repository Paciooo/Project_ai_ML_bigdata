"""
Cross-check of a multitarget.py run and its report, recomputed from the saved
predictions WITHOUT importing multitarget_report.py.

Checks:
  * predictions: finite targets and forecasts, probabilities in [0, 1],
    validation strictly before test, rows in time order, Timestral sample
    sizes;
  * every point estimate in results_return / results_direction /
    results_volatility (val and test, all rows and Timestral sample):
    MAE, RMSE, R^2 vs zero, sign accuracy; accuracy, majority accuracy,
    balanced accuracy, precision, recall, F1, confusion counts; log-RV RMSE,
    R^2 vs persistence and vs HAR-OLS, QLIKE, RV-direction accuracy (incl.
    the cross-fitted Timestral calibration on validation); interval order;
  * selection: validation z in the ranking equals the results tables, the
    rule (best per task, two extras on a different timeframe, at most two per
    task, no benchmark) reproduces selected_five.csv, and its test fields
    equal the results rows;
  * trading: every row of trading_direction_test.csv ($2.50 and $5);
  * presentation: deck_data.json (report and presentation copies identical),
    run id, selected rows, trading, charts present and identical, costs, no VIX.
It does NOT re-run the bootstrap (interval values), rebuild the data from the
1-minute bars or refit models.

Usage: python verify_multitarget.py results/multitarget/runs/<id> reports/multitarget_final [presentation_multitarget]
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MULT = 20.0
BENCH = {"return": {"zero", "train_mean"}, "direction": {"majority"}, "volatility": {"persistence", "rolling_mean_22"}}


def main(run, rep, pres=None):
    run, rep = Path(run), Path(rep)
    info = json.loads((run / "dataset_info.json").read_text())
    R, D, V = (pd.read_csv(rep / f"results_{k}.csv") for k in ("return", "direction", "volatility"))
    T = pd.read_csv(rep / "trading_direction_test.csv")
    rank = pd.read_csv(rep / "selection_validation_ranking.csv")
    five = pd.read_csv(rep / "selected_five.csv")
    problems, n = [], 0

    def check(ok, msg):
        nonlocal n
        n += 1
        if not ok:
            problems.append(msg)

    def close(a, b, tol=1e-6):
        return (pd.isna(a) and pd.isna(b)) or (not pd.isna(a) and not pd.isna(b) and abs(float(a) - float(b)) <= tol * max(1, abs(float(b))))

    for freq, fi in info.items():
        p = pd.read_parquet(run / f"predictions_{freq}.parquet")
        c = fi["consts"]
        # ---- predictions
        fc = [k for k in p if k.split("__")[0] in ("ret", "dir", "vol") and k != "vol_raw__timestral"]
        for k in fc:
            v = p[k] if not k.endswith("timestral") else p.loc[p.timestral_sample, k]
            check(np.isfinite(v).all(), f"{freq} {k}: non-finite forecasts")
            if k.startswith("dir__"):
                check(((v >= 0) & (v <= 1)).all(), f"{freq} {k}: probability outside [0, 1]")
        check(np.isfinite(p[["ret_bps", "lrv_next", "lrv_now", "move_points"]]).all().all(), f"{freq}: non-finite targets")
        ls = pd.to_datetime(p.label_start)
        check(ls[p.split == "val"].max() < ls[p.split == "test"].min(), f"{freq}: validation overlaps test")
        check(ls.is_monotonic_increasing, f"{freq}: rows not in time order")
        for sp in ("val", "test"):
            check(int((p.timestral_sample & (p.split == sp)).sum()) <= 20_000, f"{freq} {sp}: Timestral sample too large")
        check(str(p.session_id[p.split == "test"].max().date()) == fi["test"][1], f"{freq}: test end date differs from dataset_info")
        # past-to-future Timestral calibration on validation (independent implementation):
        # 5 blocks of origin sessions, block k fitted on rows whose target session precedes it
        tsv = np.full(len(p), np.nan)
        tfac = np.full(len(p), np.nan)
        tsok = np.zeros(len(p), bool)
        if "vol_raw__timestral" in p:
            vs = ((p.split == "val") & p.timestral_sample).to_numpy()
            org = pd.to_datetime(p.session_id)
            lab = pd.to_datetime(p.label_session_id)
            x, y = p.vol_raw__timestral.to_numpy(), p.lrv_next.to_numpy()
            blocks = np.array_split(np.sort(org[vs].unique()), 5)
            for k, bs in enumerate(blocks[1:], start=1):
                fit = vs & (lab < bs[0]).to_numpy()
                prd = vs & ((org >= bs[0]) & (org <= bs[-1])).to_numpy()
                check(lab[fit].max() < org[prd].min(), f"{freq}: calibration block {k} uses targets after its origins")
                b, a = np.polyfit(x[fit], y[fit], 1)
                tsv[prd] = a + b * x[prd]
                tfac[prd] = np.mean(np.exp(y[fit] - a - b * x[fit]))
                tsok |= prd
            te = (p.split == "test").to_numpy()
            check(lab[vs].max() < org[te & p.timestral_sample.to_numpy()].min(),
                  f"{freq}: test calibration uses targets after the first test origin")
            tsv[te] = p.vol__timestral.to_numpy()[te]
            tfac[te] = c["vol_level_factor"]["vol__timestral"]
            tsok |= te & p.timestral_sample.to_numpy()
            cal = json.loads((rep / "timestral_calibration.json").read_text(encoding="utf-8"))[freq]
            got = [b["eval_rows"] for b in cal["validation_past_to_future"] if b["scored"]]
            check(got == [int(((org >= bs[0]) & (org <= bs[-1])).to_numpy()[vs].sum()) for bs in blocks[1:]],
                  f"{freq}: timestral_calibration.json block sizes differ")
        for sp in ("val", "test"):
            for rows in ("all", "timestral_sample"):
                m_ = (p.split == sp).to_numpy() & (np.ones(len(p), bool) if rows == "all" else p.timestral_sample.to_numpy())
                q = p[m_]
                if q.empty:
                    continue
                sel = lambda df: df[(df.frequency == freq) & (df.split == sp) & (df.rows == rows)].set_index("model")
                # return
                r = q.ret_bps.to_numpy()
                for mdl, x in sel(R).iterrows():
                    f = q[f"ret__{mdl}"].to_numpy()
                    e = (r - f) ** 2
                    nz = r != 0
                    for k, v in (("rmse_bps", np.sqrt(e.mean())), ("mae_bps", np.abs(r - f).mean()),
                                 ("r2_oos_pct", 100 * (1 - e.sum() / (r ** 2).sum())),
                                 ("dir_acc", (np.sign(f[nz]) == np.sign(r[nz])).mean())):
                        check(close(x[k], v), f"{freq} {sp} {rows} return {mdl} {k}: {x[k]} vs {v}")
                    if pd.notna(x.r2_ci_lo_pct):
                        check(x.r2_ci_lo_pct <= x.r2_oos_pct + 1e-9 <= x.r2_ci_hi_pct + 2e-9, f"{freq} {sp} return {mdl}: interval order")
                # direction
                yv = q.y3.to_numpy()
                maj = yv == c["majority_direction"]
                for mdl, x in sel(D).iterrows():
                    pu = np.full(len(q), c["majority_direction"] > 0) if mdl == "majority" else q[f"dir__{mdl}"].to_numpy() >= 0.5
                    ok = np.where(pu, 1, -1) == yv
                    nf = yv != 0
                    pp, tt = pu[nf], yv[nf] > 0
                    tp, fp, tn, fn = (pp & tt).sum(), (pp & ~tt).sum(), (~pp & ~tt).sum(), (~pp & tt).sum()
                    rec = [tp / (tp + fn) if tp + fn else np.nan, tn / (tn + fp) if tn + fp else np.nan]
                    exp = {"accuracy": ok.mean(), "majority_acc": maj.mean(), "diff_pp": 100 * (ok.mean() - maj.mean()),
                           "tp": tp, "fp": fp, "tn": tn, "fn": fn, "balanced_acc": np.nanmean(rec),
                           "precision_up": tp / (tp + fp) if tp + fp else 0.0, "recall_up": tp / (tp + fn) if tp + fn else 0.0,
                           "f1_up": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}
                    for k, v in exp.items():
                        check(close(x[k], v), f"{freq} {sp} {rows} direction {mdl} {k}: {x[k]} vs {v}")
                    if pd.notna(x.ci_lo_pp):
                        check(x.ci_lo_pp <= x.diff_pp + 1e-9 <= x.ci_hi_pp + 2e-9, f"{freq} {sp} direction {mdl}: interval order")
                # volatility (Timestral sample: only causally calibrated rows, same rows for benchmarks)
                if rows == "timestral_sample" and "vol_raw__timestral" in p:
                    m_ = m_ & tsok
                    q = p[m_]
                    check(sel(V).n.eq(len(q)).all(), f"{freq} {sp} volatility Timestral-sample row count")
                lv, ln = q.lrv_next.to_numpy(), q.lrv_now.to_numpy()
                rv, up = np.exp(lv), lv > ln
                e_p = (lv - ln) ** 2
                ql = lambda F: np.mean(rv / F - np.log(rv / F) - 1)
                for mdl, x in sel(V).iterrows():
                    if mdl == "timestral_calibrated":
                        f, F = tsv[m_], tfac[m_] * np.exp(tsv[m_])
                    else:
                        f = q[f"vol__{mdl}"].to_numpy()
                        F = np.exp(ln) if mdl == "persistence" else c["vol_level_factor"][f"vol__{mdl}"] * np.exp(f)
                    e = (lv - f) ** 2
                    ok = (f > ln) == up
                    exp = {"rmse_log": np.sqrt(e.mean()), "r2_log_vs_persistence_pct": 100 * (1 - e.sum() / e_p.sum()),
                           "qlike": ql(F), "qlike_persistence": ql(np.exp(ln)), "voldir_acc": ok.mean(),
                           "voldir_majority_acc": (up == bool(c["majority_vol_up"])).mean()}
                    if "r2_log_vs_har_pct" in x and pd.notna(x.r2_log_vs_har_pct):
                        fh = q.vol__har_ols.to_numpy()
                        exp["r2_log_vs_har_pct"] = 100 * (1 - e.sum() / ((lv - fh) ** 2).sum())
                        exp["voldir_diff_vs_har_pp"] = 100 * (ok.mean() - ((fh > ln) == up).mean())
                        check(x.r2_log_vs_har_ci_lo_pct <= x.r2_log_vs_har_pct + 1e-9 <= x.r2_log_vs_har_ci_hi_pct + 2e-9,
                              f"{freq} {sp} vol {mdl}: HAR interval order")
                    for k, v in exp.items():
                        check(close(x[k], v), f"{freq} {sp} {rows} volatility {mdl} {k}: {x[k]} vs {v}")
                    if pd.notna(x.r2_log_ci_lo_pct):
                        check(x.r2_log_ci_lo_pct <= x.r2_log_vs_persistence_pct + 1e-9 <= x.r2_log_ci_hi_pct + 2e-9,
                              f"{freq} {sp} vol {mdl}: interval order")
        # ---- trading
        q = p[p.split == "test"]
        mv = q.move_points.to_numpy() * q.execution_eligible.to_numpy()
        for _, t in T[T.frequency == freq].iterrows():
            pu = np.full(len(q), c["majority_direction"] > 0) if t.model == "majority" else q[f"dir__{t.model}"].to_numpy() >= 0.5
            side = np.where(pu, 1, -1)
            pos = {"long_short": side, "long_only": np.where(side > 0, 1, 0), "short_only": np.where(side < 0, -1, 0)}[t.strategy]
            pos = pos * q.execution_eligible.to_numpy()
            tr = pos != 0
            for cost, col in ((2.5, "net_usd"), (5.0, "net_usd_at_5")):
                pnl = (pos * mv * MULT - cost * tr)[tr]
                check(close(t[col], pnl.sum(), 1e-9), f"{freq} {t.model} {t.strategy}: {col} {t[col]} vs {pnl.sum()}")
            pnl = (pos * mv * MULT - 2.5 * tr)[tr]
            eq = np.r_[0, np.cumsum(pnl)]
            loss = pnl[pnl <= 0].sum()
            exp = {"n_trades": tr.sum(), "gross_usd": (pos * mv * MULT)[tr].sum(),
                   "win_rate": (pnl > 0).mean() if tr.any() else np.nan,          # no trade: N/A, not 0
                   "profit_factor": pnl[pnl > 0].sum() / -loss if loss < 0 else np.nan,
                   "max_drawdown_usd": (eq - np.maximum.accumulate(eq)).min(),
                   "avg_net_per_trade_usd": pnl.mean() if tr.any() else np.nan}
            for k, v in exp.items():
                check(close(t[k], v, 1e-9), f"{freq} {t.model} {t.strategy}: {k} {t[k]} vs {v}")

    # ---- selection
    for task, df in (("return", R), ("direction", D), ("volatility", V)):
        v = df[(df.split == "val") & ~df.model.isin(BENCH[task])]
        v = v[(v.rows == "all") | v.model.str.startswith("timestral")]
        m = rank[rank.task == task].merge(v[["frequency", "model", "z"]], on=["frequency", "model"], how="left")
        check(len(m) == len(v.dropna(subset=["z"])) and np.allclose(m.val_z, m.z), f"selection ranking z differs from results ({task})")
    check(not any(r.model in BENCH[r.task] for r in rank.itertuples()), "a benchmark is in the selection ranking")
    picks = []
    for task in ("return", "direction", "volatility"):
        picks.append(rank[rank.task == task].sort_values("val_z", ascending=False).iloc[0])
    extra = []
    for b in picks:
        o = rank[(rank.task == b.task) & (rank.frequency != b.frequency)].sort_values("val_z", ascending=False)
        if len(o):
            extra.append(o.iloc[0])
    extra = sorted(extra, key=lambda r: -r.val_z)[:2]
    want = {(r.task, r.frequency, r.model) for r in picks + extra}
    got = set(zip(five.task, five.frequency, five.model))
    check(want == got and len(five) == 5, f"selection rule gives {sorted(want)}, report has {sorted(got)}")
    check(set(five.task) == {"return", "direction", "volatility"} and five.task.value_counts().max() <= 3, "task coverage")
    for _, r in five.iterrows():
        df = {"return": R, "direction": D, "volatility": V}[r.task]
        rows = "timestral_sample" if str(r.model).startswith("timestral") else "all"
        x = df[(df.split == "test") & (df.rows == rows) & (df.frequency == r.frequency) & (df.model == r.model)].iloc[0]
        for k in x.index:
            if f"t_{k}" in five and isinstance(x[k], (int, float, np.number)):
                check(close(r[f"t_{k}"], x[k], 1e-9), f"selected {r.id}: t_{k} differs from results")

    # ---- presentation data
    dd = json.loads((rep / "deck_data.json").read_text(encoding="utf-8"))
    check(dd["run_id"] == run.name, "deck_data.json from another run")
    check(dd["costs"]["round_trip_usd"] == 2.5, "deck cost is not $2.50")
    check("vix" not in json.dumps(dd).lower(), "deck data mentions VIX")
    check([(s["task"], s["frequency"], s["model"]) for s in dd["selected"]] == list(zip(five.task, five.frequency, five.model)),
          "deck selected rows differ from selected_five.csv")
    for s in dd["selected"]:
        x = five[five.id == s["id"]].iloc[0]
        for k in ("improvement_value", "ci_lo", "ci_hi", "val_z"):
            check(close(s[k], x[k], 1e-9), f"deck selected {s['id']} {k}")
    for sid, rows in dd["trading"].items():
        for t in rows:
            x = T[(T.frequency == t["frequency"]) & (T.model == t["model"]) & (T.strategy == t["strategy"])].iloc[0]
            check(close(t["net_usd"], x.net_usd, 1e-9) and close(t["max_drawdown_usd"], x.max_drawdown_usd, 1e-9),
                  f"deck trading {sid} {t['strategy']}")
    for f, fi in dd["dataset"].items():
        check(fi["test"] == info[f]["test"] and fi["train"] == info[f]["train"], f"deck dataset {f} differs")
    for sid, fn in dd["charts"].items():
        check((rep / fn).exists(), f"chart {fn} missing")
    if pres:
        pres = Path(pres)
        h = lambda fp: hashlib.sha256(fp.read_bytes()).hexdigest()
        check(h(pres / "deck_data.json") == h(rep / "deck_data.json"), "presentation deck_data.json differs from the report")
        for fn in dd["charts"].values():
            check((pres / fn).exists() and h(pres / fn) == h(rep / fn), f"presentation chart {fn} differs")
    print(f"{n} checks, {len(problems)} problems")
    for msg in problems[:50]:
        print("  " + msg)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
