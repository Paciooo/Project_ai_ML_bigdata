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
  * data profile: every value of data_profile.csv (rows, sessions, class
    shares, return mean / s.d., mean log RV, share of rising RV, lag-1
    autocorrelations within sessions);
  * presentation: deck_data.json (report and presentation copies identical),
    run id, selected rows, data profile, charts present and identical, no
    trading or P&L fields, no VIX;
  * sample_checks.json (tools/sample_checks.py): every flag above the
    threshold and equal to log(close / predecessor close) on the sequence
    that generated it, predecessor times, categories re-derived from their
    rule, counts and the daily accounting adding up, the daily test rows of
    the run (daily_sample_used) equal to the saved forecasts with every
    missing session given one reason, no saved daily forecast within 50
    sessions of an in-schedule flag when the run used the flagged-bar filter
    (dataset_info flagged_bar_filter), and the all-rows rescoring of the incomplete-
    candle check recomputed from the saved forecasts. The 1-minute prices and
    the incomplete-candle classes themselves are not re-read.
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

BENCH = {"return": {"zero", "train_mean"}, "direction": {"majority"}, "volatility": {"persistence", "rolling_mean_22"}}


def main(run, rep, pres=None):
    run, rep = Path(run), Path(rep)
    info = json.loads((run / "dataset_info.json").read_text())
    R, D, V = (pd.read_csv(rep / f"results_{k}.csv") for k in ("return", "direction", "volatility"))
    prof = pd.read_csv(rep / "data_profile.csv")
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
        # ---- data profile (independent implementation of the lag-1 autocorrelation)
        for sp in ("val", "test"):
            q = p[p.split == sp]
            x = prof[(prof.frequency == freq) & (prof.split == sp)]
            check(len(x) == 1, f"{freq} {sp}: data profile row missing")
            if len(x) != 1:
                continue
            x = x.iloc[0]
            s_ = np.zeros(len(q)) if freq == "session" else q.session_id.to_numpy()

            def ac(v):
                v = np.asarray(v, float)
                df = pd.DataFrame({"a": v[1:], "b": v[:-1]})[s_[1:] == s_[:-1]]
                return df.a.corr(df.b)
            exp = {"n": len(q), "sessions": q.label_session_id.nunique(), "share_up": (q.y3 > 0).mean(),
                   "share_down": (q.y3 < 0).mean(), "share_doji": (q.y3 == 0).mean(), "ret_mean_bps": q.ret_bps.mean(),
                   "ret_sd_bps": q.ret_bps.std(), "lrv_mean": q.lrv_next.mean(),
                   "share_rv_up": (q.lrv_next > q.lrv_now).mean(), "ret_acf1": ac(q.ret_bps), "lrv_acf1": ac(q.lrv_next)}
            for k, v in exp.items():
                check(close(x[k], v, 1e-6), f"{freq} {sp} data profile {k}: {x[k]} vs {v}")

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
    check(not any(k in dd for k in ("trading", "costs")), "deck data still holds trading or cost fields")
    check(pd.DataFrame(dd["data_profile"]).round(9).equals(prof.round(9)), "deck data profile differs from data_profile.csv")
    check("vix" not in json.dumps(dd).lower(), "deck data mentions VIX")
    check([(s["task"], s["frequency"], s["model"]) for s in dd["selected"]] == list(zip(five.task, five.frequency, five.model)),
          "deck selected rows differ from selected_five.csv")
    for s in dd["selected"]:
        x = five[five.id == s["id"]].iloc[0]
        for k in ("improvement_value", "ci_lo", "ci_hi", "val_z"):
            check(close(s[k], x[k], 1e-9), f"deck selected {s['id']} {k}")
    for sid, rows in dd.get("direction_by_year", {}).items():
        r = five[five.id == int(sid)].iloc[0]
        q = pd.read_parquet(run / f"predictions_{r.frequency}.parquet")
        q = q[q.split == "test"]
        yv = q.y3.to_numpy()
        d = 100 * ((np.where(q[f"dir__{r.model}"].to_numpy() >= 0.5, 1, -1) == yv).astype(float)
                   - (yv == info[r.frequency]["consts"]["majority_direction"]))
        yr = pd.to_datetime(q.session_id).dt.year.astype(str).to_numpy()
        for x in rows:
            v = d.mean() if x["year"] == "all" else d[yr == x["year"]].mean()
            check(close(x["gain_pp"], v, 1e-9), f"deck direction_by_year {sid} {x['year']}: {x['gain_pp']} vs {v}")
            check(x["ci_lo_pp"] <= x["gain_pp"] + 1e-9 <= x["ci_hi_pp"] + 2e-9, f"deck direction_by_year {sid} {x['year']}: interval order")
    # five-model comparison per task (one per family, task's best validation timeframe) and winners
    cmp_ = pd.read_csv(rep / "comparison_by_task.csv")
    fams = {"return": ["ridge", "random_forest", "xgboost", "dnn", "timestral"],
            "direction": ["logistic", "random_forest", "xgboost", "dnn", "timestral"],
            "volatility": ["ridge", "random_forest", "xgboost", "dnn", "timestral_calibrated"]}
    winners = []
    for task, fl in fams.items():
        rk = rank[rank.task == task].sort_values("val_z", ascending=False)
        tf = rk.iloc[0].frequency
        rk = rk[rk.frequency == tf]
        want = []
        for fam in fl:
            x = rk[rk.model.str.startswith("mlp_")] if fam == "dnn" else rk[rk.model == fam]
            if len(x):
                want.append((x.iloc[0].model, x.iloc[0].val_z))
        want = sorted(want, key=lambda t: -t[1])
        got = cmp_[cmp_.task == task].sort_values("rank_in_task")
        check([m for m, _ in want] == list(got.model) and (got.frequency == tf).all(), f"comparison {task}: rule gives {want}")
        winners.append((task, want[0][0], want[0][1]))
        df = {"return": R, "direction": D, "volatility": V}[task]
        for _, r in got.iterrows():
            rows = "timestral_sample" if str(r.model).startswith("timestral") else "all"
            x = df[(df.split == "test") & (df.rows == rows) & (df.frequency == tf) & (df.model == r.model)].iloc[0]
            for k in x.index:
                if f"t_{k}" in cmp_ and isinstance(x[k], (int, float, np.number)):
                    check(close(r[f"t_{k}"], x[k], 1e-9), f"comparison {task} {r.model}: t_{k} differs from results")
        check([r["model"] for r in dd["comparison"][task]] == list(got.model), f"deck comparison {task} differs")
    winners = sorted(winners, key=lambda t: -t[2])
    check([(w["task"], w["model"]) for w in dd["winners"]] == [(t, m) for t, m, _ in winners], "deck winners / best overall differ")
    # task x timeframe map: validation-best trained model per cell, test improvement from the results tables
    tmap = pd.read_csv(rep / "timeframe_map.csv")
    key = {"return": ("r2_oos_pct", "r2_ci_lo_pct"), "direction": ("diff_pp", "ci_lo_pp"),
           "volatility": ("r2_log_vs_persistence_pct", "r2_log_ci_lo_pct")}
    for _, r in tmap.iterrows():
        rk = rank[(rank.task == r.task) & (rank.frequency == r.frequency) & (rank.rows == "all")].sort_values("val_z", ascending=False)
        check(len(rk) and rk.iloc[0].model == r.model, f"timeframe map {r.task} {r.frequency}: not the validation-best model")
        df = {"return": R, "direction": D, "volatility": V}[r.task]
        x = df[(df.split == "test") & (df.rows == "all") & (df.frequency == r.frequency) & (df.model == r.model)].iloc[0]
        check(close(r.value, x[key[r.task][0]], 1e-9) and close(r.ci_lo, x[key[r.task][1]], 1e-9), f"timeframe map {r.task} {r.frequency}: value")
        if r.task == "volatility":
            check(close(r.vs_har, x.r2_log_vs_har_pct, 1e-9), f"timeframe map {r.frequency}: vs HAR")
    check(len(dd["timeframe_map"]) == len(tmap) and [(m["task"], m["frequency"]) for m in dd["timeframe_map"]] == list(zip(tmap.task, tmap.frequency)),
          "deck timeframe map differs from timeframe_map.csv")
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
        for fn in ("sample_checks.json", "timestral_same_sample.json", "yahoo_check.json"):     # appendix inputs (tools/)
            if (rep / fn).exists():
                check((pres / fn).exists() and h(pres / fn) == h(rep / fn), f"presentation {fn} differs from the report")
    if (rep / "sample_checks.json").exists():     # content of the sample diagnostics (tools/sample_checks.py)
        try:
            check_sample_checks(json.loads((rep / "sample_checks.json").read_text(encoding="utf-8")), run, info, check, close)
        except KeyError as e:                     # e.g. a file written by the earlier version of the tool
            check(False, f"sample_checks.json: missing field {e} (re-run tools/sample_checks.py)")
    print(f"{n} checks, {len(problems)} problems")
    for msg in problems[:50]:
        print("  " + msg)
    return 1 if problems else 0


def check_sample_checks(sc, run, info, check, close):
    """Internal consistency of sample_checks.json and its agreement with the
    saved forecasts. Cannot re-read the 1-minute data: the flagged prices and
    the incomplete-candle classes are checked for coherence, not re-measured."""
    d = sc["daily_sample"]
    thr = d["flag_threshold_pct"]
    rows = d["flagged_minutes"]
    check(len(rows) == d["n_flags_unfiltered_sequence"], "sample checks: flag count")
    for r in rows:
        tag = f"sample checks flag {r['bar_start_utc']}"
        t, tp = pd.Timestamp(r["bar_start_utc"]), pd.Timestamp(r["prev_bar_start_utc"])
        check(abs(r["flag_log_return_pct"]) > thr, f"{tag}: |flag return| not above the threshold")
        check(abs(100 * np.log(r["close"] / r["prev_close"]) - r["flag_log_return_pct"]) < 1e-3, f"{tag}: flag return != log(close / prev close)")
        check(tp < t and close((t - tp) / pd.Timedelta("1min"), r["minutes_since_prev"], 1e-9), f"{tag}: predecessor time")
        check(r["consistent_with_threshold"] is True, f"{tag}: marked inconsistent")
        if r["bar_in_schedule"]:
            later = r["log_change_flag_prev_to_30_in_schedule_min_later_pct"]
            if not r["prev_in_schedule"] and abs(r["filtered_log_return_pct"]) < thr:
                cat = "out_of_schedule_predecessor"
            elif np.sign(later) == np.sign(r["flag_log_return_pct"]) and abs(later) > thr:
                cat = "move_persists_30min"
            else:
                cat = "unresolved"
            check(r["category"] == cat, f"{tag}: category {r['category']} != rule {cat}")
            check(not r["prev_in_schedule"] or r["filtered_prev_bar_start_utc"] == r["prev_bar_start_utc"],
                  f"{tag}: in-schedule predecessor but different filtered predecessor")
        else:
            check(r["category"] == "flagged_bar_out_of_schedule", f"{tag}: category")
    ins = [r for r in rows if r["bar_in_schedule"]]
    check(d["n_flags_in_schedule"] == len(ins) and d["n_flags_inconsistent_with_threshold"] == 0, "sample checks: in-schedule / inconsistent counts")
    check(d["flag_categories_in_schedule"] == pd.Series([r["category"] for r in ins]).value_counts().to_dict(), "sample checks: category counts")
    check(d["flagged_in_schedule_after_gap_over_60min"] == sum(r["minutes_since_prev"] > 60 for r in ins), "sample checks: gap count")
    # daily sample of THIS run vs the saved daily forecasts
    u = sc["daily_sample_used"]
    ps = pd.read_parquet(run / "predictions_session.parquet")
    test_days = pd.DatetimeIndex(ps.loc[ps.split == "test", "session_id"])
    flt = info["session"].get("flagged_bar_filter", True)
    check(u["flagged_bar_filter"] == flt, "sample checks: flag-filter setting differs from dataset_info")
    check(u["test_sessions_with_forecast"] == len(test_days) == info["session"]["test"][2], "sample checks: daily test rows differ from the saved forecasts")
    check(u["test_sessions"] == u["test_sessions_with_forecast"] + u["test_sessions_without_forecast"]
          == d["daily_sessions_by_split"]["test"], "sample checks: daily sample used does not add up")
    check(sum(u["reasons"].values()) == u["test_sessions_without_forecast"]
          and all(len(u["sessions_by_reason"][k]) == v for k, v in u["reasons"].items()), "sample checks: exclusion reasons")
    check(not set(pd.DatetimeIndex([s for v in u["sessions_by_reason"].values() for s in v])) & set(test_days),
          "sample checks: an excluded session has a saved forecast")
    check(flt or "flagged_bar_filter" not in u["reasons"], "sample checks: flag exclusions with the filter off")
    # the flagged-bar filter as originally applied (study.build): internal accounting
    a = d["daily_test_sessions_accounting"]
    check(a["test_sessions"] == a["in_data_set"] + a["not_in_data_set"]
          and a["not_in_data_set"] == a["not_in_data_set_and_in_50_session_window"] + a["not_in_data_set_outside_window"], "sample checks: daily accounting does not add up")
    w = d["daily_test_sessions_in_50_session_window_by_flag_category"]
    check(w["total_in_window"] == d["daily_rows_in_50_session_window_by_split"]["test"]
          == sum(v for k, v in w.items() if k != "total_in_window"), "sample checks: window categories do not add up")
    check(w["total_in_window"] == a["not_in_data_set_and_in_50_session_window"] + a["in_window_but_in_data_set"], "sample checks: window total")
    c = d["daily_test_sessions_not_in_data_set_by_condition (not exclusive)"]
    check(c["total"] == a["not_in_data_set"] and c["flagged_minute_in_last_50_sessions (study.build)"] == a["not_in_data_set_and_in_50_session_window"],
          "sample checks: exclusion conditions")
    # with the filter on, no saved daily forecast lies in the 50-session window after a flagged session
    # (session calendar = sessions seen in the 5-minute forecasts plus the daily ones)
    p5 = pd.read_parquet(run / "predictions_5min.parquet", columns=["session_id"])
    cal = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(p5.session_id)) | set(pd.DatetimeIndex(ps.session_id))))
    for r in (ins if flt else []):
        s = pd.Timestamp(r["session"])
        if cal[0] <= s <= cal[-1]:
            k = cal.searchsorted(s)
            check(not set(cal[k:k + 50]) & set(test_days), f"sample checks: daily forecast within 50 sessions of flag {r['session']}")
    # incomplete candles: the all-rows rescoring recomputed from the saved forecasts
    for freq, r in sc["incomplete_candles"].items():
        q = pd.read_parquet(run / f"predictions_{freq}.parquet")
        q = q[q.split == "test"]
        check(r["test_rows"] == len(q) and 0 <= r["test_rows_label_candle_incomplete"] <= len(q), f"sample checks {freq}: test rows")
        lv, ln, rr, y = q.lrv_next.to_numpy(), q.lrv_now.to_numpy(), q.ret_bps.to_numpy(), q.y3.to_numpy()
        s = r["test_rescored"]["all_rows"]
        for m in ("har_ols", "mlp_128-64-32", "xgboost"):
            f = q[f"vol__{m}"].to_numpy()
            check(abs(s[f"vol_R2_log_vs_persistence_pct_{m}"] - 100 * (1 - ((lv - f) ** 2).sum() / ((lv - ln) ** 2).sum())) < 6e-4, f"sample checks {freq}: vol {m}")
        call = np.where(q["dir__mlp_256-128-64-32"].to_numpy() >= 0.5, 1, -1)
        maj = info[freq]["consts"]["majority_direction"]
        check(abs(s["dir_gain_pp_mlp_256-128-64-32"] - 100 * ((call == y).mean() - (y == maj).mean())) < 6e-4, f"sample checks {freq}: direction")
        f = q["ret__mlp_256-128-64-32"].to_numpy()
        check(abs(s["ret_R2_pct_mlp_256-128-64-32"] - 100 * (1 - ((rr - f) ** 2).sum() / (rr ** 2).sum())) < 6e-4, f"sample checks {freq}: return")


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
