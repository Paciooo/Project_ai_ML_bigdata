"""
Summarise a study.py run: tables, the pre-specified strategy choice, charts and
the data file the slide deck is built from (presentation/deck_data.json).

Selection rule (fixed before the final results were read):
  1. Per candle length, the candidate is the model / direction / threshold with
     the highest VALIDATION net P&L after $5 round-trip costs (rules excluded).
  2. The overall candidate is the one with the highest validation net P&L.
  3. It is called deployable only if, on the TEST period, net P&L > 0 at $5 AND
     at $12.50 per round trip, the 95% interval of mean daily net P&L is above 0,
     and at least 3 of 4 walk-forward years are profitable. A walk-forward year
     whose validation block allowed no threshold (fewer than 200 executable
     trades) is "not evaluable" for a filtered strategy and does not count as
     profitable.
Test results of every other configuration are shown descriptively only: they
come from a search over many configurations and are not independent tests.

Usage: python report.py results/nq_study/runs/<run id>
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RULES = ("always_long", "always_short", "reversal")


def load(run):
    r = {k: pd.read_csv(run / f"{k}.csv") for k in
         ("model_selection", "accuracy_test", "strategies", "equity_daily", "cost_sensitivity",
          "threshold_sensitivity", "walk_forward")}
    r["info"] = json.loads((run / "dataset_info.json").read_text())
    r["run_json"] = json.loads((run / "run.json").read_text())
    r["run_id"] = run.name
    r["run_dir"] = run
    wf = r["walk_forward"]
    if "threshold_status" not in wf:
        # runs before the status was recorded: tools/wf_threshold_status.py writes it from the data
        sp = run / "walk_forward_status.csv"
        if sp.exists():
            st = pd.read_csv(sp)[["frequency", "fold", "threshold_status", "n_val_executable"]]
            wf = wf.merge(st, on=["frequency", "fold"], how="left")
        elif (wf.threshold_kind == "validated").any():
            raise SystemExit(f"{run.name}: walk-forward threshold status not recorded (run made before the fix). "
                             f"Run first: python tools/wf_threshold_status.py {run}")
        else:
            wf = wf.assign(threshold_status="ok", n_val_executable=np.nan)
        if wf.loc[wf.threshold_kind == "validated", "threshold_status"].isna().any():
            raise SystemExit(f"{run.name}: walk_forward_status.csv does not cover every fold; rerun tools/wf_threshold_status.py")
        wf.loc[wf.threshold_kind == "none", "threshold_status"] = "ok"
    r["walk_forward"] = wf
    return r


def wf_evaluable(w):
    return (w.threshold_kind == "none") | (w.threshold_status == "ok")


def always_long_on(run, freq, windows, cost):
    """Net P&L of 'always long' on exactly the walk-forward test sessions (same sample)."""
    p = pd.read_parquet(run / f"test_predictions_{freq}.parquet", columns=["session_id", "move_points", "execution_eligible"])
    sid = pd.to_datetime(p.session_id)
    tot = 0.0
    for a, b in windows:
        m = (sid >= pd.Timestamp(a)) & (sid < (pd.Timestamp(b) if b != "end" else pd.Timestamp.max)) & p.execution_eligible
        tot += float((p.move_points[m] * 20.0 - cost).sum())
    return tot


def model_table(r):
    sel, acc = r["model_selection"], r["accuracy_test"].set_index(["frequency", "model"])
    rows = []
    for (f, fam), g in sel.groupby(["frequency", "family"], sort=False):
        o = g[g.original].iloc[0]
        b = g[g.selected_optimised].iloc[0]
        ao, ab = acc.loc[(f, o.model)], acc.loc[(f, b.model)]
        rows.append({"frequency": f, "family": fam, "original": o.model, "optimised": b.model,
                     "orig_acc_all": ao.acc_all, "opt_acc_all": ab.acc_all,
                     "orig_acc_nonflat": ao.acc_nonflat, "opt_acc_nonflat": ab.acc_nonflat,
                     "orig_bal": ao.balanced_acc_nonflat, "opt_bal": ab.balanced_acc_nonflat,
                     "improvement_pp": 100 * (ab.acc_all - ao.acc_all),
                     "opt_vs_majority_pp": ab.vs_majority_pp, "opt_vs_majority_ci": [ab.vs_majority_ci_lo, ab.vs_majority_ci_hi],
                     "opt_val_bal": b.val_balanced_acc, "orig_val_bal": o.val_balanced_acc})
    return pd.DataFrame(rows)


def candidates(r):
    s = r["strategies"]
    key = ["frequency", "model", "strategy", "threshold_kind"]
    val = s[(s.split == "val") & ~s.model.isin(RULES)]
    if "economic_status" in val:      # samples without any executable entry are never candidates
        val = val[val.economic_status == "evaluated"]
    out = []
    for f, g in val.groupby("frequency", sort=False):
        best = g.sort_values("net_usd", ascending=False).iloc[0]
        row = {"frequency": f, "model": best.model, "strategy": best.strategy,
               "threshold_kind": best.threshold_kind, "threshold": best.threshold, "val_net_usd": best.net_usd}
        for split in ("train", "test"):
            t = s[(s.split == split) & (s[key] == best[key]).all(axis=1)].iloc[0]
            for c in ("net_usd", "n_trades", "win_rate", "profit_factor", "max_drawdown_usd", "daily_net_ci_lo",
                      "daily_net_ci_hi", "sharpe_daily_ann", "avg_net_per_trade_usd", "gross_usd"):
                row[f"{split}_{c}"] = t.get(c, np.nan)
        cs = r["cost_sensitivity"]
        c = cs[(cs[key] == best[key]).all(axis=1)].set_index("rt_cost")["net_usd"]
        row.update({f"test_net_at_{k:g}": float(v) for k, v in c.items()})
        wf = r["walk_forward"]
        w = wf[(wf.frequency == f) & (wf.threshold_kind == best.threshold_kind)]
        ev = w[wf_evaluable(w)]
        row["wf_folds"] = int(len(w))
        row["wf_years"] = int(len(ev))                       # evaluable folds
        row["wf_not_evaluable"] = int(len(w) - len(ev))
        row["wf_years_positive"] = int((ev.net_usd > 0).sum())
        row["wf_net_usd"] = float(ev.net_usd.sum()) if len(ev) else np.nan
        for c in ("gross_usd", "gross_from_direction_mix_usd", "gross_from_timing_usd", "n_long", "n_short", "n_trades"):
            row[f"wf_{c}"] = float(ev[c].sum()) if len(ev) else np.nan
        row["wf_always_long_same_sessions_usd"] = (always_long_on(r["run_dir"], f, list(zip(ev.test_start, ev.test_end)),
                                                                  r["run_json"]["args"]["cost"]) if len(ev) else np.nan)
        wu = wf[(wf.frequency == f) & (wf.threshold_kind == "none")]
        row["wf_unfiltered_net_usd"] = float(wu.net_usd.sum())
        for c in ("gross_usd", "gross_from_direction_mix_usd", "gross_from_timing_usd"):
            row[f"wf_unfiltered_{c}"] = float(wu[c].sum())
        row["wf_unfiltered_always_long_same_sessions_usd"] = always_long_on(
            r["run_dir"], f, list(zip(wu.test_start, wu.test_end)), r["run_json"]["args"]["cost"])
        row["deployable"] = bool(row["test_net_usd"] > 0 and row.get("test_net_at_12.5", -1) > 0
                                 and row["test_daily_net_ci_lo"] > 0 and row["wf_years_positive"] >= 3
                                 and row["wf_folds"] == 4)
        out.append(row)
    c = pd.DataFrame(out)
    c["overall_choice"] = c.val_net_usd == c.val_net_usd.max()
    return c


def charts(r, choice, run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    eq = r["equity_daily"]
    f, m = choice.frequency, choice.model
    fig, ax = plt.subplots(figsize=(11, 5))
    for st, color in (("long_short", "#8E1728"), ("long_only", "#B8964B"), ("short_only", "#595959")):
        e = eq[(eq.frequency == f) & (eq.model == m) & (eq.strategy == st) & (eq.threshold_kind == "none")]
        ax.plot(pd.to_datetime(e.session), e.cum_net_usd / 1e3, label=st.replace("_", " "), color=color, lw=1.8)
    e = eq[(eq.frequency == f) & (eq.model == "always_long") & (eq.strategy == "long_only")]
    ax.plot(pd.to_datetime(e.session), e.cum_net_usd / 1e3, label="always long (benchmark)", color="#A6A6A6", lw=1.2, ls="--")
    ax.axhline(0, color="black", lw=0.8)
    d = pd.to_datetime(e.session)
    ax.set(title=f"NQ {f}, {m}: net equity after $5 round-trip costs (test {d.min():%Y-%m} to {d.max():%Y-%m})",
           ylabel="cumulative net P&L, thousand USD (1 contract)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run / "equity_curves.png", dpi=150)
    plt.close(fig)


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def deck_data(r, mt, cand, choice):
    """Everything the slides show, taken from this run's CSVs only."""
    s, eq, acc, sel = r["strategies"], r["equity_daily"], r["accuracy_test"], r["model_selection"]
    f = choice.frequency
    opt_models = list(mt[mt.frequency == f].optimised)
    show = opt_models + ["reversal", "always_long"]
    strat = s[(s.split == "test") & (s.frequency == f) & (s.threshold_kind == "none") & s.model.isin(show)]
    weekly = {}
    for st in ("long_short", "long_only", "short_only"):
        e = eq[(eq.frequency == f) & (eq.model == choice.model) & (eq.strategy == st) & (eq.threshold_kind == "none")]
        w = e.assign(session=pd.to_datetime(e.session)).set_index("session").cum_net_usd.resample("W").last().ffill()
        weekly[st] = w
    e = eq[(eq.frequency == f) & (eq.model == "always_long") & (eq.strategy == "long_only")]
    weekly["always_long"] = e.assign(session=pd.to_datetime(e.session)).set_index("session").cum_net_usd.resample("W").last().ffill()
    e = eq[(eq.frequency == f) & (eq.model == choice.model) & (eq.strategy == choice.strategy)
           & (eq.threshold_kind == choice.threshold_kind)]
    weekly["chosen"] = e.assign(session=pd.to_datetime(e.session)).set_index("session").cum_net_usd.resample("W").last().ffill()
    wk = pd.DataFrame(weekly).ffill().fillna(0.0)
    key = ["frequency", "model", "strategy", "threshold_kind"]
    ch = s[(s[key] == choice[key]).all(axis=1)]
    opt = set(mt.optimised) | set(RULES)
    grid = s[(s.split == "test") & s.model.isin(opt)][key + ["n_trades", "gross_usd", "net_usd", "win_rate",
                                                             "profit_factor", "max_drawdown_usd"]]
    cc = r["cost_sensitivity"]
    cand_cost = cc.merge(cand[key], on=key)
    test_ci_pos = s[(s.split == "test") & ~s.model.isin(RULES) & (s.daily_net_ci_lo > 0)][
        key + ["threshold", "n_trades", "net_usd", "daily_net_ci_lo"]].sort_values("daily_net_ci_lo", ascending=False)
    nn = sel[sel.family == "mlp"].merge(acc[["frequency", "model", "acc_all", "acc_nonflat", "balanced_acc_nonflat"]],
                                        on=["frequency", "model"])
    return _clean({
        "run_id": r["run_id"], "run": r["run_json"].get("git", {}), "args": r["run_json"]["args"], "info": r["info"],
        "merged_from": r["run_json"].get("merged_from", []),
        "primary_frequency": f,
        "models_primary": mt[mt.frequency == f].to_dict("records"),
        "models_all": mt.to_dict("records"),
        "accuracy_test": acc.to_dict("records"),
        "selection": sel.to_dict("records"),
        "strategies_primary": strat.to_dict("records"),
        "strategies_all_test": s[(s.split == "test") & (s.threshold_kind == "none")].to_dict("records"),
        "equity_weekly": {"dates": [d.strftime("%Y-%m-%d") for d in wk.index],
                          **{c: list(wk[c].round(1)) for c in wk.columns}},
        "candidates": cand.to_dict("records"),
        "choice": choice.to_dict(),
        "choice_splits": ch.to_dict("records"),
        "walk_forward": r["walk_forward"].to_dict("records"),
        "cost_sensitivity": r["cost_sensitivity"][(r["cost_sensitivity"][key] == choice[key]).all(axis=1)].to_dict("records"),
        "candidate_costs": cand_cost.to_dict("records"),
        "strategy_grid_test": grid.to_dict("records"),
        "nn": nn.to_dict("records"),
        "n_val_configurations": int(len(s[(s.split == "val") & ~s.model.isin(RULES)])),
        "n_models_per_frequency": int(sel.groupby("frequency").size().max()),
        "wf_always_long_by_fold": [{"fold": int(w.fold), "net_usd": always_long_on(r["run_dir"], f, [(w.test_start, w.test_end)],
                                                                             r["run_json"]["args"]["cost"])}
                                  for w in r["walk_forward"][(r["walk_forward"].frequency == f)
                                                             & (r["walk_forward"].threshold_kind == "none")].itertuples()],
        "test_ci_positive_descriptive": test_ci_pos.to_dict("records"),
        "best_test_accuracy_any_model": acc[~acc.model.isin(RULES)].sort_values("vs_majority_pp", ascending=False)
                                         .head(1).to_dict("records"),
    })


def main(run, deck_json=None):
    run = Path(run)
    r = load(run)
    mt = model_table(r)
    cand = candidates(r)
    choice = cand[cand.overall_choice].iloc[0]
    charts(r, choice, run)
    mt.to_csv(run / "summary_models.csv", index=False)
    cand.to_csv(run / "summary_candidates.csv", index=False)
    d = deck_data(r, mt, cand, choice)
    (run / "deck_data.json").write_text(json.dumps(d, indent=1, default=str), encoding="utf-8")
    if deck_json:
        Path(deck_json).write_text(json.dumps(d, indent=1, default=str), encoding="utf-8")
    print(mt.round(4).to_string())
    print(cand.round(2).T.to_string())
    return r, mt, cand, choice


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
