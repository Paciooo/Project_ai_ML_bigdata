"""
Cross-check of a study.py run, recomputed from the saved test predictions
without importing strategy.py.

What it checks (test period only):
  * run settings: $5 per round trip; latency 0 (otherwise the traded return is
    entry -> close, not the predicted body, and that must be flagged);
  * predictions: probabilities finite and in [0, 1], moves finite, labels in
    {-1, 0, 1}; each traded move has the sign of the predicted body;
  * entry timing, when the run saved it: entry >= decision time, entry >= the
    open of the predicted candle; labels in time order, label session >=
    signal session;
  * accuracy of every model (accuracy_test.csv);
  * for every test strategy (strategies.csv): trades, longs, shorts, gross and
    net P&L, win rate, profit factor, maximum drawdown (on SESSION-END equity,
    starting from 0) and the ordering of the daily-P&L interval; no non-finite
    values where a number is required;
  * a strategy whose sample has no executable entry is marked
    not_evaluable_no_entry_before_exit with N/A P&L and no equity curve, and a
    strategy with P&L has at least one executable entry;
  * every evaluated test strategy has an equity curve, one point per traded session,
    ending at its net P&L;
  * cost sensitivity at every cost level;
  * deck_data.json: same run, candidates' test P&L, walk-forward totals equal
    the evaluable walk-forward rows, no VIX.

What it does NOT do: it does not rebuild candles, features or labels from the
raw 1-minute data, refit models, re-run the bootstrap, or check the training /
validation numbers, the walk-forward refits or the text of the slides. Those
rest on the pipeline code and its tests (tests/).

Exits with status 1 on any problem.

Usage: python verify.py results/nq_study/runs/<run id> [presentation/deck_data.json]
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MULT, COST = 20.0, 5.0
TOL = 0.5          # USD
REQUIRED = ("n_trades", "n_long", "n_short", "gross_usd", "net_usd", "max_drawdown_usd")


def recompute(pos, mv, sess, cost):
    gross = pos * mv * MULT
    net = gross - cost * (pos != 0)
    t = pos != 0
    tn = net[t]
    daily = pd.Series(net).groupby(sess).sum().sort_index()
    eq = np.r_[0.0, daily.cumsum().to_numpy()]
    wins, losses = tn[tn > 0], tn[tn <= 0]
    return {"n_trades": int(t.sum()), "n_long": int((pos > 0).sum()), "n_short": int((pos < 0).sum()),
            "gross_usd": float(gross[t].sum()), "net_usd": float(tn.sum()),
            "win_rate": float((tn > 0).mean()) if t.any() else np.nan,
            "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else np.nan,
            "max_drawdown_usd": float((eq - np.maximum.accumulate(eq)).min())}


def same(a, b, tol):
    if pd.isna(a) and pd.isna(b):
        return True
    return not (pd.isna(a) or pd.isna(b)) and abs(float(a) - float(b)) <= tol


def main(run, deck=None):
    run = Path(run)
    rj = json.loads((run / "run.json").read_text())
    args = rj["args"]
    info = json.loads((run / "dataset_info.json").read_text())
    acc = pd.read_csv(run / "accuracy_test.csv")
    st = pd.read_csv(run / "strategies.csv")
    eq = pd.read_csv(run / "equity_daily.csv")
    cs = pd.read_csv(run / "cost_sensitivity.csv")
    problems, checked, notes = [], 0, []

    def check(ok, msg):
        nonlocal checked
        checked += 1
        if not ok:
            problems.append(msg)

    check(abs(args["cost"] - COST) < 1e-9, f"run used ${args['cost']} per round trip, not $5")
    zero_latency = pd.Timedelta(args["latency"]) == pd.Timedelta(0)
    if not zero_latency:
        notes.append(f"latency {args['latency']}: traded return is entry -> close, not the candle body")
    for f in acc.frequency.unique():
        p = pd.read_parquet(run / f"test_predictions_{f}.parquet")
        y, mv, el = p.y3.to_numpy(), p.move_points.to_numpy(float), p.execution_eligible.to_numpy(bool)
        sess = p.label_session_id.to_numpy()
        check(np.isfinite(mv).all(), f"{f}: non-finite moves")
        check(set(np.unique(y)) <= {-1, 0, 1}, f"{f}: labels outside -1/0/1")
        if zero_latency:
            check(not (np.sign(mv[el]) != y[el]).any(), f"{f}: traded move differs from the predicted body direction")
        else:
            check(info.get(f, {}).get("traded_return") == "delayed_entry_to_close",
                  f"{f}: positive latency but the run does not flag the delayed-entry return")
        # time provenance of entries
        ls = pd.to_datetime(p.label_start)
        check(ls.is_monotonic_increasing, f"{f}: labels not in time order")
        check((pd.to_datetime(p.label_session_id) >= pd.to_datetime(p.session_id)).all(), f"{f}: label session before signal session")
        if {"decision_at", "entry_at"} <= set(p.columns):
            ea, da = pd.to_datetime(p.entry_at[el]), pd.to_datetime(p.decision_at[el])
            check(ea.notna().all() and (ea >= da).all(), f"{f}: an entry precedes its decision time")
            check((ea >= ls[el]).all(), f"{f}: an entry precedes the predicted candle's open")
        else:
            notes.append(f"{f}: entry/decision times not saved by this run (older format); timing rests on tests/")
        for _, a in acc[acc.frequency == f].iterrows():
            pr = p[f"p_{a.model}"].to_numpy(np.float64)
            check(np.isfinite(pr).all() and ((pr >= 0) & (pr <= 1)).all(), f"{f} {a.model}: probability not finite or outside [0, 1]")
            mine = float((np.where(pr >= 0.5, 1, -1) == y).mean())
            check(abs(mine - a.acc_all) <= 2e-5, f"{f} {a.model}: accuracy {mine:.6f} vs {a.acc_all:.6f}")
        n_sess = len(np.unique(sess[el]))
        for _, s in st[(st.frequency == f) & (st.split == "test")].iterrows():
            tag = f"{f} {s.model} {s.strategy} {s.threshold_kind}"
            if s.get("economic_status", "evaluated") != "evaluated":
                # no executable entry at all: P&L must be N/A, not a silent zero, and no equity curve
                check(not el.any(), f"{tag}: marked not evaluable but has executable entries")
                check(all(pd.isna(s[c]) for c in ("net_usd", "gross_usd", "max_drawdown_usd")),
                      f"{tag}: not evaluable but reports P&L")
                check(eq[(eq.frequency == f) & (eq.model == s.model) & (eq.strategy == s.strategy)
                         & (eq.threshold_kind == s.threshold_kind)].empty, f"{tag}: not evaluable but has an equity curve")
                continue
            check(el.any(), f"{tag}: P&L reported although no row has an executable entry")
            check(all(np.isfinite(s[c]) for c in REQUIRED), f"{tag}: non-finite required metric")
            pr = p[f"p_{s.model}"].to_numpy(np.float64)
            side = np.where(pr >= 0.5, 1, -1)
            on = (np.abs(pr - 0.5) >= s.threshold) & el
            if s.strategy == "long_only":
                on &= side > 0
            elif s.strategy == "short_only":
                on &= side < 0
            pos = np.where(on, side, 0)
            m = recompute(pos, mv, sess, COST)
            for c, tol in (("n_trades", 0), ("n_long", 0), ("n_short", 0), ("gross_usd", TOL), ("net_usd", TOL),
                           ("win_rate", 1e-9), ("profit_factor", 1e-6), ("max_drawdown_usd", TOL)):
                check(same(m[c], s[c], tol), f"{tag}: {c} {m[c]} vs {s[c]}")
            if "daily_net_ci_lo" in s and pd.notna(s.daily_net_ci_lo):
                check(s.daily_net_ci_lo <= s.mean_daily_net_usd <= s.daily_net_ci_hi, f"{tag}: daily-P&L interval out of order")
            e = eq[(eq.frequency == f) & (eq.model == s.model) & (eq.strategy == s.strategy) & (eq.threshold_kind == s.threshold_kind)]
            check(len(e) > 0, f"{tag}: equity curve missing")
            if len(e):
                check(np.isfinite(e.cum_net_usd).all(), f"{tag}: non-finite equity")
                check(len(e) == n_sess, f"{tag}: equity has {len(e)} sessions, test has {n_sess}")
                check(same(e.cum_net_usd.iloc[-1], s.net_usd, TOL), f"{tag}: equity end {e.cum_net_usd.iloc[-1]:,.1f} vs net {s.net_usd:,.1f}")
            for _, c in cs[(cs.frequency == f) & (cs.model == s.model) & (cs.strategy == s.strategy)
                           & (cs.threshold_kind == s.threshold_kind)].iterrows():
                n_c = float(((pos * mv * MULT) - c.rt_cost * (pos != 0)).sum())
                check(same(n_c, c.net_usd, TOL), f"{tag}: net at ${c.rt_cost} {n_c:,.1f} vs {c.net_usd:,.1f}")
    if deck:
        d = json.loads(Path(deck).read_text(encoding="utf-8"))
        check(d["run_id"] == run.name, "deck_data.json comes from a different run")
        wf = pd.DataFrame(d["walk_forward"])
        for c in d["candidates"]:
            row = st[(st.split == "test") & (st.frequency == c["frequency"]) & (st.model == c["model"])
                     & (st.strategy == c["strategy"]) & (st.threshold_kind == c["threshold_kind"])].iloc[0]
            check(same(row.net_usd, c["test_net_usd"], TOL), f"deck candidate {c['frequency']}: {c['test_net_usd']} vs {row.net_usd}")
            w = wf[(wf.frequency == c["frequency"]) & (wf.threshold_kind == c["threshold_kind"])]
            ev = w[(w.threshold_kind == "none") | (w.threshold_status == "ok")]
            check(len(ev) == c["wf_years"] and int((ev.net_usd > 0).sum()) == c["wf_years_positive"]
                  and same(ev.net_usd.sum(), c["wf_net_usd"], TOL), f"deck candidate {c['frequency']}: walk-forward totals")
        check("vix" not in json.dumps(d).lower(), "deck data mentions VIX")
    print(f"{checked} checks, {len(problems)} problems")
    for n in sorted(set(notes)):
        print("  note: " + n)
    for p in problems[:50]:
        print("  " + p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
