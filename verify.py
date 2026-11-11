"""
Independent cross-check of a study.py run (Phase 9 validation).

Recomputes from the saved test predictions, without importing strategy.py:
  * accuracy of every model (vs accuracy_test.csv);
  * trades, long/short counts and net P&L of every test strategy
    (vs strategies.csv), with NQ = $20 per point and $5 per round trip;
  * the final value of every equity curve (vs net P&L);
  * that each traded move is close(t+1) - open(t+1), the predicted target;
  * that presentation/deck_data.json repeats the same numbers.
Exits with status 1 on any mismatch.

Usage: python verify.py results/nq_study/runs/<run id> [presentation/deck_data.json]
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MULT, COST = 20.0, 5.0


def main(run, deck=None):
    run = Path(run)
    args = json.loads((run / "run.json").read_text())["args"]
    assert abs(args["cost"] - COST) < 1e-9, "run did not use $5 per round trip"
    acc = pd.read_csv(run / "accuracy_test.csv")
    st = pd.read_csv(run / "strategies.csv")
    eq = pd.read_csv(run / "equity_daily.csv")
    problems, checked = [], 0
    for f in acc.frequency.unique():
        p = pd.read_parquet(run / f"test_predictions_{f}.parquet")
        y, mv, el = p.y3.to_numpy(), p.move_points.to_numpy(), p.execution_eligible.to_numpy(bool)
        if (np.sign(mv[el]) != y[el]).any():
            problems.append(f"{f}: traded move differs from the predicted body direction")
        for _, a in acc[acc.frequency == f].iterrows():
            pr = p[f"p_{a.model}"].to_numpy(np.float64)
            mine = float((np.where(pr >= 0.5, 1, -1) == y).mean())
            checked += 1
            if abs(mine - a.acc_all) > 2e-5:
                problems.append(f"{f} {a.model}: accuracy {mine:.6f} vs {a.acc_all:.6f}")
        for _, s in st[(st.frequency == f) & (st.split == "test")].iterrows():
            pr = p[f"p_{s.model}"].to_numpy(np.float64)
            side = np.where(pr >= 0.5, 1, -1)
            on = (np.abs(pr - 0.5) >= s.threshold) & el
            if s.strategy == "long_only":
                on &= side > 0
            elif s.strategy == "short_only":
                on &= side < 0
            pos = np.where(on, side, 0)
            net = float(((pos * mv * MULT) - COST * (pos != 0)).sum())
            checked += 1
            if (pos != 0).sum() != s.n_trades or (pos > 0).sum() != s.n_long or abs(net - s.net_usd) > 0.5:
                problems.append(f"{f} {s.model} {s.strategy} {s.threshold_kind}: trades {(pos != 0).sum()} vs {s.n_trades}, "
                                f"net {net:,.1f} vs {s.net_usd:,.1f}")
            e = eq[(eq.frequency == f) & (eq.model == s.model) & (eq.strategy == s.strategy) & (eq.threshold_kind == s.threshold_kind)]
            if len(e) and abs(e.cum_net_usd.iloc[-1] - s.net_usd) > 0.5:
                problems.append(f"{f} {s.model} {s.strategy}: equity end {e.cum_net_usd.iloc[-1]:,.1f} vs net {s.net_usd:,.1f}")
    if deck:
        d = json.loads(Path(deck).read_text(encoding="utf-8"))
        assert d["run_id"] == run.name, "deck_data.json comes from a different run"
        for c in d["candidates"]:
            row = st[(st.split == "test") & (st.frequency == c["frequency"]) & (st.model == c["model"])
                     & (st.strategy == c["strategy"]) & (st.threshold_kind == c["threshold_kind"])].iloc[0]
            checked += 1
            if abs(row.net_usd - c["test_net_usd"]) > 0.5:
                problems.append(f"deck candidate {c['frequency']}: {c['test_net_usd']} vs {row.net_usd}")
        text = json.dumps(d).lower()
        if "vix" in text:
            problems.append("deck data mentions VIX")
    print(f"{checked} checks, {len(problems)} problems")
    for p in problems[:50]:
        print("  " + p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
