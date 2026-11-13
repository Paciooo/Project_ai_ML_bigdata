"""
Threshold status of each walk-forward fold for runs made before study.py
recorded it (runs up to commit 35be806). Data check only: it rebuilds the
datasets and the fold boundaries, counts the executable validation rows of
every fold and fits NO model.

A validated threshold needs at least 200 executable validation trades
(strategy.choose_threshold). For Long & Short the unfiltered rule trades every
executable row, so a fold is

  ok                                if it has >= 200 executable validation rows,
  no_threshold_reaches_min_trades   if it has fewer (any direction).

Long-only / short-only folds with >= 200 rows are "unknown" (they depend on the
model's calls). Older code silently traded such folds unfiltered; report.py
treats them as not evaluable.

Writes <run>/walk_forward_status.csv (a new file; nothing is overwritten).

Usage: python tools/wf_threshold_status.py results/nq_study/runs/<run id>
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import experiment  # noqa: E402
import features  # noqa: E402
import study  # noqa: E402

MIN_TRADES = 200


def main(run):
    run = Path(run)
    out = run / "walk_forward_status.csv"
    if out.exists():
        raise SystemExit(f"{out} exists; not overwriting")
    rj = json.loads((run / "run.json").read_text())
    args = rj["args"]
    wf = pd.read_csv(run / "walk_forward.csv")
    minutes, *_ = experiment.load_minutes("nq", tuple(args["years"]) if args.get("years") else None)
    opens = minutes.groupby("session_id")["session_open"].first()
    session_open_of = lambda s: opens.loc[s]
    holdout_start = pd.Timestamp(config.CONFIRMATORY_HOLDOUT_START)
    usable = pd.DatetimeIndex(sorted(minutes["session_id"].unique()))
    usable = usable[usable < holdout_start]
    folds = features.walk_forward_folds(usable, "expanding", config.TRAIN_FRAC, config.VAL_FRAC, n_folds=4)
    rows = []
    for freq, g in wf.groupby("frequency", sort=False):
        _, _, meta, _, elig, _ = study.build(minutes, freq, args["latency"])
        strat = g.strategy.iloc[0]
        for k, (_, v0, t0, t1) in enumerate(folds):
            _, va, _ = features.purged_split(meta, v0, t0, session_open_of,
                                             end=t1 if t1 is not None else holdout_start)
            n = int((va & elig).sum())
            status = ("no_threshold_reaches_min_trades" if n < MIN_TRADES
                      else "ok" if strat == "long_short" else "unknown")
            rows.append({"frequency": freq, "fold": k, "strategy": strat, "val_start": str(v0.date()),
                         "val_end": str(t0.date()), "n_val_executable": n, "threshold_status": status})
            print(rows[-1], flush=True)
        del meta, elig
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1])
