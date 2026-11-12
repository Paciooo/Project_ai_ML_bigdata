"""
Recompute the direction majority class of a multitarget.py run made before
the fix (it counted dojis as "not up"). Rebuilds the data only (no model is
fitted) and writes consts.majority_direction (+ the old value and the UP share
among non-doji training candles) into dataset_info.json.

Usage: python tools/mt_fix_majority.py results/multitarget/runs/<run id>
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
import experiment  # noqa: E402
import features  # noqa: E402
import multitarget  # noqa: E402


def main(run):
    run = Path(run)
    fp = run / "dataset_info.json"
    info = json.loads(fp.read_text())
    args = json.loads((run / "run.json").read_text())["args"]
    minutes, *_ = experiment.load_minutes("nq", tuple(args["years"]) if args.get("years") else None)
    opens = minutes.groupby("session_id")["session_open"].first()
    holdout = pd.Timestamp(config.CONFIRMATORY_HOLDOUT_START)
    usable = pd.DatetimeIndex(sorted(minutes["session_id"].unique()))
    usable = usable[usable < holdout]
    (_, v0, t0, _), = features.walk_forward_folds(usable, "holdout", config.TRAIN_FRAC, config.VAL_FRAC)
    for freq in info:
        X, y3, meta, *_ = multitarget.build(minutes, freq)
        tr, _, _ = features.purged_split(meta, v0, t0, lambda s: opens.loc[s], end=holdout)
        tr_idx = np.flatnonzero(tr)
        if args.get("max_train_rows") and len(tr_idx) > args["max_train_rows"]:
            tr_idx = tr_idx[-args["max_train_rows"]:]
        y = y3.to_numpy()[tr_idx]
        if int(tr.sum()) != info[freq]["train"][2]:
            raise SystemExit(f"{freq}: rebuilt training rows {int(tr.sum())} != run {info[freq]['train'][2]}")
        c = info[freq]["consts"]
        c.setdefault("majority_direction_before_fix", c["majority_direction"])
        c["majority_direction"] = multitarget.majority_direction(y)
        c["train_up_share_nonflat"] = float((y[y != 0] > 0).mean())
        print(f"{freq}: majority {c['majority_direction_before_fix']} -> {c['majority_direction']} "
              f"(UP share of non-doji training candles {c['train_up_share_nonflat']:.4f})")
    fp.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
