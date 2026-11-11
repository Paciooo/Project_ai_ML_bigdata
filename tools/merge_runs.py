"""
Combine study.py runs that cover different candle lengths into one run folder,
so report.py / verify.py treat them as a single study (nothing is recomputed).

Used for the final deck: the intraday run (1min 5min 15min 1h) plus a later
daily-only run (--freqs session) with identical settings. The source folders
are left untouched; a candle length present in two sources is an error.

Usage: python tools/merge_runs.py <out run folder> <run A> <run B> [...]
"""

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

CSVS = ("model_selection", "accuracy_test", "strategies", "equity_daily", "cost_sensitivity",
        "threshold_sensitivity", "walk_forward")
SAME_ARGS = ("latency", "cost", "max_train_rows", "seed", "quick", "years")


def main(out, *runs):
    out, runs = Path(out), [Path(r) for r in runs]
    if out.exists():
        sys.exit(f"{out} exists; choose a new folder")
    metas = [json.loads((r / "run.json").read_text()) for r in runs]
    for k in SAME_ARGS:
        vals = {json.dumps(m["args"].get(k)) for m in metas}
        if len(vals) > 1:
            sys.exit(f"runs differ in '{k}': {vals}")
    info, seen = {}, set()
    for r in runs:
        i = json.loads((r / "dataset_info.json").read_text())
        dup = seen & set(i)
        if dup:
            sys.exit(f"candle length in two runs: {dup}")
        seen |= set(i)
        info.update(i)
    out.mkdir(parents=True)
    for k in CSVS:
        pd.concat([pd.read_csv(r / f"{k}.csv") for r in runs], ignore_index=True).to_csv(out / f"{k}.csv", index=False)
    for r in runs:
        for p in r.glob("test_predictions_*.parquet"):
            shutil.copy2(p, out / p.name)
    (out / "dataset_info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    m = dict(metas[0])
    m["args"] = dict(m["args"], freqs=[f for mm in metas for f in mm["args"]["freqs"]])
    m["merged_from"] = [{"run": r.name, "freqs": mm["args"]["freqs"], "git": mm.get("git", {})}
                        for r, mm in zip(runs, metas)]
    (out / "run.json").write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")
    print(f"merged {len(runs)} runs ({sorted(seen)}) into {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
