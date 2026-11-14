"""Merge a daily-only multitarget run into a copy of a full run.

The intraday timeframes (5min .. 1h) are copied unchanged from the base run;
the daily ("session") predictions, model rows and dataset info come from the
daily run. Neither input folder is modified. The merged folder records both
sources in run.json.

Usage:
  python tools/merge_daily_run.py results/multitarget/runs/<base> results/multitarget/runs/<daily> [tag]
"""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd


def main(base, daily, tag="merged"):
    base, daily = Path(base), Path(daily)
    out = base.parent / f"{daily.name}_{tag}"
    if out.exists():
        raise SystemExit(f"{out} exists: not overwritten")
    out.mkdir()
    bi = json.loads((base / "dataset_info.json").read_text())
    di = json.loads((daily / "dataset_info.json").read_text())
    if set(di) != {"session"}:
        raise SystemExit(f"daily run must contain only 'session', has {sorted(di)}")
    info = {f: (di["session"] if f == "session" else v) for f, v in bi.items()}
    for f in info:
        src = daily if f == "session" else base
        shutil.copy2(src / f"predictions_{f}.parquet", out / f"predictions_{f}.parquet")
    bm, dm = pd.read_csv(base / "models.csv"), pd.read_csv(daily / "models.csv")
    models = pd.concat([bm[bm.frequency != "session"], dm], ignore_index=True)
    models.to_csv(out / "models.csv", index=False)
    (out / "dataset_info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    run = {"merged": True, "intraday_from": base.name, "daily_from": daily.name,
           "base_run_json": json.loads((base / "run.json").read_text()),
           "daily_run_json": json.loads((daily / "run.json").read_text())}
    (out / "run.json").write_text(json.dumps(run, indent=2, default=str), encoding="utf-8")
    (out / "console.log").write_text(
        (base / "console.log").read_text(encoding="utf-8") + "\n=== daily run ===\n"
        + (daily / "console.log").read_text(encoding="utf-8"), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
