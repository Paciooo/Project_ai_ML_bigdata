"""
Summary of the tick files downloaded by download_ticks_ibkr.py.

Usage:
  python tools/check_ticks.py                      # default folder data_raw/ticks_ibkr/NQZ6
  python tools/check_ticks.py data_raw/ticks_ibkr/NQU6
"""

import sys
from pathlib import Path

import pandas as pd

folder = Path(sys.argv[1] if len(sys.argv) > 1 else "data_raw/ticks_ibkr/NQZ6")
files = sorted(f for f in folder.glob("*.parquet") if not f.stem.endswith(".partial"))
partials = sorted(folder.glob("*.partial.parquet"))
if not files and not partials:
    raise SystemExit(f"No files in {folder}")

print(f"{'day (UTC)':10s} {'trades':>10s}  {'first':>8s}  {'last':>8s}  {'min price':>10s}  {'max price':>10s}")
total = 0
for f in files:
    df = pd.read_parquet(f)
    total += len(df)
    if df.empty:
        print(f"{f.stem:10s} {0:>10,}  (no trades: market closed)")
        continue
    print(f"{f.stem:10s} {len(df):>10,}  {df.ts.min():%H:%M:%S}  {df.ts.max():%H:%M:%S}  "
          f"{df.price.min():>10.2f}  {df.price.max():>10.2f}")
print(f"\n{len(files)} days, {total:,} trades in total")
for f in partials:
    df = pd.read_parquet(f)
    last = f"up to {df.ts.max():%H:%M:%S} UTC" if len(df) else "just started"
    print(f"in progress: {f.stem.replace('.partial', '')}  {len(df):,} trades so far, {last}")
log_file = folder / "download.log"
if log_file.exists():
    lines = log_file.read_text(encoding="utf-8").splitlines()
    print(f"\nlast lines of {log_file}:")
    print("\n".join(lines[-5:]))
