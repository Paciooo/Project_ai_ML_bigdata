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

# Gaps: more than 3 minutes without trades, outside the daily CME break
# (about 20:55-22:05 UTC) and the session open/close. NQ trades every few
# seconds, so a long gap inside the session usually means missing data.
print("\ngaps > 3 min inside the session (possible missing data):")
n_gaps = 0
for f in files:
    ts = pd.read_parquet(f, columns=["ts"])["ts"]
    if len(ts) < 2:
        continue
    gap = ts.diff()
    start = ts.shift()
    big = gap > pd.Timedelta(minutes=3)
    for t0, g in zip(start[big], gap[big]):
        t1 = t0 + g
        in_break = t0.hour >= 20 and t1.hour <= 22 and t1.date() == t0.date()
        if not in_break:
            n_gaps += 1
            print(f"  {f.stem}: {t0:%H:%M:%S} -> {t1:%H:%M:%S} UTC ({g.total_seconds() / 60:.0f} min)")
if n_gaps == 0:
    print("  none")
for f in partials:
    df = pd.read_parquet(f)
    last = f"up to {df.ts.max():%H:%M:%S} UTC" if len(df) else "just started"
    print(f"in progress: {f.stem.replace('.partial', '')}  {len(df):,} trades so far, {last}")
log_file = folder / "download.log"
if log_file.exists():
    lines = log_file.read_text(encoding="utf-8").splitlines()
    print(f"\nlast lines of {log_file}:")
    print("\n".join(lines[-5:]))
