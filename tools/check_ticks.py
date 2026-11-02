"""
Summary and verification of the tick files from download_ticks_ibkr.py.

Reads manifest.json: a day counts as verified only if the manifest says
"complete" (coverage check against IB's 1-minute bars passed) or
"no_trades". Files without a manifest entry are reported as UNVERIFIED.
Gaps over 3 minutes inside the CME session are an extra diagnostic, not a
proof of completeness.

Usage:
  python tools/check_ticks.py                      # default folder data_raw/ticks_ibkr/NQZ6
  python tools/check_ticks.py data_raw/ticks_ibkr/NQU6
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import timeframes  # noqa: E402


def session_gaps(ts, minutes=3):
    """Gaps > `minutes` between trades of the same CME session (schedule-aware)."""
    ts = pd.DatetimeIndex(ts)
    ts = ts.tz_localize("UTC") if ts.tz is None else ts
    cal = timeframes.session_calendar(ts.floor("1min"))
    s = pd.Series(ts[cal["in_schedule"].to_numpy()])
    sid = cal["session_id"][cal["in_schedule"]].reset_index(drop=True)
    gap = s.diff()
    big = (gap > pd.Timedelta(minutes=minutes)) & (sid == sid.shift())
    return [(s[i - 1], s[i]) for i in big[big].index]


def main(folder):
    folder = Path(folder)
    files = sorted(folder.glob("????-??-??.parquet"))
    mf = folder / "manifest.json"
    man = json.loads(mf.read_text()) if mf.exists() else {"days": {}}
    days = man.get("days", {})
    if not files and not days:
        raise SystemExit(f"No files in {folder}")
    print(f"contract: {man.get('contract') or 'unknown (no manifest)'}\n")
    print(f"{'day (UTC)':10s} {'status':16s} {'trades':>10s} {'vol ratio':>9s} {'miss.min':>8s}")
    total, n_ver = 0, 0
    keys = sorted(set(f.stem for f in files) | set(days))
    for k in keys:
        e = days.get(k, {})
        st = e.get("status") or ("UNVERIFIED" if (folder / f"{k}.parquet").exists() else "missing")
        n = e.get("n_ticks")
        if n is None and (folder / f"{k}.parquet").exists():
            n = len(pd.read_parquet(folder / f"{k}.parquet", columns=["ts"]))
        n_ver += st in ("complete", "no_trades")
        total += n or 0
        vr = e.get("volume_ratio")
        print(f"{k:10s} {st:16s} {n or 0:>10,} {vr if vr is None else round(vr, 4)!s:>9s} "
              f"{e.get('n_missing_minutes', '-')!s:>8s}")
    print(f"\n{len(keys)} days, {n_ver} verified, {total:,} trades")
    print("\ngaps > 3 min inside a CME session (diagnostic):")
    n_gaps = 0
    for f in files:
        for a, b in session_gaps(pd.read_parquet(f, columns=["ts"])["ts"]):
            n_gaps += 1
            print(f"  {f.stem}: {a:%H:%M:%S} -> {b:%H:%M:%S} UTC")
    print("  none" if n_gaps == 0 else f"  {n_gaps} gaps")
    log_file = folder / "download.log"
    if log_file.exists():
        print(f"\nlast lines of {log_file}:")
        print("\n".join(log_file.read_text(encoding="utf-8").splitlines()[-5:]))
    return n_ver, len(keys)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data_raw/ticks_ibkr/NQZ6")
