"""
Download tick-by-tick CME futures trades (e.g. NQ, ES) from Databento.

Source: CME Globex MDP 3.0 feed, Databento dataset "GLBX.MDP3"
        (history from June 2010). Every trade printed on CME Globex,
        timestamped by the exchange.

Databento charges per request, so this script has two modes:

  1) Estimate only (default, free): prints the cost and size per year.
       python tools/download_ticks_databento.py --symbol NQ --start 2020-01-01 --end 2025-01-01

  2) Download (charged): one Parquet file per month, already-downloaded
     months are skipped, so the script can be stopped and re-run.
       python tools/download_ticks_databento.py --symbol NQ --start 2020-01-01 --end 2025-01-01 --download

Setup:
  pip install databento pandas pyarrow
  set DATABENTO_API_KEY=db-xxxxxxxx        (Windows PowerShell: $env:DATABENTO_API_KEY="db-...")

The contract is the continuous front month rolled by volume ("NQ.v.0"):
on each day the most traded contract is used. Output columns:
  ts (exchange timestamp, UTC), price, size, side (A=sell aggressor,
  B=buy aggressor, N=none), symbol (the actual contract, e.g. NQH4).
"""

import argparse
import os
from pathlib import Path

import pandas as pd

DATASET = "GLBX.MDP3"


def month_ranges(start, end):
    """[(month_start, next_month_start), ...] covering start -> end."""
    edges = list(pd.date_range(start, end, freq="MS"))
    edges = [pd.Timestamp(start)] + [e for e in edges if e > pd.Timestamp(start)]
    if edges[-1] < pd.Timestamp(end):
        edges.append(pd.Timestamp(end))
    return list(zip(edges[:-1], edges[1:]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbol", default="NQ", help="CME root symbol, e.g. NQ or ES")
    p.add_argument("--start", required=True, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--end", required=True, help="YYYY-MM-DD (exclusive)")
    p.add_argument("--out", default="data_raw/ticks", help="output folder")
    p.add_argument("--download", action="store_true", help="actually download (charged)")
    args = p.parse_args()

    import databento as db  # imported here so --help works without the package

    if not os.environ.get("DATABENTO_API_KEY"):
        raise SystemExit("Set the DATABENTO_API_KEY environment variable first.")
    client = db.Historical()
    query = dict(dataset=DATASET, symbols=[f"{args.symbol}.v.0"],
                 stype_in="continuous", schema="trades")

    print(f"Dataset {DATASET} available: {client.metadata.get_dataset_range(dataset=DATASET)}")

    # ---- 1) Cost estimate per year (free) ----------------------------------
    total_cost, total_gb = 0.0, 0.0
    years = pd.date_range(args.start, args.end, freq="YS").tolist()
    edges = [pd.Timestamp(args.start)] + [y for y in years if y > pd.Timestamp(args.start)] + [pd.Timestamp(args.end)]
    for a, b in zip(edges[:-1], edges[1:]):
        if a >= b:
            continue
        cost = client.metadata.get_cost(start=a.date().isoformat(), end=b.date().isoformat(), **query)
        size = client.metadata.get_billable_size(start=a.date().isoformat(), end=b.date().isoformat(), **query)
        total_cost += cost
        total_gb += size / 1e9
        print(f"  {a.date()} -> {b.date()}: ${cost:,.2f}  ({size / 1e9:.2f} GB)")
    print(f"TOTAL {args.symbol} trades {args.start} -> {args.end}: ${total_cost:,.2f} ({total_gb:.2f} GB)")

    if not args.download:
        print("\nEstimate only, nothing was charged. Add --download to fetch the data.")
        return

    # ---- 2) Download month by month (charged) ------------------------------
    out = Path(args.out) / args.symbol
    out.mkdir(parents=True, exist_ok=True)
    for a, b in month_ranges(args.start, args.end):
        f = out / f"{a:%Y-%m}.parquet"
        if f.exists():
            print(f"  skip {f.name} (already on disk)")
            continue
        store = client.timeseries.get_range(start=a.isoformat(), end=b.isoformat(), **query)
        df = store.to_df()  # index = ts_recv; ts_event = exchange timestamp
        df = (df.reset_index()[["ts_event", "price", "size", "side", "symbol"]]
                .rename(columns={"ts_event": "ts"})
                .sort_values("ts", kind="stable"))
        df.to_parquet(f, index=False, compression="zstd")
        print(f"  saved {f}  ({len(df):,} trades)")


if __name__ == "__main__":
    main()
