"""
Download tick-by-tick futures trades (e.g. NQ) from Interactive Brokers.

Source: CME Globex trades, delivered by IBKR's historical Time & Sales
        (reqHistoricalTicks, whatToShow="TRADES").

IB limits (official docs):
  * tick history only for the last ~3 years, and expired futures only up to
    2 years after expiry;
  * at most 1000 ticks per request and 60 requests per 10 minutes.
  This script waits ~10.5 s between requests to stay within the limit, so it
  downloads ~340,000 ticks per hour. A full NQ trading day can take 1-2 hours:
  it is meant for a recent window of a few weeks, not for years.

It saves one Parquet file per day (UTC) and skips days already on disk, so it
can be stopped with Ctrl+C and restarted at any time.

Usage (IB Gateway or TWS running and logged in, API enabled):
  pip install ib_async pandas pyarrow
  python tools/download_ticks_ibkr.py --symbol NQ --contract-month 202612 --start 2026-09-21 --end 2026-10-20

  --port: 4001 Gateway live, 4002 Gateway paper, 7496 TWS live, 7497 TWS paper.
  A CME market data subscription is required (otherwise IB returns error 354/162).

Choose the contract month that was the front month for the whole window
(NQ/ES roll about a week before the March/June/September/December expiry).
Output columns: ts (UTC), price, size.
"""

import argparse
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

PACING_SECONDS = 10.5  # 60 requests per 10 minutes, with a small margin


def request(fn, *args, **kwargs):
    """Call an IB historical-data function respecting the pacing limit."""
    t0 = time.time()
    out = fn(*args, **kwargs)
    time.sleep(max(0.0, PACING_SECONDS - (time.time() - t0)))
    return out


def active_minutes(ib, contract, day_start, day_end):
    """UTC minutes of the day that contain at least one trade (1-min bars)."""
    bars = request(ib.reqHistoricalData, contract, endDateTime=day_end, durationStr="86400 S",
                   barSizeSetting="1 min", whatToShow="TRADES", useRTH=False, formatDate=2)
    return [b.date for b in bars if b.volume > 0 and day_start <= b.date < day_end]


def download_day(ib, contract, day_start, day_end):
    """All trades between day_start and day_end (UTC)."""
    minutes = active_minutes(ib, contract, day_start, day_end)
    if not minutes:
        return pd.DataFrame(columns=["ts", "price", "size"])

    rows, cursor, i_min = [], minutes[0], 0
    seen_at_last_ts = 0  # ticks already kept with the same timestamp as `cursor`
    skipped_seconds = 0  # seconds with >1000 trades that could not be completed
    while True:
        ticks = request(ib.reqHistoricalTicks, contract, startDateTime=cursor, endDateTime="",
                        numberOfTicks=1000, whatToShow="TRADES", useRth=False)
        ticks = [t for t in ticks if t.time < day_end]

        # The batch starts at `cursor`, which we already partly have: skip the
        # ticks at that exact timestamp that were kept in the previous batch.
        new, skip = [], seen_at_last_ts
        for t in ticks:
            if t.time == cursor and skip > 0:
                skip -= 1
                continue
            new.append(t)

        if new:
            rows.extend((t.time, t.price, t.size) for t in new)
            last = new[-1].time
            if last == cursor:  # still inside the same second
                seen_at_last_ts += len(new)
            else:
                seen_at_last_ts = sum(1 for t in new if t.time == last)
            cursor = last
            print(f"    {cursor:%H:%M:%S}  {len(rows):,} ticks", end="\r")
        elif ticks:
            # Only ticks we already have: this second holds more than 1000
            # trades, more than one request can return. Move on to the next
            # second (the rest of this one is lost; rare, and reported below).
            cursor, seen_at_last_ts = cursor.replace(microsecond=0) + timedelta(seconds=1), 0
            skipped_seconds += 1
        else:
            # No ticks from here to the end of the session: jump to the next
            # minute that has trades.
            nxt = cursor.replace(second=0, microsecond=0) + timedelta(minutes=1)
            while i_min < len(minutes) and minutes[i_min] < nxt:
                i_min += 1
            if i_min >= len(minutes):
                break
            cursor, seen_at_last_ts = minutes[i_min], 0

    print()
    if skipped_seconds:
        print(f"    note: {skipped_seconds} second(s) may have had more than 1000 trades (possibly incomplete)")
    return pd.DataFrame(rows, columns=["ts", "price", "size"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbol", default="NQ")
    p.add_argument("--exchange", default="CME")
    p.add_argument("--contract-month", required=True, help="e.g. 202612 for December 2026")
    p.add_argument("--start", required=True, help="first UTC day, YYYY-MM-DD")
    p.add_argument("--end", required=True, help="last UTC day (exclusive), YYYY-MM-DD")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=4002)
    p.add_argument("--client-id", type=int, default=77)
    p.add_argument("--out", default="data_raw/ticks_ibkr")
    args = p.parse_args()

    from ib_async import IB, Future  # imported here so --help works without the package

    ib = IB()
    ib.connect(args.host, args.port, clientId=args.client_id, readonly=True)
    contract = Future(args.symbol, lastTradeDateOrContractMonth=args.contract_month,
                      exchange=args.exchange, currency="USD", includeExpired=True)
    ib.qualifyContracts(contract)
    print(f"Contract: {contract.localSymbol}  (conId {contract.conId})")

    out = Path(args.out) / contract.localSymbol
    out.mkdir(parents=True, exist_ok=True)
    day = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    try:
        while day < end:
            f = out / f"{day:%Y-%m-%d}.parquet"
            if day.weekday() == 5 or f.exists():  # CME is closed on Saturday (UTC)
                day += timedelta(days=1)
                continue
            print(f"  {day:%Y-%m-%d} ...")
            df = download_day(ib, contract, day, day + timedelta(days=1))
            df.to_parquet(f, index=False, compression="zstd")
            print(f"  saved {f}  ({len(df):,} trades)")
            day += timedelta(days=1)
    finally:
        ib.disconnect()


if __name__ == "__main__":
    main()
