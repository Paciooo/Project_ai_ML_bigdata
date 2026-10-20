"""
Download tick-by-tick futures trades (e.g. NQ) from Interactive Brokers.

Source: CME Globex trades, delivered by IBKR's historical Time & Sales
        (reqHistoricalTicks, whatToShow="TRADES").

IB limits (official docs):
  * tick history only for the last ~3 years, and expired futures only up to
    2 years after expiry;
  * at most 1000 ticks per request and 60 requests per 10 minutes.

To leave room for other programs using the same IB connection, this script:
  * makes at most --max-requests per 10 minutes (default 30, half of IB's
    limit), i.e. about 180,000 ticks per hour;
  * pauses while the US stock and options markets are open (Monday-Friday
    09:25-16:20 New York time) and works at night and on weekends;
    use --run-during-us-hours to disable the pause;
  * reconnects by itself if IB Gateway disconnects (e.g. daily restart).
A full NQ trading day takes a few hours, so it is meant for a recent window
of a few weeks, not for years.

It saves one Parquet file per day (UTC) and skips days already on disk, so it
can be stopped with Ctrl+C and restarted at any time.

Usage (IB Gateway or TWS running and logged in, API enabled):
  pip install ib_async pandas pyarrow
  python tools/download_ticks_ibkr.py --symbol NQ --contract-month 202612 --start 2026-09-21 --end 2026-10-20 --port 4001

  --port: 4001 Gateway live, 4002 Gateway paper, 7496 TWS live, 7497 TWS paper.
  A CME market data subscription is required (otherwise IB returns error 354/162).

Choose the contract month that was the front month for the whole window
(NQ/ES roll about a week before the March/June/September/December expiry).
Output columns: ts (UTC), price, size.
"""

import argparse
import time
from datetime import datetime, timedelta, timezone
from datetime import time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

# Set from the command line in main()
PACING_SECONDS = 20.0            # 600 s / max requests per 10 minutes
PAUSE_DURING_US_HOURS = True
CONNECT = {}                     # host, port, clientId for reconnecting

NEW_YORK = ZoneInfo("America/New_York")
US_SESSION = (dtime(9, 25), dtime(16, 20))  # stocks 9:30-16:00, index options to 16:15


def us_markets_open(now=None):
    """True during the US stock/options session (Mon-Fri, New York time)."""
    now = (now or datetime.now(timezone.utc)).astimezone(NEW_YORK)
    return now.weekday() < 5 and US_SESSION[0] <= now.time() < US_SESSION[1]


def wait(ib, seconds):
    """Sleep while keeping the IB connection alive."""
    (ib.sleep if hasattr(ib, "sleep") else time.sleep)(max(0.0, seconds))


def request(fn, *args, **kwargs):
    """Call an IB historical-data function politely.

    Waits while the US markets are open, reconnects if needed, and spaces
    requests by PACING_SECONDS. Retries the same request after a disconnect,
    so a lost connection never looks like "no data".
    """
    ib = fn.__self__
    while True:
        if PAUSE_DURING_US_HOURS and us_markets_open():
            print(f"\n  US markets open: paused until 16:20 New York time "
                  f"(now {datetime.now(NEW_YORK):%a %H:%M} NY)")
            while us_markets_open():
                wait(ib, 60)
            print("  resuming")
        while not ib.isConnected():
            try:
                ib.connect(**CONNECT)
                print("  reconnected to IB")
            except Exception as e:  # Gateway restarting or not logged in yet
                print(f"  IB not reachable ({e}); retrying in 60 s")
                time.sleep(60)
        t0 = time.time()
        try:
            out = fn(*args, **kwargs)
        except ConnectionError:
            out = None
        wait(ib, PACING_SECONDS - (time.time() - t0))
        if out is not None and ib.isConnected():
            return out
        print("  connection lost during a request; retrying it")


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
    p.add_argument("--max-requests", type=int, default=30,
                   help="max historical requests per 10 minutes (IB limit: 60)")
    p.add_argument("--run-during-us-hours", action="store_true",
                   help="do not pause while US stocks/options are trading")
    args = p.parse_args()

    global PACING_SECONDS, PAUSE_DURING_US_HOURS
    PACING_SECONDS = 600 / min(args.max_requests, 60) + 0.5
    PAUSE_DURING_US_HOURS = not args.run_during_us_hours
    print(f"Max {600 / PACING_SECONDS:.0f} requests per 10 min; "
          f"{'pausing' if PAUSE_DURING_US_HOURS else 'NOT pausing'} during US market hours")

    from ib_async import IB, Future  # imported here so --help works without the package

    CONNECT.update(host=args.host, port=args.port, clientId=args.client_id, readonly=True)
    ib = IB()
    ib.connect(**CONNECT)
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
