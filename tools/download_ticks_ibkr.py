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

Robustness:
  * every request has a timeout (--timeout, default 120 s): if IB never
    answers, the request is retried, and after 3 timeouts in a row the
    connection is reset;
  * everything is written to <out>/<contract>/download.log (with IB's own
    error messages), so you can see what happened even if the window closed;
  * the day in progress is saved every ~10 minutes to <day>.partial.parquet
    and resumed from there after a restart; the final <day>.parquet replaces
    it when the day is complete;
  * a day for which IB returns no 1-minute bars (an error, not a holiday) is
    not saved, so the next run retries it.
Days already on disk are skipped: stop with Ctrl+C and restart at any time.

Usage (IB Gateway or TWS running and logged in, API enabled):
  pip install ib_async pandas pyarrow
  python tools/download_ticks_ibkr.py --symbol NQ --contract-month 202612 --start 2026-09-21 --end 2026-10-20 --port 4002

  --port: 4001 Gateway live, 4002 Gateway paper, 7496 TWS live, 7497 TWS paper.
  A CME market data subscription is required (otherwise IB returns error 354/162).

Choose the contract month that was the front month for the whole window
(NQ/ES roll about a week before the March/June/September/December expiry).
Output columns: ts (UTC), price, size.
"""

import argparse
import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from datetime import time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

# Set from the command line in main()
PACING_SECONDS = 20.0            # 600 s / max requests per 10 minutes
PAUSE_DURING_US_HOURS = True
REQUEST_TIMEOUT = 120.0          # seconds to wait for an answer from IB
CHECKPOINT_EVERY = 30            # requests between partial saves (~10 min at 30/10 min)
CONNECT = {}                     # host, port, clientId for reconnecting

NEW_YORK = ZoneInfo("America/New_York")
US_SESSION = (dtime(9, 25), dtime(16, 20))  # stocks 9:30-16:00, index options to 16:15
INFO_CODES = {2104, 2106, 2107, 2108, 2119, 2158}  # IB "data farm OK" messages

log = logging.getLogger("ticks")


def setup_logging(log_file):
    """Log to the console and to a file (appending across runs)."""
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)


def on_ib_error(reqId, errorCode, errorString, contract=None):
    """Write IB's error messages to the log."""
    if errorCode in INFO_CODES:
        return
    log.warning(f"IB error {errorCode} (req {reqId}): {errorString}")


def us_markets_open(now=None):
    """True during the US stock/options session (Mon-Fri, New York time)."""
    now = (now or datetime.now(timezone.utc)).astimezone(NEW_YORK)
    return now.weekday() < 5 and US_SESSION[0] <= now.time() < US_SESSION[1]


def wait(ib, seconds):
    """Sleep while keeping the IB connection alive."""
    (ib.sleep if hasattr(ib, "sleep") else time.sleep)(max(0.0, seconds))


def request(ib, name, **kwargs):
    """Call IB's async historical-data method `name` politely and safely.

    Waits while the US markets are open, reconnects if needed, spaces
    requests by PACING_SECONDS and gives up on an answer after
    REQUEST_TIMEOUT seconds. A timeout or a lost connection never looks like
    "no data": the same request is retried.
    """
    timeouts = 0
    while True:
        if PAUSE_DURING_US_HOURS and us_markets_open():
            log.info(f"US markets open: paused until 16:20 New York time "
                     f"(now {datetime.now(NEW_YORK):%a %H:%M} NY)")
            while us_markets_open():
                wait(ib, 60)
            log.info("resuming")
        while not ib.isConnected():
            try:
                ib.connect(**CONNECT)
                log.info("reconnected to IB")
            except Exception as e:  # Gateway restarting or not logged in yet
                log.warning(f"IB not reachable ({e}); retrying in 60 s")
                time.sleep(60)

        t0 = time.time()
        out = None
        try:
            coro = getattr(ib, name + "Async")(**kwargs)
            out = ib.run(asyncio.wait_for(coro, REQUEST_TIMEOUT))
        except asyncio.TimeoutError:
            timeouts += 1
            log.warning(f"{name}: no answer from IB after {REQUEST_TIMEOUT:.0f} s "
                        f"(timeout {timeouts}); retrying")
            if timeouts >= 3:
                log.warning("3 timeouts in a row: resetting the connection")
                ib.disconnect()
                timeouts = 0
        except ConnectionError as e:
            log.warning(f"{name}: connection error ({e}); retrying")
        wait(ib, PACING_SECONDS - (time.time() - t0))
        if out is not None and ib.isConnected():
            return out
        if out is not None:
            log.warning(f"{name}: connection lost during the request; retrying")


def active_minutes(ib, contract, day_start, day_end):
    """UTC minutes of the day that contain at least one trade (1-min bars)."""
    bars = request(ib, "reqHistoricalData", contract=contract, endDateTime=day_end,
                   durationStr="86400 S", barSizeSetting="1 min", whatToShow="TRADES",
                   useRTH=False, formatDate=2, timeout=0)  # timeout=0: ours applies
    return [b.date for b in bars if b.volume > 0 and day_start <= b.date < day_end]


def save_rows(rows, path):
    pd.DataFrame(rows, columns=["ts", "price", "size"]).to_parquet(path, index=False, compression="zstd")


def download_day(ib, contract, day_start, day_end, partial_file):
    """All trades between day_start and day_end (UTC), or None if IB gave no bars.

    Progress is saved to `partial_file` every CHECKPOINT_EVERY requests and
    resumed from it if it exists.
    """
    minutes = active_minutes(ib, contract, day_start, day_end)
    if not minutes:
        return None

    rows, cursor, i_min = [], minutes[0], 0
    seen_at_last_ts = 0  # ticks already kept with the same timestamp as `cursor`
    if partial_file.exists():
        old = pd.read_parquet(partial_file)
        if len(old):
            rows = list(old.itertuples(index=False, name=None))
            cursor = rows[-1][0].to_pydatetime() if hasattr(rows[-1][0], "to_pydatetime") else rows[-1][0]
            seen_at_last_ts = int((old["ts"] == old["ts"].iloc[-1]).sum())
            log.info(f"resuming from {cursor:%H:%M:%S} UTC ({len(rows):,} ticks already saved)")

    skipped_seconds, n_requests = 0, 0  # skipped: seconds with >1000 trades
    while True:
        ticks = request(ib, "reqHistoricalTicks", contract=contract, startDateTime=cursor,
                        endDateTime="", numberOfTicks=1000, whatToShow="TRADES", useRth=False)
        ticks = [t for t in ticks if t.time < day_end]
        n_requests += 1

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
            print(f"    {cursor:%H:%M:%S} UTC  {len(rows):,} ticks", end="\r")
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

        if n_requests % CHECKPOINT_EVERY == 0 and rows:
            save_rows(rows, partial_file)
            log.info(f"checkpoint: up to {cursor:%H:%M:%S} UTC, {len(rows):,} ticks")

    print()
    if skipped_seconds:
        log.info(f"note: {skipped_seconds} second(s) may have had more than 1000 trades (possibly incomplete)")
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
    p.add_argument("--timeout", type=float, default=120, help="seconds to wait for each IB answer")
    p.add_argument("--run-during-us-hours", action="store_true",
                   help="do not pause while US stocks/options are trading")
    args = p.parse_args()

    global PACING_SECONDS, PAUSE_DURING_US_HOURS, REQUEST_TIMEOUT
    PACING_SECONDS = 600 / min(args.max_requests, 60) + 0.5
    PAUSE_DURING_US_HOURS = not args.run_during_us_hours
    REQUEST_TIMEOUT = args.timeout

    from ib_async import IB, Future  # imported here so --help works without the package

    CONNECT.update(host=args.host, port=args.port, clientId=args.client_id, readonly=True)
    ib = IB()
    ib.errorEvent += on_ib_error
    ib.connect(**CONNECT)
    contract = Future(args.symbol, lastTradeDateOrContractMonth=args.contract_month,
                      exchange=args.exchange, currency="USD", includeExpired=True)
    ib.qualifyContracts(contract)

    out = Path(args.out) / contract.localSymbol
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out / "download.log")
    log.info(f"start: {contract.localSymbol} (conId {contract.conId}), {args.start} -> {args.end}, "
             f"max {600 / PACING_SECONDS:.0f} requests/10 min, timeout {REQUEST_TIMEOUT:.0f} s, "
             f"{'pausing' if PAUSE_DURING_US_HOURS else 'NOT pausing'} during US market hours")

    day = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    try:
        while day < end:
            f = out / f"{day:%Y-%m-%d}.parquet"
            partial = out / f"{day:%Y-%m-%d}.partial.parquet"
            if day.weekday() == 5 or f.exists():  # CME is closed on Saturday (UTC)
                day += timedelta(days=1)
                continue
            log.info(f"{day:%Y-%m-%d}: downloading")
            df = download_day(ib, contract, day, day + timedelta(days=1), partial)
            if df is None:
                log.warning(f"{day:%Y-%m-%d}: IB returned no 1-minute bars (error or no data); "
                            f"day NOT saved, it will be retried on the next run")
            else:
                df.to_parquet(f, index=False, compression="zstd")
                partial.unlink(missing_ok=True)
                log.info(f"{day:%Y-%m-%d}: saved {f} ({len(df):,} trades)")
            day += timedelta(days=1)
        log.info("finished")
    except KeyboardInterrupt:
        log.info("stopped by the user (Ctrl+C); restart the same command to continue")
    except Exception:
        log.exception("stopped by an error")
        raise
    finally:
        ib.disconnect()


if __name__ == "__main__":
    main()
