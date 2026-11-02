"""
Download historical futures BARS (not ticks) from Interactive Brokers.

Uses reqHistoricalData directly (whatToShow=TRADES, all hours), one request
per UTC day, so long bar histories do not require downloading every trade.
Read-only: no orders. Same polite request layer as the tick script
(tools/ib_common.py): pacing, pause during US market hours, typed errors,
retries, atomic files and a manifest.

What IBKR can return (check the manifest; nothing is promised):
  * an individual futures contract only while it is listed or expired for
    less than about 2 years (includeExpired=True);
  * the continuous future (secType CONTFUT) only with an EMPTY endDateTime,
    i.e. one request ending now (use --continuous with --duration); its
    roll adjustment is IB's, not this project's;
  * a reconstruction of 2006-2025 NQ history is NOT possible this way; the
    actual coverage returned is recorded per day.
Daily bars ("1 day") of futures may carry the SETTLEMENT price as the close,
published after the session: they are flagged in the manifest and must not
be used as if known at the last trade. Build session candles from intraday
bars (last-trade close) instead.

Output: <out>/<localSymbol>/<bar size>/<YYYY-MM-DD>.parquet with ts_start
(UTC), open, high, low, close, volume, wap, bar_count, and manifest.json.

Usage:
  python tools/download_bars_ibkr.py --symbol NQ --contract-month 202512 --start 2025-09-01 --end 2025-12-19 --bar-size "1 min"
  python tools/download_bars_ibkr.py --symbol NQ --continuous --duration "1 Y" --bar-size "1 hour"
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ib_common import Manifest, Requester, RequestError, atomic_parquet  # noqa: E402

log = logging.getLogger("ib")
COLUMNS = ["ts_start", "open", "high", "low", "close", "volume", "wap", "bar_count"]


def bars_frame(bars):
    return pd.DataFrame([(b.date, b.open, b.high, b.low, b.close, b.volume, getattr(b, "average", None),
                          getattr(b, "barCount", None)) for b in bars], columns=COLUMNS)


def download_days(req, man, out, contract, start, end, bar_size):
    day, summary = start, {}
    out.mkdir(parents=True, exist_ok=True)
    while day < end:
        key = f"{bar_size.replace(' ', '')}/{day:%Y-%m-%d}"
        if day.weekday() == 5 or man.day(key).get("status") in ("complete", "no_data"):
            day += timedelta(days=1)
            continue
        nxt = day + timedelta(days=1)
        entry = {"utc_start": day.isoformat(), "utc_end": nxt.isoformat(), "bar_size": bar_size,
                 "close_may_be_settlement": bar_size.endswith("day")}
        try:
            res = req.call("reqHistoricalData", contract=contract, endDateTime=nxt, durationStr="86400 S",
                           barSizeSetting=bar_size, whatToShow="TRADES", useRTH=False, formatDate=2, timeout=0)
        except RequestError as e:
            man.set_day(key, status="incomplete", error_kind=e.kind, unresolved_errors=str(e.errors), **entry)
            summary[key] = "incomplete"
            if e.kind in ("entitlement", "permission", "fatal"):
                raise
            day = nxt
            continue
        df = bars_frame([b for b in res.data if day <= b.date < nxt])
        if df.empty:
            man.set_day(key, status="no_data", reason=res.status, n_bars=0, **entry)
            summary[key] = "no_data"
        else:
            atomic_parquet(df, out / f"{day:%Y-%m-%d}.parquet")
            man.set_day(key, status="complete", n_bars=len(df), first=str(df.ts_start.min()),
                        last=str(df.ts_start.max()), **entry)
            summary[key] = "complete"
        day = nxt
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbol", default="NQ")
    p.add_argument("--exchange", default="CME")
    p.add_argument("--contract-month", help="e.g. 202512 (individual contract)")
    p.add_argument("--continuous", action="store_true", help="CONTFUT, single request ending now")
    p.add_argument("--duration", default="1 M", help="--continuous only, e.g. '1 Y'")
    p.add_argument("--start", help="first UTC day (contract mode)")
    p.add_argument("--end", help="last UTC day, exclusive (contract mode)")
    p.add_argument("--bar-size", default="1 min", choices=["1 min", "5 mins", "15 mins", "1 hour", "4 hours", "1 day"])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=4002)
    p.add_argument("--client-id", type=int, default=78)
    p.add_argument("--out", default="data_raw/bars_ibkr")
    p.add_argument("--max-requests", type=int, default=30)
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--run-during-us-hours", action="store_true")
    a = p.parse_args()
    if a.continuous == bool(a.contract_month):
        p.error("give exactly one of --contract-month or --continuous")
    if a.contract_month and not (a.start and a.end):
        p.error("--start and --end are required with --contract-month")

    from ib_async import IB, ContFuture, Future
    connect = dict(host=a.host, port=a.port, clientId=a.client_id, readonly=True)
    ib = IB()
    req = Requester(ib, connect, pacing_s=600 / min(max(a.max_requests, 1), 60) + 0.5, timeout_s=a.timeout,
                    pause_us_hours=not a.run_during_us_hours)
    ib.connect(**connect)
    c = (ContFuture(a.symbol, exchange=a.exchange, currency="USD") if a.continuous else
         Future(a.symbol, lastTradeDateOrContractMonth=a.contract_month, exchange=a.exchange,
                currency="USD", includeExpired=True))
    ib.qualifyContracts(c)
    name = c.localSymbol or f"{a.symbol}_CONT"
    out = Path(a.out) / name
    out.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(out / "download.log", encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", handlers=[logging.StreamHandler(), fh])
    man = Manifest(out, {"symbol": a.symbol, "localSymbol": name, "conId": c.conId, "secType": c.secType,
                         "contract_month": a.contract_month, "lastTradeDate": c.lastTradeDateOrContractMonth,
                         "multiplier": c.multiplier, "whatToShow": "TRADES", "useRTH": False})
    try:
        if a.continuous:   # IB requires an empty endDateTime for CONTFUT
            res = req.call("reqHistoricalData", contract=c, endDateTime="", durationStr=a.duration,
                           barSizeSetting=a.bar_size, whatToShow="TRADES", useRTH=False, formatDate=2, timeout=0)
            df = bars_frame(res.data)
            f = out / f"{a.bar_size.replace(' ', '')}_continuous_{datetime.now(timezone.utc):%Y%m%d}.parquet"
            atomic_parquet(df, f)
            man.set_day(f.stem, status="complete" if len(df) else "no_data", n_bars=len(df),
                        first=str(df.ts_start.min()) if len(df) else None, last=str(df.ts_start.max()) if len(df) else None,
                        duration=a.duration, bar_size=a.bar_size, close_may_be_settlement=a.bar_size.endswith("day"))
            log.info(f"saved {len(df):,} bars to {f}")
        else:
            s = datetime.fromisoformat(a.start).replace(tzinfo=timezone.utc)
            e = datetime.fromisoformat(a.end).replace(tzinfo=timezone.utc)
            summary = download_days(req, man, out / a.bar_size.replace(" ", ""), c, s, e, a.bar_size)
            log.info(f"finished: {pd.Series(summary).value_counts().to_dict() if summary else 'nothing to do'}")
    finally:
        ib.disconnect()


if __name__ == "__main__":
    main()
