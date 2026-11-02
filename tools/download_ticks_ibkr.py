"""
Download tick-by-tick futures trades (e.g. NQ) from Interactive Brokers.

Source: CME Globex trades from IBKR's historical Time & Sales
        (reqHistoricalTicks, whatToShow="TRADES"). Read-only: no orders.

IB limits (official docs): tick history for about the last 3 years, expired
futures up to 2 years after expiry; 1000 ticks per request (IB completes the
last second, so an answer can hold more); 60 requests per 10 minutes.

Politeness: at most --max-requests per 10 minutes (default 30), pause while
US stocks/options trade (Mon-Fri 09:25-16:20 New York) unless
--run-during-us-hours, reconnect after Gateway restarts.

Correctness (see tools/ib_common.py for the request layer):
  * pagination by whole seconds: a page starting at second S contains every
    trade of the seconds it reaches; the next page starts at the last second
    + 1 s, so same-second trades are neither lost nor duplicated and
    identical repeated trades are kept;
  * an empty answer with an unresolved IB error is a failure, never "no
    data": after the retries the day is recorded as incomplete;
  * coverage check before a day is final: every 1-minute bar with volume
    must contain trades and the total traded volume must match the 1-minute
    bars within --volume-tolerance. Otherwise the trades are kept in
    <day>.incomplete.parquet and the day is marked failed_coverage;
  * manifest.json stores the contract (conId, month), each day's UTC range,
    status, counts, coverage and unresolved errors; schema-versioned;
  * atomic writes (temporary file + rename) for checkpoints, final files and
    the manifest; <day>.partial.parquet + the manifest cursor resume a day.
Existing <day>.parquet files without a manifest entry (older downloads) are
left untouched and reported as UNVERIFIED; --recheck-unverified downloads
them again into a fresh file and replaces them only if coverage passes.

Columns: ts (UTC), price, size, exchange, special_conditions, past_limit,
unreported. Nothing is filtered (unreported/odd trades keep their flags).

Usage (IB Gateway/TWS running, API enabled, CME data subscription):
  pip install -r requirements-ibkr.txt
  python tools/download_ticks_ibkr.py --symbol NQ --contract-month 202612 --start 2026-09-21 --end 2026-10-20 --port 4002
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ib_common import Manifest, Requester, RequestError, atomic_parquet  # noqa: E402

COLUMNS = ["ts", "price", "size", "exchange", "special_conditions", "past_limit", "unreported"]
log = logging.getLogger("ib")


def setup_logging(log_file):
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)


def tick_row(t):
    a = getattr(t, "tickAttribLast", None)
    return (t.time, float(t.price), float(t.size), getattr(t, "exchange", ""),
            getattr(t, "specialConditions", ""), bool(getattr(a, "pastLimit", False)),
            bool(getattr(a, "unreported", False)))


def floor_second(ts):
    return ts.replace(microsecond=0)


def minute_bars(req, contract, day_start, day_end):
    """1-minute TRADES bars of the day: {minute start: volume}."""
    res = req.call("reqHistoricalData", contract=contract, endDateTime=day_end, durationStr="86400 S",
                   barSizeSetting="1 min", whatToShow="TRADES", useRTH=False, formatDate=2, timeout=0)
    return {b.date: float(b.volume) for b in res.data if day_start <= b.date < day_end and b.volume > 0}


def fetch_trades(req, contract, day_start, day_end, active, rows, cursor, checkpoint, every=30):
    """Page through the day's trades from `cursor`. Returns stats dict."""
    minutes = sorted(active)
    pages, empty_pages, big_seconds = 0, 0, 0
    while True:
        res = req.call("reqHistoricalTicks", contract=contract, startDateTime=cursor, endDateTime="",
                       numberOfTicks=1000, whatToShow="TRADES", useRth=False)
        pages += 1
        kept = [t for t in res.data if cursor <= t.time < day_end]
        if kept:
            rows.extend(tick_row(t) for t in kept)
            last = floor_second(kept[-1].time)
            n_last = sum(1 for t in kept if floor_second(t.time) == last)
            big_seconds += n_last >= 1000           # diagnostic only, not proof of truncation
            cursor = last + timedelta(seconds=1)    # IB completed second `last`
        elif any(t.time < cursor for t in res.data):
            # Answer before the cursor (no progress): skip to the next minute
            # with trades; the coverage check reports anything lost.
            empty_pages += 1
            later = [m for m in minutes if m > cursor]
            if not later:
                break
            cursor = later[0]
        else:
            break   # nothing at/after the cursor before day_end: the day is done
        if pages % every == 0:
            checkpoint(rows, cursor)
    return {"pages": pages, "pages_without_progress": empty_pages, "seconds_with_1000_plus": big_seconds}


def coverage(rows, active, tolerance):
    df = pd.DataFrame(rows, columns=COLUMNS)
    if df.empty:
        tick_vol = pd.Series(dtype=float)
    else:
        ts = pd.DatetimeIndex(df["ts"])
        ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
        tick_vol = df["size"].groupby(ts.floor("1min")).sum()
    bar_keys = pd.DatetimeIndex(list(active))
    bar_keys = bar_keys.tz_localize("UTC") if len(bar_keys) and bar_keys.tz is None else bar_keys
    missing = [str(m) for m in bar_keys if m not in tick_vol.index]
    bar_total, tick_total = float(sum(active.values())), float(tick_vol.sum())
    ratio = tick_total / bar_total if bar_total else (1.0 if tick_total == 0 else float("inf"))
    ok = not missing and abs(ratio - 1) <= tolerance
    return ok, {"bar_volume": bar_total, "tick_volume": tick_total, "volume_ratio": ratio,
                "n_active_minutes": len(active), "n_missing_minutes": len(missing),
                "missing_minutes_sample": missing[:10]}


def download_day(req, man, out, contract, day, tolerance, checkpoint_every=30):
    key = f"{day:%Y-%m-%d}"
    final, partial = out / f"{key}.parquet", out / f"{key}.partial.parquet"
    day_end = day + timedelta(days=1)
    entry = {"utc_start": day.isoformat(), "utc_end": day_end.isoformat()}
    try:
        active = minute_bars(req, contract, day, day_end)
        if not active:
            man.set_day(key, status="no_trades", n_ticks=0, reason="IB returned no 1-minute bars "
                        "with volume, without an error (closed market or no trading)", **entry)
            atomic_parquet(pd.DataFrame(columns=COLUMNS), final)
            log.info(f"{key}: no trades (verified by the 1-minute bars)")
            return "no_trades"
        rows, cursor = [], min(active)
        st = man.day(key)
        if st.get("status") == "in_progress" and partial.exists():
            old = pd.read_parquet(partial)
            rows = list(old[COLUMNS].itertuples(index=False, name=None))
            cursor = datetime.fromisoformat(st["cursor"])
            log.info(f"{key}: resuming at {cursor:%H:%M:%S} UTC ({len(rows):,} trades saved)")

        def checkpoint(r, c):
            atomic_parquet(pd.DataFrame(r, columns=COLUMNS), partial)
            man.set_day(key, status="in_progress", cursor=c.isoformat(), n_ticks=len(r), **entry)
            log.info(f"{key}: checkpoint at {c:%H:%M:%S} UTC, {len(r):,} trades")

        stats = fetch_trades(req, contract, day, day_end, active, rows, cursor, checkpoint, checkpoint_every)
    except RequestError as e:
        man.set_day(key, status="incomplete", unresolved_errors=str(e.errors), error_kind=e.kind,
                    **entry, **{k: v for k, v in man.day(key).items() if k in ("cursor", "n_ticks")})
        log.warning(f"{key}: INCOMPLETE ({e.kind}); progress kept, rerun to resume")
        if e.kind in ("entitlement", "permission", "fatal"):
            raise
        return "incomplete"
    ok, cov = coverage(rows, active, tolerance)
    df = pd.DataFrame(rows, columns=COLUMNS)
    if ok:
        atomic_parquet(df, final)
        partial.unlink(missing_ok=True)
        man.set_day(key, status="complete", n_ticks=len(df), file=final.name, **entry, **cov, **stats)
        log.info(f"{key}: complete, {len(df):,} trades, volume ratio {cov['volume_ratio']:.4f}")
        return "complete"
    atomic_parquet(df, out / f"{key}.incomplete.parquet")
    partial.unlink(missing_ok=True)
    man.set_day(key, status="failed_coverage", n_ticks=len(df), **entry, **cov, **stats)
    log.warning(f"{key}: coverage check FAILED ({cov['n_missing_minutes']} minutes without trades, "
                f"volume ratio {cov['volume_ratio']:.4f}); saved as {key}.incomplete.parquet, not final")
    return "failed_coverage"


def run_range(req, man, out, contract, start, end, tolerance=0.02, recheck_unverified=False,
              checkpoint_every=30):
    day, summary = start, {}
    while day < end:
        key = f"{day:%Y-%m-%d}"
        st = man.day(key).get("status")
        final = out / f"{key}.parquet"
        if day.weekday() == 5 or st in ("complete", "no_trades"):
            day += timedelta(days=1)
            continue
        if final.exists() and st is None and not recheck_unverified:
            log.info(f"{key}: file exists without manifest entry -> UNVERIFIED, left untouched "
                     f"(use --recheck-unverified to check it)")
            summary[key] = "unverified"
            day += timedelta(days=1)
            continue
        log.info(f"{key}: downloading")
        summary[key] = download_day(req, man, out, contract, day, tolerance, checkpoint_every)
        day += timedelta(days=1)
    return summary


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
    p.add_argument("--max-requests", type=int, default=30, help="per 10 minutes (IB limit: 60)")
    p.add_argument("--timeout", type=float, default=120, help="seconds to wait for each answer")
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--volume-tolerance", type=float, default=0.02)
    p.add_argument("--recheck-unverified", action="store_true")
    p.add_argument("--run-during-us-hours", action="store_true")
    args = p.parse_args()
    if not 1 <= args.max_requests <= 60:
        p.error("--max-requests must be between 1 and 60")

    from ib_async import IB, Future  # imported here so --help works without the package

    connect = dict(host=args.host, port=args.port, clientId=args.client_id, readonly=True)
    ib = IB()
    req = Requester(ib, connect, pacing_s=600 / args.max_requests + 0.5, timeout_s=args.timeout,
                    max_retries=args.max_retries, pause_us_hours=not args.run_during_us_hours)
    ib.connect(**connect)
    contract = Future(args.symbol, lastTradeDateOrContractMonth=args.contract_month,
                      exchange=args.exchange, currency="USD", includeExpired=True)
    ib.qualifyContracts(contract)
    out = Path(args.out) / contract.localSymbol
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out / "download.log")
    man = Manifest(out, {"symbol": args.symbol, "localSymbol": contract.localSymbol,
                         "conId": contract.conId, "contract_month": args.contract_month,
                         "lastTradeDate": contract.lastTradeDateOrContractMonth,
                         "multiplier": contract.multiplier, "exchange": args.exchange,
                         "whatToShow": "TRADES"})
    log.info(f"start: {contract.localSymbol} (conId {contract.conId}) {args.start} -> {args.end}")
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    try:
        summary = run_range(req, man, out, contract, start, end, args.volume_tolerance, args.recheck_unverified)
        log.info(f"finished: {pd.Series(summary).value_counts().to_dict() if summary else 'nothing to do'}")
    except KeyboardInterrupt:
        log.info("stopped by the user (Ctrl+C); rerun the same command to resume")
    except Exception:
        log.exception("stopped by an error")
        raise
    finally:
        ib.disconnect()


if __name__ == "__main__":
    main()
