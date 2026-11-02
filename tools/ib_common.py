"""
Shared request layer for the IBKR historical-data scripts (read-only API use).

Every request returns a typed `Result` or raises `RequestError`; an empty
answer that came with an unresolved error is NEVER turned into "no data".

Error handling (IB codes, from the TWS API documentation):
  no_data      162 "HMDS query returned no data" -> Result(status="no_data")
  pacing       162/420 "pacing violation"          -> retry with backoff
  session      10187 (another session / failed)    -> retry with backoff
  connection   1100, 1102 (?), 504, 2105, 2103     -> reconnect and retry
  entitlement  354, 10089, 10090, 10167, 10168      -> stop at once
  permission   321, 10197                           -> stop at once
  fatal        200 (no security definition), 203    -> stop at once
  other                                             -> retry with backoff
After `max_retries` retryable failures the request raises RequestError, so
the caller records the range as incomplete instead of empty.

Errors are matched to the request by request ID: ib_async assigns IDs from
`ib.client._reqIdSeq`, so the IDs used by a call are those between the
sequence value before and after it. If that attribute is missing, any
request-tied error during the call counts (conservative).
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from datetime import time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

NEW_YORK = ZoneInfo("America/New_York")
US_SESSION = (dtime(9, 25), dtime(16, 20))   # stocks 9:30-16:00, index options to 16:15
INFO_CODES = {2104, 2106, 2107, 2108, 2119, 2158}
NO_RETRY = {"entitlement", "permission", "fatal"}

log = logging.getLogger("ib")


def classify(code, message=""):
    msg = (message or "").lower()
    if code == 162 and "pacing" in msg or code == 420:
        return "pacing"
    if code in (162, 165) and ("no data" in msg or "returned no" in msg):
        return "no_data"
    if code == 10187:
        return "session"
    if code in (1100, 1102, 504, 502, 2103, 2105, 2110):
        return "connection"
    if code in (354, 10089, 10090, 10167, 10168):
        return "entitlement"
    if code in (321, 10197):
        return "permission"
    if code in (200, 203):
        return "fatal"
    return "other"


class RequestError(Exception):
    def __init__(self, kind, errors, what=""):
        self.kind, self.errors = kind, errors
        super().__init__(f"{what}: {kind} {errors}")


@dataclass
class Result:
    data: list
    status: str = "ok"                 # ok | no_data
    errors: list = field(default_factory=list)
    attempts: int = 1


def us_markets_open(now=None):
    now = (now or datetime.now(timezone.utc)).astimezone(NEW_YORK)
    return now.weekday() < 5 and US_SESSION[0] <= now.time() < US_SESSION[1]


class Requester:
    """Polite, typed access to IB's historical endpoints."""

    def __init__(self, ib, connect_kwargs, pacing_s=20.5, timeout_s=120.0, max_retries=5,
                 backoff_s=60.0, max_backoff_s=900.0, pause_us_hours=True, sleep=None, clock=None):
        self.ib, self.connect_kwargs = ib, connect_kwargs
        self.pacing_s, self.timeout_s, self.max_retries = pacing_s, timeout_s, max_retries
        self.backoff_s, self.max_backoff_s, self.pause_us_hours = backoff_s, max_backoff_s, pause_us_hours
        self.sleep = sleep or self._sleep
        self.clock = clock or time.time
        self.errors = []               # (reqId, code, message, kind)
        ib.errorEvent += self.on_error

    def on_error(self, reqId, errorCode, errorString, contract=None, *_):
        if errorCode in INFO_CODES:
            return
        kind = classify(errorCode, errorString)
        log.warning(f"IB error {errorCode} ({kind}, req {reqId}): {errorString}")
        self.errors.append((reqId, errorCode, errorString, kind))

    def _sleep(self, seconds):
        end = time.time() + max(0.0, seconds)
        while (left := end - time.time()) > 0:
            try:
                (self.ib.sleep if hasattr(self.ib, "sleep") else time.sleep)(left)
            except (ConnectionError, OSError) as e:
                log.warning(f"connection lost while waiting ({e})")
                time.sleep(max(0.0, end - time.time()))

    def _ensure_connected(self):
        while not self.ib.isConnected():
            try:
                self.ib.connect(**self.connect_kwargs)
                log.info("reconnected to IB")
            except Exception as e:      # Gateway restarting or not logged in yet
                log.warning(f"IB not reachable ({e}); retrying in {self.backoff_s:.0f} s")
                self.sleep(self.backoff_s)

    def _req_seq(self):
        return getattr(getattr(self.ib, "client", None), "_reqIdSeq", None)

    def call(self, name, **kwargs):
        """One historical request with retries. Returns Result or raises RequestError."""
        failures, timeouts, seen = 0, 0, []
        while True:
            if self.pause_us_hours and us_markets_open():
                log.info("US markets open: paused until 16:20 New York time")
                while us_markets_open():
                    self.sleep(60)
            self._ensure_connected()
            t0, n_err, seq0 = self.clock(), len(self.errors), self._req_seq()
            out, problem = None, None
            try:
                coro = getattr(self.ib, name + "Async")(**kwargs)
                out = self.ib.run(asyncio.wait_for(coro, self.timeout_s))
            except asyncio.TimeoutError:
                timeouts += 1
                problem = ("timeout", None, f"no answer after {self.timeout_s:.0f} s")
                if timeouts % 3 == 0:
                    log.warning("3 timeouts in a row: resetting the connection")
                    self.ib.disconnect()
            except (ConnectionError, OSError) as e:
                problem = ("connection", None, str(e))
            seq1 = self._req_seq()
            mine = [e for e in self.errors[n_err:] if e[0] is not None and e[0] > 0
                    and (seq0 is None or seq1 is None or seq0 <= e[0] < max(seq1, seq0 + 1))]
            self.sleep(self.pacing_s - (self.clock() - t0))
            if problem is None and out is not None and not self.ib.isConnected():
                problem = ("connection", None, "connection lost during the request")
            if problem is None and mine:
                kinds = {e[3] for e in mine}
                if out is not None and len(out) > 0:
                    log.warning(f"{name}: data with error(s) {mine}; data kept")
                    return Result(list(out), "ok", mine, failures + 1)
                if kinds == {"no_data"}:
                    return Result([], "no_data", mine, failures + 1)
                stop = kinds & NO_RETRY
                if stop:
                    raise RequestError(sorted(stop)[0], mine, name)
                problem = (sorted(kinds - {"no_data"})[0], mine[-1][1], mine[-1][2])
            if problem is None:
                if out is None:
                    problem = ("other", None, "no result")
                else:
                    return Result(list(out), "ok", [], failures + 1)
            failures += 1
            seen.append(problem)
            if failures > self.max_retries:
                raise RequestError(problem[0], seen, name)
            wait = 0 if problem[0] == "timeout" else min(self.max_backoff_s, self.backoff_s * 2 ** (failures - 1))
            log.warning(f"{name}: {problem[0]} ({problem[2]}); retry {failures}/{self.max_retries} in {wait:.0f} s")
            self.sleep(wait)


# ---------------------------------------------------------------------------
# Atomic files and the manifest
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 2


def atomic_write_bytes(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_parquet(df, path):
    import io
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, compression="zstd")
    atomic_write_bytes(path, buf.getvalue())


class Manifest:
    """manifest.json next to the data: contract identity and per-day status."""

    def __init__(self, folder, contract=None):
        self.path = Path(folder) / "manifest.json"
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {
            "schema_version": SCHEMA_VERSION, "contract": contract or {}, "days": {}}
        if contract:
            old = self.data.get("contract") or {}
            if old.get("conId") and contract.get("conId") and old["conId"] != contract["conId"]:
                raise ValueError(f"manifest belongs to conId {old['conId']}, not {contract['conId']}")
            self.data["contract"] = {**old, **contract}

    def day(self, key):
        return self.data["days"].get(key, {})

    def set_day(self, key, **entry):
        entry["updated_utc"] = datetime.now(timezone.utc).isoformat()
        self.data["days"][key] = entry
        self.save()

    def save(self):
        atomic_write_bytes(self.path, json.dumps(self.data, indent=1, default=str).encode())
