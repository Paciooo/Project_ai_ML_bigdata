"""IBKR tick download against a mocked IB (no network, no brokerage)."""

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import download_ticks_ibkr as dl
from fake_ib import FakeIB
from ib_common import Manifest, Requester, RequestError, classify

DAY = datetime(2026, 10, 13, tzinfo=timezone.utc)   # a Tuesday


def trades_fixture():
    """~3,000 trades incl. a burst of 2,500 trades in one second and identical repeats."""
    t, out = DAY + timedelta(hours=14), []
    for i in range(300):
        out.append((t + timedelta(seconds=2 * i), 20000 + i * 0.25, 1.0))
    burst = DAY + timedelta(hours=14, minutes=3, seconds=1)
    out += [(burst, 20100.0, 2.0)] * 2500             # identical records: legitimate trades
    out += [(DAY + timedelta(hours=15, minutes=30, seconds=s), 20050.0, 1.0) for s in range(0, 60, 7)]
    return sorted(out, key=lambda x: x[0])


def requester(ib, retries=3):
    return Requester(ib, {}, pacing_s=0, timeout_s=0.05, max_retries=retries, backoff_s=0,
                     pause_us_hours=False)


def run(tmp_path, ib, **kw):
    req = requester(ib, kw.pop("retries", 3))
    man = Manifest(tmp_path, {"conId": 1, "localSymbol": "NQZ6"})
    res = dl.run_range(req, man, tmp_path, object(), DAY, DAY + timedelta(days=1), **kw)
    return res, man


def test_complete_day_no_loss_no_duplicates(tmp_path):
    trades = trades_fixture()
    res, man = run(tmp_path, FakeIB(trades))
    assert res["2026-10-13"] == "complete"
    df = pd.read_parquet(tmp_path / "2026-10-13.parquet")
    assert len(df) == len(trades)                        # all 2,500 burst trades kept
    assert df["size"].sum() == sum(t[2] for t in trades)
    entry = man.day("2026-10-13")
    assert entry["volume_ratio"] == pytest.approx(1.0) and entry["n_missing_minutes"] == 0
    assert entry["seconds_with_1000_plus"] >= 1          # diagnostic, not a failure
    assert json.loads((tmp_path / "manifest.json").read_text())["schema_version"] == 2


def test_repeated_10187_is_incomplete_not_empty(tmp_path):
    ib = FakeIB(trades_fixture(), script={i: ("error", 10187, "Failed to request historical ticks")
                                          for i in range(1, 50)})
    res, man = run(tmp_path, ib)
    assert res["2026-10-13"] == "incomplete"
    assert not (tmp_path / "2026-10-13.parquet").exists()
    assert man.day("2026-10-13")["error_kind"] == "session"


def test_single_10187_then_success(tmp_path):
    ib = FakeIB(trades_fixture(), script={2: ("error", 10187, "another session")})
    res, _ = run(tmp_path, ib)
    assert res["2026-10-13"] == "complete"


def test_timeout_and_disconnect_are_retried(tmp_path):
    ib = FakeIB(trades_fixture(), script={1: ("timeout",), 3: ("disconnect",)})
    res, _ = run(tmp_path, ib)
    assert res["2026-10-13"] == "complete"
    assert ib.connects >= 1
    df = pd.read_parquet(tmp_path / "2026-10-13.parquet")
    assert len(df) == len(trades_fixture())


def test_entitlement_error_stops(tmp_path):
    ib = FakeIB(trades_fixture(), bars_script={1: ("error", 354, "not subscribed")})
    with pytest.raises(RequestError) as e:
        run(tmp_path, ib)
    assert e.value.kind == "entitlement"


def test_holiday_no_data_is_recorded(tmp_path):
    res, man = run(tmp_path, FakeIB([]))
    assert res["2026-10-13"] == "no_trades"
    assert man.day("2026-10-13")["status"] == "no_trades"


def test_failed_coverage_not_promoted(tmp_path, monkeypatch):
    trades = trades_fixture()
    ib = FakeIB(trades)
    real = ib.reqHistoricalTicksAsync

    async def lossy(**kw):              # IB silently drops the 15:30 trades
        out = await real(**kw)
        return [t for t in out if t.time.hour != 15]
    ib.reqHistoricalTicksAsync = lossy
    res, man = run(tmp_path, ib)
    assert res["2026-10-13"] == "failed_coverage"
    assert not (tmp_path / "2026-10-13.parquet").exists()
    assert (tmp_path / "2026-10-13.incomplete.parquet").exists()
    assert man.day("2026-10-13")["n_missing_minutes"] > 0


def test_resume_after_failure_without_loss(tmp_path):
    trades = trades_fixture()
    # first run: checkpoint every page, then every request fails from call 3 on
    ib = FakeIB(trades, script={i: ("error", 10187, "x") for i in range(3, 40)})
    req = requester(ib, 2)
    man = Manifest(tmp_path, {"conId": 1})
    r = dl.run_range(req, man, tmp_path, object(), DAY, DAY + timedelta(days=1), checkpoint_every=1)
    assert r["2026-10-13"] == "incomplete"
    assert (tmp_path / "2026-10-13.partial.parquet").exists()
    assert man.day("2026-10-13").get("cursor")
    # second run resumes from the checkpoint
    ib2 = FakeIB(trades)
    man2 = Manifest(tmp_path, {"conId": 1})
    man2.data["days"]["2026-10-13"]["status"] = "in_progress"   # as left by the checkpoint
    r2 = dl.run_range(requester(ib2), man2, tmp_path, object(), DAY, DAY + timedelta(days=1))
    assert r2["2026-10-13"] == "complete"
    df = pd.read_parquet(tmp_path / "2026-10-13.parquet")
    assert len(df) == len(trades)
    assert not (tmp_path / "2026-10-13.partial.parquet").exists()


def test_existing_file_without_manifest_is_unverified(tmp_path):
    pd.DataFrame({"ts": [DAY], "price": [1.0], "size": [1.0]}).to_parquet(tmp_path / "2026-10-13.parquet")
    res, _ = run(tmp_path, FakeIB(trades_fixture()))
    assert res["2026-10-13"] == "unverified"
    assert len(pd.read_parquet(tmp_path / "2026-10-13.parquet")) == 1   # untouched


def test_classify_codes():
    assert classify(162, "HMDS query returned no data") == "no_data"
    assert classify(162, "Historical Market Data Service error message:API historical data query cancelled: pacing violation") == "pacing"
    assert classify(10187, "") == "session"
    assert classify(354, "") == "entitlement"


def test_error_for_other_request_is_ignored():
    ib = FakeIB(trades_fixture())
    req = requester(ib)
    ib.errorEvent.emit(9999, 10187, "belongs to another request", None)   # not ours
    res = req.call("reqHistoricalTicks", contract=None, startDateTime=DAY, endDateTime="",
                   numberOfTicks=10, whatToShow="TRADES", useRth=False)
    assert len(res.data) >= 10 and res.status == "ok"


def test_manifest_rejects_other_contract(tmp_path):
    Manifest(tmp_path, {"conId": 1}).save()
    with pytest.raises(ValueError):
        Manifest(tmp_path, {"conId": 2})


def test_bar_downloader_manifest_and_errors(tmp_path):
    import download_bars_ibkr as bars_dl
    ib = FakeIB(trades_fixture())
    man = Manifest(tmp_path, {"conId": 1})
    s = bars_dl.download_days(requester(ib), man, tmp_path / "1min", object(), DAY,
                              DAY + timedelta(days=2), "1 min")
    assert s["1min/2026-10-13"] == "complete" and s["1min/2026-10-14"] == "no_data"
    df = pd.read_parquet(tmp_path / "1min" / "2026-10-13.parquet")
    assert (df["high"] >= df["low"]).all() and df["volume"].sum() == sum(t[2] for t in trades_fixture())
    ib2 = FakeIB(trades_fixture(), bars_script={i: ("error", 10187, "x") for i in range(1, 20)})
    man2 = Manifest(tmp_path / "b", {"conId": 1})
    (tmp_path / "b").mkdir(exist_ok=True)
    s2 = bars_dl.download_days(requester(ib2), man2, tmp_path / "b" / "1min", object(), DAY,
                               DAY + timedelta(days=1), "1 min")
    assert s2["1min/2026-10-13"] == "incomplete"


def test_check_ticks_reads_manifest(tmp_path, capsys):
    import check_ticks
    run(tmp_path, FakeIB(trades_fixture()))
    pd.DataFrame({"ts": pd.to_datetime(["2026-10-14 14:00"], utc=True), "price": [1.0],
                  "size": [1.0]}).to_parquet(tmp_path / "2026-10-14.parquet")   # no manifest entry
    n_ver, n = check_ticks.main(tmp_path)
    out = capsys.readouterr().out
    assert n_ver == 1 and n == 2 and "UNVERIFIED" in out
