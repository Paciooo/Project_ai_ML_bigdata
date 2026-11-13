"""Compare the futures-data daily NQ series with Yahoo Finance daily data.

Light check, no model: daily candles built from the TradeStation 1-minute
bars (in-schedule CME session, the same calendar as the study) against
Yahoo's NQ=F (front-month futures, not back-adjusted) and ^NDX (cash index).

  python tools/yahoo_compare.py <yahoo_dir> [<out_dir> ...]

<yahoo_dir> (e.g. data_raw/yahoo, gitignored) holds the raw Yahoo chart-API
JSON files raw_NQ%3DF.json and raw_%5ENDX.json; missing files are downloaded
(public chart endpoint, daily bars 2000-2025). The summary
yahoo_check.json (statistics only, no price series) is written to
<yahoo_dir> and to every <out_dir> (reports/multitarget_final, presentation).
"""
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config, data, timeframes  # noqa: E402

CT = "America/Chicago"


URL = ("https://query1.finance.yahoo.com/v8/finance/chart/{}?period1=946684800&period2=1767225600"
       "&interval=1d&events=div%2Csplit")


def download(ydir):
    ydir.mkdir(parents=True, exist_ok=True)
    for sym in ("NQ%3DF", "%5ENDX"):
        f = ydir / f"raw_{sym}.json"
        if not f.exists():
            req = urllib.request.Request(URL.format(sym), headers={"User-Agent": "Mozilla/5.0"})
            f.write_bytes(urllib.request.urlopen(req, timeout=60).read())


def yahoo(path):
    r = json.loads(Path(path).read_text())["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    tz = r["meta"]["exchangeTimezoneName"]
    d = pd.to_datetime(r["timestamp"], unit="s", utc=True).tz_convert(tz).tz_localize(None).normalize()
    df = pd.DataFrame({k: q[k] for k in ("open", "high", "low", "close")}, index=d).dropna()
    return df[~df.index.duplicated(keep="last")]


def futures_daily(y0, y1):
    raw = data.load_futures_minutes(config.futures_data_dir(), config.FUTURES_SYMBOL, y0, y1).reset_index()
    bars, _ = timeframes.from_tradestation_minutes(raw)
    m, _ = timeframes.add_sessions(bars)
    ct = pd.DatetimeIndex(m["bar_start"]).tz_convert(CT)
    m["cash"] = (ct.hour * 60 + ct.minute) < 15 * 60          # bars before 15:00 CT = 16:00 ET
    g = m.groupby("session_id")
    d = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                      "close": g["close"].last()})
    d["close_1500ct"] = m[m["cash"]].groupby("session_id")["close"].last()
    d.index = pd.DatetimeIndex(d.index).tz_localize(None) if getattr(d.index, "tz", None) else pd.DatetimeIndex(d.index)
    return d


def compare(a, b, name):
    """a, b: close series on common dates (consecutive common days)."""
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1, sort=True).dropna()
    ra, rb = np.log(j["a"]).diff(), np.log(j["b"]).diff()
    gap = j.index.to_series().diff().dt.days
    ok = ra.notna() & (gap <= 4)
    ra, rb = ra[ok], rb[ok]
    diff = 100 * (ra - rb)
    beta = float(np.polyfit(rb, ra, 1)[0])
    return {"pair": name, "n_days": int(ok.sum()), "from": str(ra.index[0].date()), "to": str(ra.index[-1].date()),
            "corr_returns": round(float(ra.corr(rb)), 4),
            "corr_abs_returns": round(float(ra.abs().corr(rb.abs())), 4),
            "beta_a_on_b": round(beta, 4),
            "std_ret_a_pct": round(100 * float(ra.std()), 4), "std_ret_b_pct": round(100 * float(rb.std()), 4),
            "mean_abs_diff_pp": round(float(diff.abs().mean()), 4),
            "median_abs_diff_pp": round(float(diff.abs().median()), 4),
            "p95_abs_diff_pp": round(float(diff.abs().quantile(0.95)), 4),
            "share_same_sign": round(float((np.sign(ra) == np.sign(rb)).mean()), 4),
            "worst_days": [{"date": str(k.date()), "ret_a_pct": round(100 * ra[k], 3), "ret_b_pct": round(100 * rb[k], 3)}
                           for k in diff.abs().nlargest(8).index]}, diff


TOL_PP = 0.5          # day disagrees if |futures - reference| daily return > 0.5 pp (p95 of the gap: 0.16 pp vs ^NDX)
SPLIT_EDGES = ("2017-08-25", "2021-07-15", "2025-06-02")


def day_checks(fut, nqf, ndx, offs, flags_json):
    """Per daily session: do the re-levelled daily RETURNS agree with Yahoo?

    Close-to-close log returns of the re-levelled futures series (15:00 CT vs
    ^NDX, session close vs NQ=F) and the daily range vs NQ=F. A session is
    'confirmed' when its 15:00 CT return is within TOL_PP of ^NDX, or its
    session-close return within TOL_PP of NQ=F (NQ=F has its own roll errors,
    ^NDX misses the overnight hours, so either reference may be off alone).
    Closing returns only: the ranges are reported but NOT part of the rule, so
    a session with a wrong high/low is still 'confirmed' if its closes agree.
    This does not validate the whole candle, single minutes or realized
    volatility. The re-levelling offset is estimated from Yahoo itself (not
    an independent validation, not the contracts' original prices).
    """
    o = offs.reindex(fut.index).ffill().bfill()
    d = pd.DataFrame(index=fut.index)
    d["r_fut_1500"] = 100 * np.log(fut["close_1500ct"] + o).diff()
    d["r_fut_close"] = 100 * np.log(fut["close"] + o).diff()
    d["r_ndx"] = 100 * np.log(ndx["close"]).reindex(fut.index).diff()
    d["r_nqf"] = 100 * np.log(nqf["close"]).reindex(fut.index).diff()
    d["range_fut"] = 100 * np.log((fut["high"] + o) / (fut["low"] + o))
    d["range_nqf"] = 100 * np.log(nqf["high"] / nqf["low"]).reindex(fut.index)
    d["gap_ndx"] = (d["r_fut_1500"] - d["r_ndx"]).abs()
    d["gap_nqf"] = (d["r_fut_close"] - d["r_nqf"]).abs()
    has_ref = d["gap_ndx"].notna() | d["gap_nqf"].notna()
    d["confirmed"] = (d["gap_ndx"] <= TOL_PP) | (d["gap_nqf"] <= TOL_PP)
    d["status"] = np.where(~has_ref, "no_reference", np.where(d["confirmed"], "confirmed", "disagrees"))
    a, b, c = (pd.Timestamp(x) for x in SPLIT_EDGES)
    d["split"] = np.where(d.index < a, "train", np.where(d.index < b, "val", np.where(d.index < c, "test", "after")))
    # flag sessions (in-schedule flags, tools/sample_checks.py) and the study's 50-session window
    fl = [m for m in flags_json if m.get("session")]
    fs_ = pd.Series(0, index=d.index)
    for m in fl:
        t = pd.Timestamp(m["session"])
        if t in fs_.index:
            fs_[t] = 1
    d["flag_session"] = fs_.astype(bool)
    d["in_50_window"] = fs_.rolling(50, min_periods=1).max().astype(bool)
    rows = []
    for m in fl:
        t = pd.Timestamp(m["session"])
        r = d.loc[t] if t in d.index else None
        rows.append({"session": m["session"], "category": m["category"], "flag_log_return_pct": m["flag_log_return_pct"],
                     **({k: (None if pd.isna(r[k]) else round(float(r[k]), 3)) for k in
                         ("r_fut_1500", "r_ndx", "r_fut_close", "r_nqf", "range_fut", "range_nqf")} if r is not None else {}),
                     "status": None if r is None else r["status"]})
    def grp(m):
        x = d[m & (d["status"] != "no_reference")]
        return {"n": int(len(x)), "confirmed_pct": round(100 * float(x["confirmed"].mean()), 2),
                "median_gap_ndx_pp": round(float(x["gap_ndx"].median()), 3),
                "p95_gap_ndx_pp": round(float(x["gap_ndx"].quantile(0.95)), 3),
                "no_reference": int((m & (d["status"] == "no_reference")).sum())}
    test = d["split"] == "test"
    out = {"tolerance_pp": TOL_PP, "flag_sessions": rows,
           "test_in_50_window": grp(test & d["in_50_window"]), "test_outside_window": grp(test & ~d["in_50_window"]),
           "all_in_50_window": grp(d["in_50_window"] & (d["split"] != "after")),
           "all_outside_window": grp(~d["in_50_window"] & (d["split"] != "after")),
           "test_sessions": int(test.sum()), "test_in_50_window_count": int((test & d["in_50_window"]).sum()),
           "disagreeing_sessions": [{"session": str(k.date()), "split": v["split"], "r_fut_1500": round(v["r_fut_1500"], 3),
                                     "r_ndx": None if pd.isna(v["r_ndx"]) else round(v["r_ndx"], 3),
                                     "r_fut_close": round(v["r_fut_close"], 3),
                                     "r_nqf": None if pd.isna(v["r_nqf"]) else round(v["r_nqf"], 3)}
                                    for k, v in d[d["status"] == "disagrees"].iterrows()]}
    return out, d


def main(ydir, *outs):
    ydir = Path(ydir)
    download(ydir)
    nqf, ndx = yahoo(ydir / "raw_NQ%3DF.json"), yahoo(ydir / "raw_%5ENDX.json")
    y0, y1 = config.FUTURES_YEARS
    fut = futures_daily(y0, y1)
    out = {"futures_sessions": len(fut), "futures_range": [str(fut.index[0].date()), str(fut.index[-1].date())]}

    # 1) same instrument: futures-data session close vs Yahoo NQ=F close
    out["fut_vs_yahoo_NQF"], d1 = compare(fut["close"], nqf["close"], "futures-data close vs Yahoo NQ=F close")
    # 2) futures 15:00 CT (cash close time) vs NDX cash close
    out["fut1500_vs_NDX"], d2 = compare(fut["close_1500ct"], ndx["close"], "futures-data 15:00 CT vs ^NDX close")
    out["yahooNQF_vs_NDX"], _ = compare(nqf["close"], ndx["close"], "Yahoo NQ=F vs ^NDX")

    # 3) daily range (log high/low), the volatility proxy closest to the study
    j = pd.concat([np.log(fut["high"] / fut["low"]).rename("f"), np.log(nqf["high"] / nqf["low"]).rename("y")],
                  axis=1, sort=True).dropna()
    j = j[(j > 0).all(axis=1)]
    out["range_vs_yahoo_NQF"] = {"n": len(j), "corr_log_range": round(float(np.log(j["f"]).corr(np.log(j["y"]))), 4),
                                 "median_ratio": round(float((j["f"] / j["y"]).median()), 4)}

    # 4) level offset = back-adjustment (Yahoo unadjusted minus futures-data adjusted)
    lv = pd.concat([nqf["close"].rename("y"), fut["close"].rename("f")], axis=1, sort=True).dropna()
    off = lv["y"] - lv["f"]
    yearly = off.groupby(off.index.year).median().round(1)
    out["level_offset_median_by_year"] = {int(k): float(v) for k, v in yearly.items()}
    out["price_ratio_unadj_over_adj_by_year"] = {int(k): round(float(v), 4) for k, v in
                                                 (lv["y"] / lv["f"]).groupby(lv.index.year).median().items()}
    for dt in ["2010-03-16", "2010-03-17", "2013-03-12", "2017-03-24", "2019-03-14", "2023-03-16"]:
        t = pd.Timestamp(dt)
        if t in off.index:
            out.setdefault("offset_on_artefact_dates", {})[dt] = round(float(off[t]), 1)
    # 5) undo the additive back-adjustment approximately: offset = rolling median of
    #    (Yahoo NQ=F - futures-data) over 21 common days, then returns on the
    #    rebuilt unadjusted level. If the remaining gap vanishes, the back-adjustment
    #    explains the difference in return scale.
    offs = off.rolling(21, center=True, min_periods=5).median()
    unadj = (fut["close"] + offs.reindex(fut.index).ffill().bfill())
    out["unadj_fut_vs_yahoo_NQF"], _ = compare(unadj, nqf["close"], "futures-data re-levelled vs Yahoo NQ=F")
    un1500 = fut["close_1500ct"] + offs.reindex(fut.index).ffill().bfill()
    out["unadj_fut1500_vs_NDX"], _ = compare(un1500, ndx["close"], "futures-data 15:00 CT re-levelled vs ^NDX")
    # per period: scale of daily log returns, futures-data vs ^NDX
    per = {}
    for lab, a, b in [("2006-2012", "2006", "2012"), ("2013-2020", "2013", "2020"),
                      ("train-val to 2021-07-14", "2006", "2021-07-14"), ("test 2021-07-15..2025-05-30", "2021-07-15", "2025")]:
        r, _ = compare(fut["close_1500ct"][a:b], ndx["close"][a:b], lab)
        ru, _ = compare(un1500[a:b], ndx["close"][a:b], lab)
        per[lab] = {"beta_adj": r["beta_a_on_b"], "corr_adj": r["corr_returns"],
                    "std_ratio_adj": round(r["std_ret_a_pct"] / r["std_ret_b_pct"], 4),
                    "beta_relevelled": ru["beta_a_on_b"], "corr_relevelled": ru["corr_returns"],
                    "std_ratio_relevelled": round(ru["std_ret_a_pct"] / ru["std_ret_b_pct"], 4)}
    out["by_period_vs_NDX"] = per
    scj = Path(__file__).resolve().parents[1] / "reports" / "multitarget_final" / "sample_checks.json"
    out["day_checks"], dt = day_checks(fut, nqf, ndx, offs, json.load(open(scj))["daily_sample"]["flagged_minutes"])
    out["note"] = ("Yahoo NQ=F: front-month futures, not back-adjusted (own roll artefacts, e.g. 2020-03-23); "
                   "^NDX: cash index close 16:00 ET. Re-levelled = futures-data + rolling 21-day median of "
                   "(Yahoo NQ=F - futures-data), a retrospective estimate of the unadjusted price built from "
                   "Yahoo itself (not the contracts' original prices, not an independent validation). "
                   "day_checks 'confirmed' = close-to-close returns within tolerance_pp only; ranges are "
                   "reported, not used; minutes and realized volatility are not validated.")
    for d in (ydir, *map(Path, outs)):
        json.dump(out, open(d / "yahoo_check.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
