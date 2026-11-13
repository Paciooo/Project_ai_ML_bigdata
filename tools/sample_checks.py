"""
Sample checks (no training): rebuilds the candles from the 1-minute data and
measures two things the saved forecasts depend on.

A. Daily sample end. Traces, filter by filter, why daily rows after
   2025-01-30 are not in the data set, and how many daily rows of each split
   fall in the 50-session "no flagged minute" window of study.build. Lists
   every flagged 1-minute bar (|1-min log return| > 2%, quality.flag_suspect)
   with the values that generated the flag: the flag is computed BEFORE the
   schedule filter (timeframes.add_sessions), so its predecessor can be an
   out-of-schedule bar that the experiment later drops. Those values come
   from experiment.load_minutes (report key suspect_flag_details); values
   recomputed on the filtered, in-schedule sequence are reported separately
   (filtered_*). Categories are descriptive and do not prove that a price is
   genuine or wrong.

B. Incomplete intraday candles. A full-length candle (not `is_partial`) can
   contain fewer 1-minute bars than scheduled: timeframes.aggregate counts
   them (`n_minutes`) but does not use the count. TradeStation has no bar for
   a minute without trades, so a missing minute is either a quiet minute or a
   dataset gap; scheduled halts and the overnight break are outside the bins.
   Classified per candle: complete; isolated missing minutes (session with
   <= 50 missing minutes in that timeframe); gap session (> 50). Then the
   saved TEST forecasts are rescored without the rows whose label candle is
   incomplete (point estimates only), to measure the impact.

Usage: python tools/sample_checks.py results/multitarget/runs/<id> reports/multitarget_final
       (needs the 1-minute data: FUTURES_DATA_DIR)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
import experiment  # noqa: E402
import multitarget  # noqa: E402
import study  # noqa: E402
import timeframes  # noqa: E402

FREQS = ("5min", "15min", "30min", "1h")
SPLIT_EDGES = ("2017-08-25", "2021-07-15", "2025-06-02")


def split_of(sid):
    a, b, c = (pd.Timestamp(x) for x in SPLIT_EDGES)
    return np.where(sid < a, "train", np.where(sid < b, "val", np.where(sid < c, "test", "after")))


def daily_checks(minutes, rep):
    b, elig = experiment.bars_for(minutes, None, "session")
    b = b.reset_index(drop=True)
    X = multitarget.build(minutes, "session")[0]
    susp = b["suspect"].astype(bool)
    lookback = max([*config.RETURN_LAGS, *[n + 1 for n in config.MOMENTUM_WINDOWS],
                    *[n + 1 for n in config.VOLATILITY_WINDOWS], 21 if study.EXTRA_GROUPS else 0])
    window_clean = ~susp.rolling(lookback, min_periods=1).max().astype(bool)
    clean50 = ~susp.rolling(50, min_periods=1).max().astype(bool)
    sid = b["session_id"]
    tail = (sid > "2025-01-30") & (sid < SPLIT_EDGES[2])
    kept = b.index.isin(X.index)
    lost = tail & ~kept
    out = {"last_daily_test_session_kept": str(sid[kept & (sid < SPLIT_EDGES[2])].max().date()),
           "sessions_2025-01-31_to_2025-05-30": int(tail.sum()), "of_which_not_in_data_set": int(lost.sum()),
           "lost_reasons": {"label_session_contains_flagged_minute": int((lost & susp.shift(-1, fill_value=False)).sum()),
                            "flagged_minute_in_feature_window": int((lost & ~window_clean).sum()),
                            "flagged_minute_in_last_50_sessions (study.build)": int((lost & ~clean50).sum()),
                            "label_not_next_session": int((lost & ~pd.Series(elig)).sum())},
           "daily_rows_in_50_session_window_by_split": pd.Series(~clean50.to_numpy()).groupby(split_of(sid)).sum().astype(int).to_dict(),
           "daily_sessions_by_split": pd.Series(1, index=b.index).groupby(split_of(sid)).sum().astype(int).to_dict()}
    test_lost = (split_of(sid) == "test") & ~kept
    nxt = susp.shift(-1, fill_value=False).to_numpy()
    out["daily_test_sessions_not_in_data_set_by_condition (not exclusive)"] = {
        "total": int(test_lost.sum()),
        "label_session_contains_flagged_minute": int((test_lost & nxt).sum()),
        "flagged_minute_in_feature_window": int((test_lost & ~window_clean.to_numpy()).sum()),
        "flagged_minute_in_last_50_sessions (study.build)": int((test_lost & ~clean50.to_numpy()).sum()),
        "label_not_next_session": int((test_lost & ~np.asarray(elig)).sum()),
        "none_of_the_above": int((test_lost & clean50.to_numpy() & ~nxt & np.asarray(elig)).sum())}
    out.update(flag_checks(minutes, rep, b, sid, susp, clean50, kept))
    return out


PERSIST_MIN = 30      # in-schedule minutes after the flag used to describe what the price did next


def flag_checks(minutes, rep, b, sid, susp, clean50, kept):
    """Flagged 1-minute bars, described with the values that GENERATED each
    flag (quality.suspect_details on the unfiltered sequence, stored by
    experiment.load_minutes) and, separately labelled, with values recomputed
    on the in-schedule sequence the experiment uses. Categories are
    descriptive; none of them proves that a price was genuine or wrong."""
    thr = 100 * config.SPIKE_THRESHOLD
    det = rep["suspect_flag_details"]
    t_in = pd.DatetimeIndex(minutes["bar_start"])
    pos = pd.Series(np.arange(len(minutes)), index=t_in)
    lc = np.log(minutes["close"].to_numpy(float))
    rows = []
    for d in det:
        t, tp = pd.Timestamp(d["bar_start_utc"]), pd.Timestamp(d["prev_bar_start_utc"])
        cal = timeframes.session_calendar(pd.DatetimeIndex([t, tp]))
        r = dict(d, consistent_with_threshold=bool(abs(d["flag_log_return_pct"]) > thr
                                                  and abs(100 * np.log(d["close"] / d["prev_close"]) - d["flag_log_return_pct"]) < 1e-3),
                 bar_in_schedule=bool(cal.in_schedule[0]), prev_in_schedule=bool(cal.in_schedule[1]),
                 bar_time_et=str(t.tz_convert("America/New_York").strftime("%a %H:%M")))
        if r["bar_in_schedule"]:
            i = int(pos[t])
            r["session"] = str(minutes["session_id"].iloc[i].date())
            r["filtered_prev_bar_start_utc"] = str(minutes["bar_start"].iloc[i - 1])
            r["filtered_log_return_pct"] = round(100 * float(lc[i] - lc[i - 1]), 4)
            j = min(i + PERSIST_MIN, len(lc) - 1)
            r[f"log_change_flag_prev_to_{PERSIST_MIN}_in_schedule_min_later_pct"] = round(100 * float(lc[j] - np.log(d["prev_close"])), 3)
            later = r[f"log_change_flag_prev_to_{PERSIST_MIN}_in_schedule_min_later_pct"]
            if not r["prev_in_schedule"] and abs(r["filtered_log_return_pct"]) < thr:
                r["category"] = "out_of_schedule_predecessor"     # flag measured against a bar the experiment drops
            elif np.sign(later) == np.sign(d["flag_log_return_pct"]) and abs(later) > thr:
                r["category"] = "move_persists_30min"
            else:
                r["category"] = "unresolved"
        else:
            r["category"] = "flagged_bar_out_of_schedule"         # dropped by add_sessions, no effect on the sample
        rows.append(r)
    out = {"flag_threshold_pct": thr, "flagged_minutes": rows,
           "n_flags_unfiltered_sequence": len(rows),
           "n_flags_in_schedule": sum(r["bar_in_schedule"] for r in rows),
           "n_flags_inconsistent_with_threshold": sum(not r["consistent_with_threshold"] for r in rows),
           "flag_categories_in_schedule": pd.Series([r["category"] for r in rows if r["bar_in_schedule"]]).value_counts().to_dict(),
           "flagged_in_schedule_after_gap_over_60min": sum(r["bar_in_schedule"] and r["minutes_since_prev"] > 60 for r in rows)}
    # daily test sessions in the 50-session window, by the category of the flags behind the window
    sess_cat = {}
    for r in rows:
        if r["bar_in_schedule"]:
            sess_cat.setdefault(pd.Timestamp(r["session"]), set()).add(r["category"])
    sp = split_of(sid)
    win = {}
    for c in ("out_of_schedule_predecessor", "move_persists_30min", "unresolved"):
        s = pd.Series([c in sess_cat.get(pd.Timestamp(x), ()) for x in sid], index=b.index)
        win[c] = s.rolling(50, min_periods=1).max().astype(bool).to_numpy()
    inw = ~clean50.to_numpy()
    test = sp == "test"
    only = lambda c: inw & win[c] & ~np.logical_or.reduce([win[o] for o in win if o != c])
    out["daily_test_sessions_in_50_session_window_by_flag_category"] = {
        "total_in_window": int((test & inw).sum()),
        **{f"only_{c}": int((test & only(c)).sum()) for c in win},
        "several_categories": int((test & inw & (np.sum([win[c] for c in win], axis=0) > 1)).sum())}
    out["daily_test_sessions_accounting"] = {
        "test_sessions": int(test.sum()), "in_data_set": int((test & kept).sum()),
        "not_in_data_set": int((test & ~kept).sum()),
        "not_in_data_set_and_in_50_session_window": int((test & ~kept & inw).sum()),
        "not_in_data_set_outside_window": int((test & ~kept & ~inw).sum()),
        "in_window_but_in_data_set": int((test & kept & inw).sum())}
    return out


def candle_checks(minutes, run):
    out = {}
    info = json.loads((run / "dataset_info.json").read_text())
    for freq in FREQS:
        bb, _ = experiment.bars_for(minutes, None, freq)
        bb = bb.reset_index(drop=True)
        sched = ((bb["bar_end"] - bb["bar_start"]) / pd.Timedelta("1min")).round().astype(int)
        full = ~bb["is_partial"].astype(bool)
        miss = (sched - bb["n_minutes"]).where(full, 0)
        gap_sess = miss.groupby(bb["session_id"]).transform("sum") > 50
        cls = pd.Series(np.where(miss == 0, "complete", np.where(gap_sess, "gap_session", "isolated")))
        ct = pd.DatetimeIndex(bb["bar_start"]).tz_convert("America/Chicago")
        tod = ct.hour * 60 + ct.minute
        rth = (tod >= 8 * 60 + 30) & (tod < 15 * 60)
        sp = split_of(bb["session_id"])
        year = pd.DatetimeIndex(bb["session_id"]).year
        r = {"share_full_candles_with_missing_minutes_pct": {
                 "all_years": round(100 * float((miss[full] > 0).mean()), 3),
                 "test": round(100 * float((miss[full & (sp == "test")] > 0).mean()), 3),
                 "test_08:30-15:00_CT": round(100 * float((miss[full & (sp == "test") & rth] > 0).mean()), 3),
                 "test_other_hours": round(100 * float((miss[full & (sp == "test") & ~rth] > 0).mean()), 3)},
             "by_year_pct": {int(k): round(v, 2) for k, v in (100 * (miss[full] > 0)).groupby(year[full]).mean().items()},
             "test_class_counts": cls[full & (sp == "test")].value_counts().to_dict(),
             "test_gap_sessions": sorted({str(s.date()) for s in bb.session_id[(cls == "gap_session") & (sp == "test")]})}
        p = pd.read_parquet(run / f"predictions_{freq}.parquet")
        q = p[p.split == "test"]
        lab = pd.Series(cls.to_numpy(), index=pd.DatetimeIndex(bb["bar_start"]))
        lc = lab.reindex(pd.DatetimeIndex(q.label_start)).to_numpy()
        ok = lc == "complete"
        r["test_rows"] = int(len(q))
        r["test_rows_label_candle_incomplete"] = int((~ok).sum())
        lv, ln, rr, y = q.lrv_next.to_numpy(), q.lrv_now.to_numpy(), q.ret_bps.to_numpy(), q.y3.to_numpy()
        maj = info[freq]["consts"]["majority_direction"]
        sens = {}
        for name, k in (("all_rows", np.ones(len(q), bool)), ("complete_label_candles_only", ok)):
            s = {}
            for m in ("har_ols", "mlp_128-64-32", "xgboost"):
                f = q[f"vol__{m}"].to_numpy()
                s[f"vol_R2_log_vs_persistence_pct_{m}"] = round(100 * (1 - ((lv - f)[k] ** 2).sum() / ((lv - ln)[k] ** 2).sum()), 3)
            call = np.where(q["dir__mlp_256-128-64-32"].to_numpy() >= 0.5, 1, -1)
            s["dir_gain_pp_mlp_256-128-64-32"] = round(100 * ((call == y)[k].mean() - (y == maj)[k].mean()), 3)
            f = q["ret__mlp_256-128-64-32"].to_numpy()
            s["ret_R2_pct_mlp_256-128-64-32"] = round(100 * (1 - ((rr - f)[k] ** 2).sum() / (rr[k] ** 2).sum()), 3)
            sens[name] = s
        r["test_rescored"] = sens
        out[freq] = r
    return out


def main(run, out):
    run, out = Path(run), Path(out)
    minutes, _, rep, _ = experiment.load_minutes("nq")
    res = {"note": "point estimates from saved forecasts; no training, no bootstrap",
           "daily_sample": daily_checks(minutes, rep), "incomplete_candles": candle_checks(minutes, run)}
    (out / "sample_checks.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    d = res["daily_sample"]
    print(json.dumps({k: d[k] for k in d if k != "flagged_minutes"}, indent=1))
    for f, r in res["incomplete_candles"].items():
        print(f, r["share_full_candles_with_missing_minutes_pct"], r["test_rows_label_candle_incomplete"], r["test_rescored"])
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
