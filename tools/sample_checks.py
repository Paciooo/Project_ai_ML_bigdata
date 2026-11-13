"""
Sample checks (no training): rebuilds the candles from the 1-minute data and
measures two things the saved forecasts depend on.

A. Daily sample end. Traces, filter by filter, why daily rows after
   2025-01-30 are not in the data set, and how many daily rows of each split
   fall in the 50-session "no flagged minute" window of study.build. Lists
   every flagged 1-minute bar (|1-min log return| > 2%, quality.flag_suspect)
   with the gap to the previous bar, so a weekend/holiday opening gap can be
   told apart from an isolated bad print.

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

FREQS = ("5min", "15min", "30min", "1h")
SPLIT_EDGES = ("2017-08-25", "2021-07-15", "2025-06-02")


def split_of(sid):
    a, b, c = (pd.Timestamp(x) for x in SPLIT_EDGES)
    return np.where(sid < a, "train", np.where(sid < b, "val", np.where(sid < c, "test", "after")))


def daily_checks(minutes):
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
    lr = np.log(minutes["close"]).diff()
    gap = minutes["bar_start"].diff() / pd.Timedelta("1min")
    m = minutes[minutes["suspect"].astype(bool)]
    out["flagged_minutes"] = [{"session": str(r.session_id.date()), "bar_start_utc": str(r.bar_start),
                               "log_return_pct": round(100 * float(lr[i]), 2), "minutes_since_previous_bar": float(gap[i])}
                              for i, r in zip(m.index, m.itertuples())]
    out["flagged_after_gap_over_60min"] = int((gap[m.index] > 60).sum())
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
           "daily_sample": daily_checks(minutes), "incomplete_candles": candle_checks(minutes, run)}
    (out / "sample_checks.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    d = res["daily_sample"]
    print(json.dumps({k: d[k] for k in d if k != "flagged_minutes"}, indent=1))
    for f, r in res["incomplete_candles"].items():
        print(f, r["share_full_candles_with_missing_minutes_pct"], r["test_rows_label_candle_incomplete"], r["test_rescored"])
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
