"""
Timestral vs the trained models on EXACTLY the same test rows (Timestral's fixed
random sample), from the saved forecasts: no training, no Timestral re-run.

For every timeframe it reports, on the Timestral test sample:
  return      R^2 vs the zero forecast (Timestral, best network of the task,
              XGBoost)
  direction   accuracy minus the majority class on the BODY target the study
              uses (C(t+1) vs O(t+1)) AND on the CLOSE-TO-CLOSE target
              (C(t+1) vs C(t)), which is what Timestral's median actually
              forecasts; dojis / zero returns count as misses in both
  volatility  log-RV R^2 vs persistence; Timestral's forecast is its
              q10-q90 price-forecast spread mapped to log RV by the calibration
              fitted on validation, the others forecast log RV directly; plus
              the raw correlation of Timestral's spread with next log RV
Intervals: 95% moving-block bootstrap, 20-session blocks (as in the report).
The networks were trained for the body target; the close-to-close row for them
is shown only to separate "target mismatch" from "no signal".

Usage: python tools/timestral_same_sample.py results/multitarget/runs/<id> reports/multitarget_final
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import multitarget_report as mr  # noqa: E402


def main(run, out):
    run, out = Path(run), Path(out)
    info = json.loads((run / "dataset_info.json").read_text())
    cmp_ = pd.read_csv(out / "comparison_by_task.csv")
    best = {t: cmp_[(cmp_.task == t) & cmp_.model.str.startswith("mlp_")].model.iloc[0] for t in ("return", "direction", "volatility")}
    rows = []
    for freq, fi in info.items():
        p = pd.read_parquet(run / f"predictions_{freq}.parquet")
        q = p[(p.split == "test") & p.timestral_sample]
        if q.empty:
            continue
        sess = q.label_session_id.to_numpy()
        tag = {"frequency": freq, "n": len(q)}
        r = q.ret_bps.to_numpy()
        for m in ("timestral", best["return"], "xgboost"):
            e = (r - q[f"ret__{m}"].to_numpy()) ** 2
            est, lo, hi = mr.block_boot_ratio(e, r ** 2, sess)
            rows.append({**tag, "task": "return", "target": "log return", "model": m, "metric": "R2_vs_zero_pct",
                         "value": 100 * est, "ci_lo": 100 * lo, "ci_hi": 100 * hi})
        y_body = q.y3.to_numpy()
        y_cc = np.sign(r).astype(int)
        maj = info[freq]["consts"]["majority_direction"]
        for m in ("timestral", best["direction"], "xgboost"):
            call = np.where(q[f"dir__{m}"].to_numpy() >= 0.5, 1, -1)
            for tname, y in (("body C(t+1) vs O(t+1)", y_body), ("close-to-close C(t+1) vs C(t)", y_cc)):
                d = (call == y).astype(float) - (y == maj)
                est, lo, hi, _ = mr.block_boot(d, sess)
                rows.append({**tag, "task": "direction", "target": tname, "model": m, "metric": "acc_minus_majority_pp",
                             "value": 100 * est, "ci_lo": 100 * lo, "ci_hi": 100 * hi,
                             "accuracy": float((call == y).mean())})
        lv, ln = q.lrv_next.to_numpy(), q.lrv_now.to_numpy()
        e_p = (lv - ln) ** 2
        for m, col in (("timestral (spread + calibration)", "vol__timestral"), (best["volatility"], f"vol__{best['volatility']}"),
                       ("xgboost", "vol__xgboost"), ("har_ols", "vol__har_ols")):
            f = q[col].to_numpy()
            est, lo, hi = mr.block_boot_ratio((lv - f) ** 2, e_p, sess)
            rows.append({**tag, "task": "volatility", "target": "log RV", "model": m, "metric": "R2_log_vs_persistence_pct",
                         "value": 100 * est, "ci_lo": 100 * lo, "ci_hi": 100 * hi, "corr_with_next_log_rv": float(np.corrcoef(f, lv)[0, 1])})
        rows.append({**tag, "task": "volatility", "target": "log RV", "model": "timestral raw spread (uncalibrated)",
                     "metric": "corr_with_next_log_rv", "value": float(np.corrcoef(q.vol_raw__timestral, lv)[0, 1]),
                     "corr_with_next_log_rv": float(np.corrcoef(q.vol_raw__timestral, lv)[0, 1])})
        rows.append({**tag, "task": "volatility", "target": "log RV", "model": "persistence (current log RV)",
                     "metric": "corr_with_next_log_rv", "value": float(np.corrcoef(ln, lv)[0, 1]),
                     "corr_with_next_log_rv": float(np.corrcoef(ln, lv)[0, 1])})
    df = pd.DataFrame(rows)
    df.to_csv(out / "timestral_same_sample.csv", index=False)
    (out / "timestral_same_sample.json").write_text(df.to_json(orient="records", indent=1), encoding="utf-8")   # deck appendix
    pd.set_option("display.width", 220)
    print(df[df.frequency == "5min"].round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
