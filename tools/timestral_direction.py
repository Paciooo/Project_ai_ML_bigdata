"""
Zero-shot UP/DOWN with Mistral's Timestral on common forecast origins.

Fair-comparison protocol:
  * forecast ORIGINS are fixed once per (data source, frequency, target,
    sample size, seed) and saved as a manifest
    (results/<source>/timestral/origins_<freq>_<target>_n<N>_s<seed>.parquet);
    every Timestral size and every baseline is scored on exactly those rows.
    The sample of a frequency does not depend on which other frequencies are
    run (its own random generator, seeded from the seed and the frequency);
  * origins are eligible test rows of the experiment pipeline (same bars,
    sessions, quarantine, target, purged holdout split) with at least
    --context bars of history of the same frequency;
  * the named baseline is "close-median sign": UP if the forecast median of
    the next close is above the last close (float32, as the model sees it).
    Ties are counted. Quantiles are saved (q10/q90) but NOT used as
    probabilities of an up move;
  * --baseline-run joins the per-origin predictions of an experiment run
    (logreg, xgboost, rules) for paired comparisons on the same rows;
  * inference time, device, model id and revision are saved per run.
With 500-2,000 origins the 95% interval of an accuracy is about +-2 to
+-4.4 pp: differences of ~1 pp between sizes cannot be ranked reliably.
Pretraining-data overlap with NQ is UNKNOWN (no documentation establishes it).

Setup: pip install -r requirements-timestral.txt (needs torch >= 2.4).
Hugging Face downloads use plain HTTPS: HF_HUB_DISABLE_XET=1 is set before
any Hugging Face import unless you set the variable yourself.

Usage:
  python tools/timestral_direction.py --model mistralai/Timestral-5M --frequency 1min 1h session
  python tools/timestral_direction.py --model mistralai/Timestral-209M --n-samples 500
  python tools/timestral_direction.py --smoke          # offline: fixture data + a tiny random model
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")             # before any HF-dependent import
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
import evaluation  # noqa: E402
import experiment  # noqa: E402
import features  # noqa: E402
import runlog  # noqa: E402


def positive_int(s):
    v = int(s)
    if v <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return v


def eligible_origins(minutes, freq, target, context, seed, holdout_start):
    """Test rows of the holdout split with >= context bars of history."""
    bars, elig = experiment.bars_for(minutes, None, freq)
    X, y3, meta, _ = features.build_timed_dataset(
        bars, freq, target, config.RETURN_LAGS, config.MOMENTUM_WINDOWS, config.VOLATILITY_WINDOWS,
        label_eligible=elig)
    sessions = pd.DatetimeIndex(sorted(minutes["session_id"].unique()))
    usable = sessions[sessions < holdout_start] if holdout_start is not None else sessions
    (_, v, t, _), = features.walk_forward_folds(usable, "holdout", config.TRAIN_FRAC, config.VAL_FRAC)
    opens = minutes.groupby("session_id")["session_open"].first()
    tr, va, te = features.purged_split(meta, v, t, lambda s: opens.loc[s], end=holdout_start, min_rows=1)
    pos = np.flatnonzero(te)
    pos = pos[meta.index[pos] >= context - 1]                # meta.index = position in bars
    o = meta.iloc[pos].copy()
    o["bar_pos"], o["y3"] = meta.index[pos], y3.iloc[pos].to_numpy()
    o["ret_lag1"] = X["ret_lag1"].iloc[pos].to_numpy()
    tr_y = y3[tr]
    o.attrs["majority"] = 1 if (tr_y[tr_y != 0] > 0).mean() >= 0.5 else -1
    return o, bars["close"].to_numpy(float)


def sample_origins(o, n, seed, freq):
    rng = np.random.default_rng([seed, zlib.crc32(freq.encode())])   # independent per frequency
    if len(o) > n:
        o = o.iloc[np.sort(rng.choice(len(o), size=n, replace=False))]
    return o


class FakeForecaster:
    """Offline smoke stand-in (NOT Timestral): median = last value + noise."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def forecast(self, ctx, prediction_length):
        import torch
        last = torch.as_tensor(ctx[:, -1:], dtype=torch.float32)
        q = last[:, None, :, None] + torch.tensor([-1.0, 0.0, 1.0]) + float(self.rng.normal(0, 0.1))
        return type("F", (), {"median": q[..., 1], "quantiles": q})


def tiny_random_checkpoint(folder):
    """A randomly initialised Timestral model saved like an export (interface test only)."""
    import torch
    from safetensors.torch import save_file
    from timestral import TimestralConfig, TimestralModel
    cfg = dict(patch_size=8, d_model=32, num_heads=4, mlp_hidden_dim=64,
               block_schedule=("time", "time_variate"), quantile_knots=(0.1, 0.25, 0.5, 0.75, 0.9),
               block_size=4, context_length=256)
    torch.manual_seed(0)
    m = TimestralModel(TimestralConfig(**cfg))
    Path(folder).mkdir(parents=True, exist_ok=True)
    save_file(m.state_dict(), str(Path(folder) / "model.safetensors"))
    (Path(folder) / "config.json").write_text(json.dumps({**cfg, "block_schedule": list(cfg["block_schedule"]),
                                                         "quantile_knots": list(cfg["quantile_knots"])}))
    return folder


def forecast_origins(fc, o, close, context, batch):
    med, q10, q90, ties = [], [], [], 0
    last32 = []
    for i in range(0, len(o), batch):
        p = o["bar_pos"].to_numpy()[i:i + batch]
        ctx = np.stack([close[j - context + 1:j + 1] for j in p])
        out = fc.forecast(ctx, prediction_length=1)
        m = out.median[:, 0, 0].cpu().numpy().astype(np.float64)
        med.append(m)
        if hasattr(out, "quantile_levels"):
            lv = out.quantile_levels
            q = out.quantiles[:, 0, 0, :].cpu().numpy()
            q10.append(q[:, int(np.argmin(np.abs(np.array(lv) - 0.1)))])
            q90.append(q[:, int(np.argmin(np.abs(np.array(lv) - 0.9)))])
        last32.append(ctx[:, -1].astype(np.float32).astype(np.float64))
        print(f"  {min(i + batch, len(o)):,}/{len(o):,} forecasts", end="\r")
    print()
    med, last = np.concatenate(med), np.concatenate(last32)
    res = {"median": med, "last_f32": last, "pred": np.where(med > last, 1, -1), "tie": med == last}
    if q10:
        res["q10"], res["q90"] = np.concatenate(q10), np.concatenate(q90)
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="mistralai/Timestral-5M", help="HF repo id or a local export folder")
    p.add_argument("--revision", default=None, help="HF revision (commit/tag); default: latest")
    p.add_argument("--data-source", choices=["nq", "fixture"], default="nq")
    p.add_argument("--target", choices=["next_close_direction", "next_candle_body_direction"],
                   default="next_close_direction")
    p.add_argument("--frequency", nargs="+", default=["1min", "1h", "session"],
                   choices=["1min", "5min", "15min", "1h", "4h", "session"])
    p.add_argument("--context", type=positive_int, default=512)
    p.add_argument("--n-samples", type=positive_int, default=2000, help="common origins per frequency")
    p.add_argument("--batch", type=positive_int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=config.SEEDS[0])
    p.add_argument("--years", type=int, nargs=2, default=None)
    p.add_argument("--baseline-run", default=None, help="experiment run folder for paired baselines")
    p.add_argument("--smoke", action="store_true", help="offline: fixture data and a tiny RANDOM model")
    a = p.parse_args(argv)
    if a.smoke:
        a.data_source, a.context, a.n_samples, a.frequency = "fixture", 64, 200, ["1min", "5min"]

    minutes, _, rep, contract = experiment.load_minutes(a.data_source, a.years, smoke=a.smoke)
    holdout = pd.Timestamp(config.CONFIRMATORY_HOLDOUT_START)
    base = Path(runlog.PROJECT_DIR) / config.RESULTS_DIR / ("smoke" if a.smoke else a.data_source) / "timestral"
    base.mkdir(parents=True, exist_ok=True)

    if a.smoke:
        try:
            from timestral import TimestralForecaster
            fc = TimestralForecaster.from_pretrained(tiny_random_checkpoint(base / "tiny_random_model"))
            model_id = "tiny RANDOM Timestral (interface test, not a forecaster)"
        except ImportError:
            fc, model_id = FakeForecaster(a.seed), "FakeForecaster (timestral not installed)"
    else:
        from timestral import TimestralForecaster
        fc = TimestralForecaster.from_pretrained(a.model, device=a.device, revision=a.revision)
        model_id = a.model
    run_dir = runlog.new_run_dir(config.RESULTS_DIR, ("smoke" if a.smoke else a.data_source),
                                 tag="timestral_" + Path(a.model).name if not a.smoke else "timestral_smoke")
    rows = []
    for freq in a.frequency:
        o, close = eligible_origins(minutes, freq, a.target, a.context, a.seed, holdout)
        if len(o) == 0:
            print(f"[{freq}] no eligible origins with {a.context} bars of history: skipped")
            rows.append({"frequency": freq, "skipped": "no eligible origins"})
            continue
        mf = base / f"origins_{freq}_{a.target}_ctx{a.context}_n{a.n_samples}_s{a.seed}.parquet"
        if mf.exists():
            keep = pd.read_parquet(mf)["bar_start"]
            o = o[o["bar_start"].isin(keep)]
            if len(o) != len(keep):
                raise SystemExit(f"{mf} does not match the current data ({len(o)} of {len(keep)} origins)")
        else:
            o = sample_origins(o, a.n_samples, a.seed, freq)
            o[["bar_start", "decision_at", "label_end", "session_id", "y3"]].to_parquet(mf, index=False)
        t0 = time.time()
        res = forecast_origins(fc, o, close, a.context, a.batch)
        secs = time.time() - t0
        y3 = o["y3"].to_numpy()
        maj = o.attrs["majority"]
        out = o[["bar_start", "decision_at", "label_end", "session_id", "y3"]].copy()
        out["pred_timestral"], out["tie"], out["median"] = res["pred"], res["tie"], res["median"]
        for k in ("q10", "q90"):
            if k in res:
                out[k] = res[k]
        for rule in ("majority", "persistence", "reversal"):
            out[f"pred_{rule}"] = evaluation.rule_predictions(o, rule, maj)
        if a.baseline_run:
            bp = pd.read_parquet(Path(a.baseline_run) / f"predictions_{freq}.parquet")
            cols = [c for c in bp.columns if c.startswith("pred_") and c[5:] not in ("majority", "persistence", "reversal")]
            out = out.merge(bp[["bar_start", *cols]], on="bar_start", how="left")
        out.to_parquet(run_dir / f"timestral_predictions_{freq}.parquet", index=False)
        base_ok = out["pred_majority"].to_numpy() == y3
        for name in [c[5:] for c in out.columns if c.startswith("pred_")]:
            pred = out[f"pred_{name}"].to_numpy()
            if np.isnan(pred.astype(float)).any():
                continue
            m = evaluation.direction_metrics(pred.astype(int), y3)
            pdiff = evaluation.paired_difference(pred == y3, base_ok, out["session_id"], min_sessions=60)
            rows.append({"frequency": freq, "model": model_id if name == "timestral" else name,
                         "context": a.context, "n_origins": len(out), "acc_all": m["acc_all"],
                         "acc_nonflat": m["acc_nonflat"], "coverage_nonflat": m["coverage_nonflat"],
                         "acc_ci95_binomial_descriptive": 1.96 * np.sqrt(m["acc_all"] * (1 - m["acc_all"]) / len(out)),
                         "diff_vs_majority_pp": pdiff["diff_pp"], "ci_lo_pp_b5": pdiff["ci_lo_pp_b5"],
                         "ci_hi_pp_b5": pdiff["ci_hi_pp_b5"], "verdict": pdiff["verdict"],
                         "pred_up_share": m["pred_up_share"],
                         "ties": int(res["tie"].sum()) if name == "timestral" else None,
                         "seconds": round(secs, 1) if name == "timestral" else None})
    res = pd.DataFrame(rows)
    res.to_csv(run_dir / "timestral_results.csv", index=False, float_format="%.4f")
    runlog.write_run_json(run_dir, {"args": vars(a), "model_id": model_id, "revision": a.revision or "latest",
                                    "device": a.device, "data_report": rep, "contract": contract,
                                    "pretraining_overlap": "unknown",
                                    "baseline": "close-median sign (median > last close in float32)"})
    agg = Path(runlog.PROJECT_DIR) / config.RESULTS_DIR / ("smoke" if a.smoke else a.data_source) / "timestral_results_all.csv"
    res.assign(run=run_dir.name).to_csv(agg, mode="a", header=not agg.exists(), index=False, float_format="%.4f")
    show = [c for c in ["frequency", "model", "n_origins", "acc_all", "acc_nonflat", "diff_vs_majority_pp",
                        "ci_lo_pp_b5", "ci_hi_pp_b5", "verdict", "seconds"] if c in res]
    print(res[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nSaved to {run_dir} (and appended to {agg})")
    return run_dir


if __name__ == "__main__":
    main()
