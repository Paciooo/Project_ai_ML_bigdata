"""
Evaluation metrics: accuracy, balanced accuracy, F1 and confusion matrix.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score


def evaluate(model, X, y):
    """Return a dict of classification metrics for one model on one sample."""
    pred = model.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y, pred)
    return {
        "accuracy": acc,
        # 95% half-width of a binomial CI (normal approximation). A rough guide
        # to how much accuracy moves by chance alone, NOT a test against the
        # baseline: see `add_baseline_comparison` for that comparison.
        "acc_ci95": 1.96 * np.sqrt(acc * (1 - acc) / len(y)),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "f1": f1_score(y, pred, zero_division=0),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "pred_up_share": pred.mean(),
    }


def class_distribution(y):
    """Number of observations and share of UP moves."""
    return {"n": len(y), "n_up": int(y.sum()), "n_down": int(len(y) - y.sum()),
            "share_up": y.mean()}


def add_baseline_comparison(results):
    """Add each model's improvement over the majority baseline.

    Computed within the same seed and frequency (same test sample), in
    percentage points (pp):
      acc_vs_baseline_pp     = accuracy - baseline accuracy
      balacc_vs_baseline_pp  = balanced accuracy - 0.5 (the baseline's value)
    """
    base = (results[results["model"] == "Majority baseline"]
            .set_index(["seed", "frequency"])["accuracy"])
    key = list(zip(results["seed"], results["frequency"]))
    results["acc_vs_baseline_pp"] = 100 * (results["accuracy"].values - base.loc[key].values)
    results["balacc_vs_baseline_pp"] = 100 * (results["balanced_accuracy"] - 0.5)
    return results


# ---------------------------------------------------------------------------
# Three-class outcomes, simple rules, probabilities and paired bootstrap
# ---------------------------------------------------------------------------
#
# Flat policy. Outcomes are y3 in {-1, 0, +1}. Models predict UP (+1) or DOWN
# (-1). Two views are reported:
#   * "nonflat" (legacy benchmark): rows with y3 != 0 only, accuracy of the
#     sign; reported with its coverage (share of eligible rows that moved).
#   * "all": every eligible row; a flat outcome counts as a miss for any
#     UP/DOWN call (neither direction happened). Also used by the P&L.

from sklearn.metrics import brier_score_loss, log_loss  # noqa: E402


def rule_predictions(X, rule, majority):
    """Majority / persistence / reversal rules on the most recent return.

    persistence: same sign as ret_lag1; reversal: opposite sign. A zero last
    return carries no sign, so both fall back to the training majority.
    """
    if rule == "majority":
        return np.full(len(X), majority)
    s = np.sign(X["ret_lag1"].to_numpy())
    s = s if rule == "persistence" else -s
    return np.where(s == 0, majority, s).astype(int)


def direction_metrics(pred, y3, proba_up=None):
    """Metrics for UP/DOWN calls `pred` (+1/-1) against outcomes y3."""
    pred, y3 = np.asarray(pred), np.asarray(y3)
    nf = y3 != 0
    yb, pb = (y3[nf] > 0).astype(int), (pred[nf] > 0).astype(int)
    tn, fp, fn, tp = confusion_matrix(yb, pb, labels=[0, 1]).ravel()
    out = {
        "n_all": int(len(y3)), "n_nonflat": int(nf.sum()), "n_flat": int((~nf).sum()),
        "coverage_nonflat": float(nf.mean()) if len(y3) else np.nan,
        "acc_nonflat": float((pb == yb).mean()) if nf.any() else np.nan,
        "acc_all": float((pred == y3).mean()) if len(y3) else np.nan,
        "balanced_acc_nonflat": balanced_accuracy_score(yb, pb) if nf.any() else np.nan,
        "macro_f1_nonflat": f1_score(yb, pb, average="macro", zero_division=0) if nf.any() else np.nan,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "pred_up_share": float((pred > 0).mean()) if len(pred) else np.nan,
    }
    if proba_up is not None and nf.any():
        p = np.clip(np.asarray(proba_up)[nf], 1e-6, 1 - 1e-6)
        out["brier_nonflat"] = brier_score_loss(yb, p)
        out["log_loss_nonflat"] = log_loss(yb, p, labels=[0, 1])
        out["mean_proba_up"] = float(p.mean())
    return out


def calibration_table(proba_up, y3, n_bins=10):
    """Mean predicted P(up) vs observed up share, on non-flat rows, by decile."""
    nf = np.asarray(y3) != 0
    p, y = np.asarray(proba_up)[nf], (np.asarray(y3)[nf] > 0).astype(int)
    q = pd.qcut(p, n_bins, labels=False, duplicates="drop")
    return (pd.DataFrame({"bin": q, "p": p, "y": y}).groupby("bin")
            .agg(mean_pred=("p", "mean"), obs_up=("y", "mean"), n=("y", "size")).reset_index())


def block_bootstrap(values, blocks, n_boot=2000, block_len=1, seed=0, stat="mean"):
    """Circular moving-block bootstrap over ordered groups (e.g. sessions).

    `values` is per-row; rows are summed within their block group (a session),
    whole groups are resampled in runs of `block_len` consecutive groups, so
    dependence inside a session and across `block_len` sessions is kept.
    stat="mean": mean per row (sum / count). stat="sum_per_group": mean of the
    group sums (e.g. mean daily P&L). Returns (estimate, lo95, hi95).
    """
    v = pd.Series(np.asarray(values, dtype=float))
    g = pd.Series(np.asarray(blocks))
    agg = v.groupby(g.to_numpy(), sort=True).agg(["sum", "size"])
    sums, cnts = agg["sum"].to_numpy(), agg["size"].to_numpy()
    k = len(sums)
    if k == 0:
        return np.nan, np.nan, np.nan
    est = sums.sum() / cnts.sum() if stat == "mean" else sums.mean()
    block_len = max(1, min(block_len, k // 2))   # longer blocks than k/2 give no variation
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(k / block_len))
    starts = rng.integers(0, k, size=(n_boot, n_blocks))
    idx = ((starts[:, :, None] + np.arange(block_len)) % k).reshape(n_boot, -1)[:, :k]
    s, c = sums[idx].sum(axis=1), cnts[idx].sum(axis=1)
    boot = s / c if stat == "mean" else s / k
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(est), float(lo), float(hi)


def paired_difference(correct_model, correct_base, sessions, block_lens=(1, 5, 20), seed=0,
                      min_sessions=60):
    """Model-minus-baseline accuracy (pp) with block-bootstrap 95% intervals.

    Paired: both are scored on the same rows. The verdict is "inconclusive"
    when there are fewer than `min_sessions` independent sessions, or when
    the interval at the longest block length includes 0.
    """
    d = np.asarray(correct_model, float) - np.asarray(correct_base, float)
    n_sess = len(pd.unique(np.asarray(sessions)))
    out = {"n_sessions": n_sess}
    for L in block_lens:
        est, lo, hi = block_bootstrap(d, sessions, block_len=L, seed=seed)
        out["diff_pp"] = 100 * est
        out[f"ci_lo_pp_b{L}"], out[f"ci_hi_pp_b{L}"] = 100 * lo, 100 * hi
    L = block_lens[-1]
    if n_sess < min_sessions:
        out["verdict"] = "inconclusive (few sessions)"
    elif out[f"ci_lo_pp_b{L}"] > 0:
        out["verdict"] = "better than baseline"
    elif out[f"ci_hi_pp_b{L}"] < 0:
        out["verdict"] = "worse than baseline"
    else:
        out["verdict"] = "no detectable difference"
    return out
