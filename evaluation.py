"""
Evaluation metrics: accuracy, balanced accuracy, F1 and confusion matrix.
"""

import numpy as np
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
