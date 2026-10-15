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
        # 95% half-width of a binomial CI: how much accuracy moves by chance alone
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
