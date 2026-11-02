"""
Change-point regimes, an ADAPTATION of Oh & Kim (2002), "Analyzing stock
market tick data using piecewise nonlinear model", Expert Systems with
Applications 22, 249-255.

The paper (KOSPI 200, 1990-2000): five lagged inputs (embedding dimension 6
from a correlation-dimension analysis); Pettitt test + binary segmentation of
the DAILY index into homogeneous groups (3 change-points -> 4 groups); a
backpropagation network that predicts the group of day t+1 from day-t
inputs; one 1-minute network per group; evaluated on 10 random test days
by forecast error and a 1-minute trading rule.

Here, differently (and so not a replication):
  * the target is NQ next-bar DIRECTION, not the price level;
  * segmentation runs on each training fold only, on daily absolute log
    returns ("vol", default: a stationary series) or on log closes
    ("level", as in the paper);
  * the group of session s is predicted from sessions completed before s
    (never from s's own close); test history is never segmented;
  * the number of change-points and the minimum segment length are chosen on
    validation; the five lags are compared with three on validation; no
    chaos analysis (five lags are a candidate, not an assumption);
  * groups with too few training rows fall back to the pooled model.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def pettitt(x):
    """Pettitt (1979) change-point test. Returns (index of last point before the
    change, K statistic, approximate p-value)."""
    x = np.asarray(x, float)
    T = len(x)
    if T < 3:
        return None, 0.0, 1.0
    r = rankdata(x)
    t = np.arange(1, T)
    U = 2 * np.cumsum(r)[:-1] - t * (T + 1)       # U_{t,T} via ranks
    k = int(np.argmax(np.abs(U)))
    K = float(np.abs(U[k]))
    p = min(1.0, 2 * np.exp(-6 * K ** 2 / (T ** 3 + T ** 2)))
    return k, K, p


def binary_segmentation(x, max_change_points=3, min_size=20, alpha=0.05):
    """Vostrikova-style binary segmentation with the Pettitt test.

    Splits greedily at the most significant change among current segments
    until `max_change_points` are found or no segment has a significant
    change with both sides >= min_size. Returns sorted segment start indices
    (the first is 0).
    """
    starts = [0]
    segs = [(0, len(x))]
    while len(starts) - 1 < max_change_points:
        best = None
        for a, b in segs:
            if b - a < 2 * min_size:
                continue
            k, _, p = pettitt(x[a:b])
            cut = a + k + 1
            if p < alpha and cut - a >= min_size and b - cut >= min_size and (best is None or p < best[0]):
                best = (p, a, b, cut)
        if best is None:
            break
        _, a, b, cut = best
        segs.remove((a, b))
        segs += [(a, cut), (cut, b)]
        starts.append(cut)
    return sorted(starts)


def session_features(sessions):
    """Causal features at each session's close: five lagged daily log returns
    and 5/20-session realized volatility (completed sessions only)."""
    r = np.log(sessions["close"]).diff()
    f = pd.DataFrame(index=sessions.index)
    for k in range(1, 6):
        f[f"d_ret{k}"] = r.shift(k - 1)
    f["d_rv5"] = r.rolling(5).std()
    f["d_rv20"] = r.rolling(20).std()
    return f


class RegimeRouter:
    """Fit on training sessions; predicts each later session's group causally."""

    def __init__(self, n_change_points=3, min_size=20, series="vol", seed=0):
        self.n_cp, self.min_size, self.series, self.seed = n_change_points, min_size, series, seed

    def fit(self, sessions_train):
        s = sessions_train.sort_values("session_id").reset_index(drop=True)
        r = np.log(s["close"]).diff()
        x = (r.abs() if self.series == "vol" else np.log(s["close"])).fillna(0).to_numpy()
        starts = binary_segmentation(x, self.n_cp, self.min_size)
        group = np.zeros(len(s), int)
        for g, a in enumerate(starts):
            group[a:] = g
        self.train_group = pd.Series(group, index=pd.DatetimeIndex(s["session_id"]))
        self.n_groups = len(starts)
        self.change_sessions = [s["session_id"].iloc[a] for a in starts[1:]]
        f = session_features(s)
        X, y = f.iloc[:-1], group[1:]                 # features at s-1 -> group of s
        ok = X.notna().all(axis=1).to_numpy()
        self.clf = None
        if self.n_groups > 1 and len(np.unique(y[ok])) > 1:
            self.clf = make_pipeline(StandardScaler(), MLPClassifier(
                hidden_layer_sizes=(8,), max_iter=500, random_state=self.seed, alpha=1e-2))
            self.clf.fit(X[ok], y[ok])
        return self

    def predict_sessions(self, sessions_all):
        """Group predicted for each session from the previous completed session
        (-1 where unknown: first session, missing features, or one group)."""
        s = sessions_all.sort_values("session_id").reset_index(drop=True)
        f = session_features(s)
        pred = np.full(len(s), -1)
        if self.clf is not None:
            ok = f.notna().all(axis=1).to_numpy()
            p = np.full(len(s), -1)
            p[ok] = self.clf.predict(f[ok])
            pred[1:] = p[:-1]
        elif self.n_groups == 1:
            pred[1:] = 0
        return pd.Series(pred, index=pd.DatetimeIndex(s["session_id"]))


class RegimeModel:
    """One model per group, pooled fallback for sparse or unknown groups."""

    def __init__(self, make_model, min_rows=2000):
        self.make_model, self.min_rows = make_model, min_rows

    def fit(self, X, y, groups):
        self.pooled = self.make_model().fit(X, y)
        self.per = {}
        for g in np.unique(groups):
            m = groups == g
            if g >= 0 and m.sum() >= self.min_rows and len(np.unique(y[m])) == 2:
                self.per[g] = self.make_model().fit(X[m], y[m])
        return self

    def predict_proba_up(self, X, groups):
        p = self.pooled.predict_proba(X)[:, 1]
        for g, m in self.per.items():
            sel = groups == g
            if sel.any():
                p[sel] = m.predict_proba(X[sel])[:, 1]
        return p
