"""
Figures saved to the results directory.
"""

import os

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np


def plot_price_path(ticks, daily, out_dir):
    """Full simulated GBM path (ticks) with the daily closes on top."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(ticks.index, ticks["price"], lw=0.3, color="tab:blue",
            label=f"Tick (trade) prices: {len(ticks):,} ticks at random times")
    ax.plot(daily.index, daily["close"], lw=1.2, color="tab:orange", label="Daily close")
    ax.set_title("Simulated GBM price path (one path, all frequencies derived from it)")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "01_gbm_price_path.png"), dpi=150)
    plt.close(fig)


def _candles(ax, ohlc, title):
    """Minimal candlestick chart drawn with matplotlib only."""
    x = np.arange(len(ohlc))
    up = ohlc["close"] >= ohlc["open"]
    colors = np.where(up, "tab:green", "tab:red")
    ax.vlines(x, ohlc["low"], ohlc["high"], color=colors, lw=0.8)
    ax.bar(x, (ohlc["close"] - ohlc["open"]).abs(), bottom=ohlc[["open", "close"]].min(axis=1),
           color=colors, width=0.7)
    step = max(len(ohlc) // 8, 1)
    ax.set_xticks(x[::step])
    fmt = "%Y-%m-%d" if "daily" in title.lower() else "%m-%d %H:%M"
    ax.set_xticklabels(ohlc.index[::step].strftime(fmt), rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel("Price")
    ax.grid(alpha=0.3)


def plot_candlesticks(daily, hourly, out_dir, n_days=60, n_hours=72):
    """Sample candlestick charts for daily and hourly data."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    _candles(axes[0], daily.iloc[:n_days], f"Daily candles (first {n_days} days)")
    _candles(axes[1], hourly.iloc[:n_hours], f"Hourly candles (first {n_hours} hours)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "02_candlesticks.png"), dpi=150)
    plt.close(fig)


def plot_accuracy_bars(results, out_dir, seed):
    """Grouped bar chart: test accuracy per model and frequency."""
    freqs = list(results["frequency"].unique())
    models = list(results["model"].unique())
    width = 0.8 / len(models)
    x = np.arange(len(freqs))

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, m in enumerate(models):
        sub = results[results["model"] == m].set_index("frequency").loc[freqs]
        ax.bar(x + i * width - 0.4 + width / 2, sub["accuracy"], width,
               yerr=sub["acc_ci95"], capsize=3, label=m)
    ax.axhline(0.5, color="black", ls="--", lw=1, label="Coin flip (50%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f.capitalize() for f in freqs])
    ax.set_ylim(0.35, 0.65)
    ax.set_ylabel("Out-of-sample accuracy (test set)")
    ax.set_title(f"Test accuracy by model and frequency, seed {seed} "
                 f"(error bars: approx. 95% binomial CI)")
    ax.legend(fontsize=8, ncol=4, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "03_accuracy_comparison.png"), dpi=150)
    plt.close(fig)


def plot_results_table(results, out_dir, seed):
    """Render the results table as an image (handy for slides)."""
    cols = ["frequency", "model", "n_test", "accuracy", "acc_vs_baseline_pp", "balanced_accuracy", "f1"]
    tbl = results[cols].copy()
    for c in ["accuracy", "balanced_accuracy", "f1"]:
        tbl[c] = tbl[c].map("{:.3f}".format)
    tbl["acc_vs_baseline_pp"] = tbl["acc_vs_baseline_pp"].map("{:+.1f} pp".format)
    tbl.loc[tbl["model"] == "Majority baseline", "acc_vs_baseline_pp"] = "–"
    tbl["n_test"] = tbl["n_test"].map("{:,}".format)

    fig, ax = plt.subplots(figsize=(11, 0.3 * len(tbl) + 0.6))
    ax.axis("off")
    t = ax.table(cellText=tbl.values,
                 colLabels=["Frequency", "Model", "Test obs.", "Accuracy", "vs baseline",
                            "Balanced acc.", "F1"],
                 loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.3)
    ax.set_title(f"Out-of-sample (test set) results, seed {seed}", pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "04_results_table.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(importances, out_dir):
    """Random Forest feature importance, one panel per frequency."""
    fig, axes = plt.subplots(1, len(importances), figsize=(13, 4.5), sharex=True)
    for ax, (freq, imp) in zip(np.atleast_1d(axes), importances.items()):
        imp = imp.sort_values()
        ax.barh(imp.index, imp.values, color="tab:green")
        ax.axvline(1 / len(imp), color="black", ls="--", lw=1)  # equal-importance line
        ax.set_title(f"{freq.capitalize()}")
        ax.set_xlabel("Impurity-based importance")
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Random Forest feature importance (dashed line = equal importance)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "05_rf_feature_importance.png"), dpi=150)
    plt.close(fig)


def plot_confusion_matrices(results, out_dir):
    """Grid of test-set confusion matrices (rows: frequency, cols: model)."""
    freqs = list(results["frequency"].unique())
    models = list(results["model"].unique())
    fig, axes = plt.subplots(len(freqs), len(models),
                             figsize=(2.6 * len(models), 2.6 * len(freqs)))
    for i, f in enumerate(freqs):
        for j, m in enumerate(models):
            r = results[(results["frequency"] == f) & (results["model"] == m)].iloc[0]
            cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
            ax = axes[i, j]
            ax.imshow(cm / cm.sum(), cmap="Blues", vmin=0, vmax=0.6)  # same scale everywhere
            for a in range(2):
                for b in range(2):
                    share = cm[a, b] / cm.sum()
                    color = "white" if share > 0.35 else "black"
                    ax.text(b, a, f"{cm[a, b]:,}\n({share:.0%})", ha="center", va="center",
                            fontsize=8, color=color)
            ax.set_xticks([0, 1], ["Pred DOWN", "Pred UP"], fontsize=7)
            ax.set_yticks([0, 1], ["True DOWN", "True UP"], fontsize=7)
            if i == 0:
                ax.set_title(m, fontsize=9)
            if j == 0:
                ax.set_ylabel(f.capitalize(), fontsize=10)
    fig.suptitle("Test-set confusion matrices")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "06_confusion_matrices.png"), dpi=150)
    plt.close(fig)


def plot_multi_seed(results_all, focus_models, out_dir):
    """Test accuracy of the focus models across seeds.

    Each dot is one seed (one independent price path); the bar is the mean.
    The grey band is the spread of the majority baseline across seeds.
    """
    freqs = list(results_all["frequency"].unique())
    width = 0.8 / len(focus_models)
    x = np.arange(len(freqs))
    fig, ax = plt.subplots(figsize=(11, 5))

    for k, f in enumerate(freqs):
        base = results_all[(results_all["frequency"] == f) &
                           (results_all["model"] == "Majority baseline")]["accuracy"]
        ax.fill_between([k - 0.45, k + 0.45], base.min(), base.max(), color="grey", alpha=0.25,
                        label="Majority baseline (range over seeds)" if k == 0 else None)

    for i, m in enumerate(focus_models):
        pos = x + i * width - 0.4 + width / 2
        sub = results_all[results_all["model"] == m]
        means = [sub[sub["frequency"] == f]["accuracy"].mean() for f in freqs]
        stds = [sub[sub["frequency"] == f]["accuracy"].std() for f in freqs]
        ax.bar(pos, means, width, yerr=stds, capsize=3, alpha=0.8, label=f"{m} (mean ± std)")
        for p, f in zip(pos, freqs):
            vals = sub[sub["frequency"] == f]["accuracy"]
            ax.scatter(np.full(len(vals), p), vals, color="black", s=8, zorder=3)

    ax.axhline(0.5, color="black", ls="--", lw=1)
    n_seeds = results_all["seed"].nunique()
    ax.set_xticks(x)
    ax.set_xticklabels([f.capitalize() for f in freqs])
    ax.set_ylim(0.35, 0.65)
    ax.set_ylabel("Out-of-sample accuracy (test set)")
    ax.set_title(f"Test accuracy over {n_seeds} independent GBM paths (dots = individual seeds)")
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "07_multi_seed_accuracy.png"), dpi=150)
    plt.close(fig)
