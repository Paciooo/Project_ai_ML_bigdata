"""
Run the full experiment:

1. Simulate one GBM tick path and aggregate it to hourly and daily candles.
2. Build features and the UP/DOWN target for each frequency.
3. Split by common calendar dates 60/20/20 (with a gap), train on train.
4. Use the validation set for model selection, report the test set.
5. Repeat for several seeds (independent price paths) and summarise.
6. Save tables and figures to results/.

Usage:  python main.py
"""

import os

import pandas as pd

import config
import data
import evaluation
import features
import models
import plots


def run_frequency(name, df, seed, val_start, test_start):
    """Train and evaluate all models for one data frequency."""
    X, y, drop_info = features.build_dataset(
        df, config.RETURN_LAGS, config.MOMENTUM_WINDOWS, config.VOLATILITY_WINDOWS)
    splits = features.calendar_split(X, y, val_start, test_start, config.SPLIT_GAP)

    # Number of observations and class distribution per split
    summary = {"seed": seed, "frequency": name, **drop_info,
               "n_gap_dropped": 2 * config.SPLIT_GAP}
    for split, (Xs, ys) in splits.items():
        for k, v in evaluation.class_distribution(ys).items():
            summary[f"{split}_{k}"] = v
        summary[f"{split}_first"], summary[f"{split}_last"] = Xs.index[0], Xs.index[-1]
    print(f"  [{name}] usable obs: {len(X):,} | train {summary['train_n']:,} / "
          f"val {summary['val_n']:,} / test {summary['test_n']:,} | "
          f"share UP: train {summary['train_share_up']:.3f}, test {summary['test_share_up']:.3f} | "
          f"unchanged prices dropped: {drop_info['n_unchanged_dropped']}")

    # Train on the training set only
    X_train, y_train = splits["train"]
    fitted = models.train_models(models.get_models(seed), X_train, y_train)

    rows = []
    for model_name, model in fitted.items():
        val = evaluation.evaluate(model, *splits["val"])
        test = evaluation.evaluate(model, *splits["test"])
        rows.append({"seed": seed, "frequency": name, "model": model_name,
                     "n_test": summary["test_n"],
                     "val_accuracy": val["accuracy"],
                     "val_balanced_accuracy": val["balanced_accuracy"],
                     **test})
    results = pd.DataFrame(rows)

    # Model selection uses the validation set only (never the test set)
    best = results.loc[results["val_balanced_accuracy"].idxmax(), "model"]
    results["selected_on_val"] = results["model"] == best
    print(f"  [{name}] model selected on validation (balanced accuracy): {best}")

    rf = fitted["Random Forest"].named_steps["model"]
    importance = pd.Series(rf.feature_importances_, index=X.columns)
    return results, summary, importance


def run_seed(seed):
    """Simulate one price path and run every frequency on it."""
    ticks = data.simulate_gbm_ticks(
        config.S0, config.MU, config.SIGMA, config.YEARS, seed, config.TICKS_PER_HOUR,
        config.DAYS_PER_YEAR, config.HOURS_PER_DAY, config.START_DATE)
    datasets = data.build_datasets(ticks)

    # Same calendar split dates for every frequency
    val_start, test_start = features.calendar_boundaries(
        datasets["daily"].index, config.TRAIN_FRAC, config.VAL_FRAC)
    print(f"  split dates: train < {val_start.date()} <= validation < {test_start.date()} <= test")

    # Check that the candles really come from the tick path
    for msg in data.validate_aggregation(ticks, datasets, config.YEARS,
                                         config.DAYS_PER_YEAR, config.HOURS_PER_DAY):
        print("  OK -", msg)
    m = data.gbm_moment_check(ticks, config.MU, config.SIGMA)
    print(f"  standardised tick returns: mean {m['z_mean']:+.4f} (theory 0), "
          f"std {m['z_std']:.4f} (theory 1), lag-1 autocorrelation {m['lag1_autocorr']:+.4f}")

    all_results, summaries, importances = [], [], {}
    for name, df in datasets.items():
        res, summ, imp = run_frequency(name, df, seed, val_start, test_start)
        all_results.append(res)
        summaries.append(summ)
        importances[name] = imp
    results = evaluation.add_baseline_comparison(pd.concat(all_results, ignore_index=True))
    return results, pd.DataFrame(summaries), importances, ticks, datasets


def main():
    out = config.RESULTS_DIR
    os.makedirs(out, exist_ok=True)
    main_seed = config.SEEDS[0]

    all_results, all_summaries = [], []
    for seed in config.SEEDS:
        print(f"\n=== Seed {seed} ===")
        results, summary, importances, ticks, datasets = run_seed(seed)
        all_results.append(results)
        all_summaries.append(summary)

        if seed == main_seed:
            # Detailed tables and figures for one representative path
            results.to_csv(os.path.join(out, "results_table.csv"), index=False, float_format="%.4f")
            pd.DataFrame(importances).to_csv(os.path.join(out, "rf_feature_importance.csv"),
                                             float_format="%.4f")
            plots.plot_price_path(ticks, datasets["daily"], out)
            plots.plot_candlesticks(datasets["daily"], datasets["hourly"], out)
            plots.plot_accuracy_bars(results, out, main_seed)
            plots.plot_results_table(results, out, main_seed)
            plots.plot_feature_importance(importances, out)
            plots.plot_confusion_matrices(results, out)
            main_results = results
        del ticks, datasets  # free memory before the next path

    # ---- Multi-seed summary ------------------------------------------------
    results_all = pd.concat(all_results, ignore_index=True)
    summary_all = pd.concat(all_summaries, ignore_index=True)
    results_all.to_csv(os.path.join(out, "results_all_seeds.csv"), index=False, float_format="%.4f")
    summary_all.to_csv(os.path.join(out, "dataset_summary.csv"), index=False, float_format="%.4f")

    seed_stats = (results_all.groupby(["frequency", "model"], sort=False)
                  .agg(n_seeds=("seed", "nunique"),
                       accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
                       accuracy_min=("accuracy", "min"), accuracy_max=("accuracy", "max"),
                       balanced_accuracy_mean=("balanced_accuracy", "mean"),
                       balanced_accuracy_std=("balanced_accuracy", "std"),
                       f1_mean=("f1", "mean"),
                       acc_vs_baseline_pp_mean=("acc_vs_baseline_pp", "mean"),
                       acc_vs_baseline_pp_std=("acc_vs_baseline_pp", "std"),
                       n_seeds_beating_baseline=("acc_vs_baseline_pp", lambda v: int((v > 0).sum())))
                  .reset_index())
    seed_stats.to_csv(os.path.join(out, "multi_seed_summary.csv"), index=False, float_format="%.4f")
    plots.plot_multi_seed(results_all, config.FOCUS_MODELS, out)

    cols = ["frequency", "model", "n_test", "accuracy", "acc_ci95", "acc_vs_baseline_pp",
            "balanced_accuracy", "f1", "selected_on_val"]
    print(f"\nOut-of-sample (test) results, seed {main_seed}:")
    print(main_results[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nTest accuracy across {len(config.SEEDS)} seeds; improvement over the baseline in pp:")
    print(seed_stats[["frequency", "model", "accuracy_mean", "accuracy_std",
                      "acc_vs_baseline_pp_mean", "acc_vs_baseline_pp_std",
                      "n_seeds_beating_baseline", "balanced_accuracy_mean"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nAll tables and figures saved to '{out}/'.")


if __name__ == "__main__":
    main()
