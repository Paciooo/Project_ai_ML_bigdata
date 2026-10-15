"""
Run the full experiment:

1. Simulate one GBM tick path and aggregate it to hourly and daily candles.
2. Build features and the UP/DOWN target for each frequency.
3. Split chronologically 60/20/20, train the models on the training set.
4. Use the validation set for model selection, report the test set.
5. Save tables and figures to results/.

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


def run_frequency(name, df):
    """Train and evaluate all models for one data frequency."""
    X, y, drop_info = features.build_dataset(
        df, config.RETURN_LAGS, config.MOMENTUM_WINDOWS, config.VOLATILITY_WINDOWS)
    splits = features.chronological_split(X, y, config.TRAIN_FRAC, config.VAL_FRAC)

    # Number of observations and class distribution per split
    summary = {"frequency": name, **drop_info}
    for split, (_, ys) in splits.items():
        for k, v in evaluation.class_distribution(ys).items():
            summary[f"{split}_{k}"] = v
    print(f"  usable obs: {len(X):,} | train {summary['train_n']:,} / "
          f"val {summary['val_n']:,} / test {summary['test_n']:,} | "
          f"share UP: train {summary['train_share_up']:.3f}, test {summary['test_share_up']:.3f} | "
          f"unchanged prices dropped: {drop_info['n_unchanged_dropped']}")

    # Train on the training set only
    X_train, y_train = splits["train"]
    fitted = models.train_models(models.get_models(config.SEED), X_train, y_train)

    rows = []
    for model_name, model in fitted.items():
        val = evaluation.evaluate(model, *splits["val"])
        test = evaluation.evaluate(model, *splits["test"])
        rows.append({"frequency": name, "model": model_name,
                     "n_test": summary["test_n"],
                     "val_accuracy": val["accuracy"],
                     "val_balanced_accuracy": val["balanced_accuracy"],
                     **test})
    results = pd.DataFrame(rows)

    # Model selection uses the validation set only (never the test set)
    best = results.loc[results["val_balanced_accuracy"].idxmax(), "model"]
    results["selected_on_val"] = results["model"] == best
    print(f"  model selected on validation (balanced accuracy): {best}")

    rf = fitted["Random Forest"].named_steps["model"]
    importance = pd.Series(rf.feature_importances_, index=X.columns)
    return results, summary, importance


def main():
    out = config.RESULTS_DIR
    os.makedirs(out, exist_ok=True)

    # ---- 1. Synthetic data -------------------------------------------------
    print("Simulating GBM tick path ...")
    ticks = data.simulate_gbm_ticks(
        config.S0, config.MU, config.SIGMA, config.YEARS, config.SEED,
        config.DAYS_PER_YEAR, config.HOURS_PER_DAY, config.TICKS_PER_HOUR, config.START_DATE)
    datasets = data.build_datasets(ticks)

    print("Checking aggregation:")
    for msg in data.validate_aggregation(ticks, datasets, config.TICKS_PER_HOUR, config.HOURS_PER_DAY):
        print("  OK -", msg)
    m = data.gbm_moment_check(ticks, config.MU, config.SIGMA, config.DAYS_PER_YEAR,
                              config.HOURS_PER_DAY, config.TICKS_PER_HOUR)
    print(f"  tick log-return std {m['std_sim']:.3e} (theory {m['std_theory']:.3e}), "
          f"lag-1 autocorrelation {m['lag1_autocorr']:+.4f}")

    # ---- 2-4. Features, training, evaluation per frequency -----------------
    all_results, summaries, importances = [], [], {}
    for name, df in datasets.items():
        print(f"\n[{name}]")
        res, summ, imp = run_frequency(name, df)
        all_results.append(res)
        summaries.append(summ)
        importances[name] = imp

    results = pd.concat(all_results, ignore_index=True)
    summary = pd.DataFrame(summaries)

    # ---- 5. Save tables ----------------------------------------------------
    results.to_csv(os.path.join(out, "results_table.csv"), index=False, float_format="%.4f")
    summary.to_csv(os.path.join(out, "dataset_summary.csv"), index=False, float_format="%.4f")
    pd.DataFrame(importances).to_csv(os.path.join(out, "rf_feature_importance.csv"), float_format="%.4f")

    # ---- 5. Save figures ---------------------------------------------------
    plots.plot_price_path(ticks, datasets["daily"], out)
    plots.plot_candlesticks(datasets["daily"], datasets["hourly"], out)
    plots.plot_accuracy_bars(results, out)
    plots.plot_results_table(results, out)
    plots.plot_feature_importance(importances, out)
    plots.plot_confusion_matrices(results, out)

    cols = ["frequency", "model", "n_test", "accuracy", "acc_ci95", "balanced_accuracy", "f1", "selected_on_val"]
    print("\nOut-of-sample (test) results:")
    print(results[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nAll tables and figures saved to '{out}/'.")


if __name__ == "__main__":
    main()
