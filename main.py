"""
Run the full experiment on the data chosen in config.DATA_SOURCE:

"synthetic" (results/synthetic/)
  1. Simulate one GBM tick path and aggregate it to hourly and daily candles.
  2-4. Features, calendar split, training, validation and test (below).
  5. Repeat for several seeds (independent price paths) and summarise.

"nq" (results/nq/)
  1. Load real NQ futures 1-minute bars, remove bad prints, and build hourly
     and daily candles from the same minutes.
  2-4. Same pipeline as for the synthetic data.

Common steps:
  2. Build features and the UP/DOWN target for each frequency.
  3. Split by common calendar dates 60/20/20 (with a gap), train on train.
  4. Use the validation set for model selection, report the test set.

Usage (see README for all options):
  python main.py --smoke                              # offline fixture, ~1 minute
  python main.py --data-source nq --frequency 1min 1h session
  python main.py --data-source nq --legacy            # the original pipeline
  python main.py --data-source synthetic              # GBM null experiment (legacy)
Without --data-source the value of config.DATA_SOURCE is used. Every run
writes to its own folder results/<source>/runs/<run id>/.
"""

import argparse

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

    # Check that the candles really come from the tick path
    for msg in data.validate_aggregation(ticks, datasets, config.YEARS,
                                         config.DAYS_PER_YEAR, config.HOURS_PER_DAY):
        print("  OK -", msg)
    m = data.gbm_moment_check(ticks, config.MU, config.SIGMA)
    print(f"  standardised tick returns: mean {m['z_mean']:+.4f} (theory 0), "
          f"std {m['z_std']:.4f} (theory 1), lag-1 autocorrelation {m['lag1_autocorr']:+.4f}")

    results, summary, importances = run_frequencies(datasets, seed)
    return results, summary, importances, ticks, datasets


def run_frequencies(datasets, seed, split_days=None):
    """Calendar split on common dates, then train/evaluate every frequency.

    The split dates come from `split_days` (default: the daily candles).
    """
    days = split_days if split_days is not None else datasets["daily"].index
    val_start, test_start = features.calendar_boundaries(days, config.TRAIN_FRAC, config.VAL_FRAC)
    print(f"  split dates: train < {val_start.date()} <= validation < {test_start.date()} <= test")

    all_results, summaries, importances = [], [], {}
    for name, df in datasets.items():
        res, summ, imp = run_frequency(name, df, seed, val_start, test_start)
        all_results.append(res)
        summaries.append(summ)
        importances[name] = imp
    results = evaluation.add_baseline_comparison(pd.concat(all_results, ignore_index=True))
    return results, pd.DataFrame(summaries), importances


def save_detailed_outputs(out, results, summary, importances, prices, price_label, price_title,
                          datasets, subtitle, last_candles, daily=None):
    """Tables and figures 01-06 for one dataset."""
    daily = daily if daily is not None else datasets["daily"]
    results.to_csv(os.path.join(out, "results_table.csv"), index=False, float_format="%.4f")
    summary.to_csv(os.path.join(out, "dataset_summary.csv"), index=False, float_format="%.4f")
    pd.DataFrame(importances).to_csv(os.path.join(out, "rf_feature_importance.csv"), float_format="%.4f")
    plots.plot_price_path(prices, daily, out, price_label, price_title)
    plots.plot_candlesticks(daily, datasets["hourly"], out, last=last_candles)
    plots.plot_accuracy_bars(results, out, subtitle)
    plots.plot_results_table(results, out, subtitle)
    plots.plot_feature_importance(importances, out)
    plots.plot_confusion_matrices(results, out)


def print_results(results, title):
    cols = ["frequency", "model", "n_test", "accuracy", "acc_ci95", "acc_vs_baseline_pp",
            "balanced_accuracy", "f1", "selected_on_val"]
    print(f"\n{title}:")
    print(results[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def main_nq(out):
    """LEGACY pipeline. Real NQ futures: 1-minute bars plus hourly and daily candles.

    Kept for comparison with the saved results; known issues fixed in
    experiment.py: implicit resampling labels, future-dependent spike removal,
    flats dropped before evaluation.
    """
    y0, y1 = config.FUTURES_YEARS
    print(f"Loading {config.FUTURES_SYMBOL} 1-minute bars {y0}-{y1} from {config.futures_data_dir()} ...")
    minutes = data.load_futures_minutes(config.futures_data_dir(), config.FUTURES_SYMBOL, y0, y1)
    minutes, report = data.clean_minute_bars(minutes, config.SPIKE_THRESHOLD)
    print(f"  {report['bars_raw']:,} bars loaded; removed {report['saturday_bars_removed']} Saturday bars "
          f"and {report['spike_bars_removed']} isolated spikes -> {report['bars_clean']:,} bars")
    datasets = data.minute_datasets(minutes)
    print(f"  {minutes.index.min()} -> {minutes.index.max()}: "
          + ", ".join(f"{k} {len(v):,}" for k, v in datasets.items()))

    seed = config.SEEDS[0]  # only for the models' random_state: real data has one history
    results, summary, importances = run_frequencies(datasets, seed)
    save_detailed_outputs(out, results, summary, importances, datasets["hourly"]["close"],
                          f"Hourly close ({len(minutes):,} 1-minute bars in total)",
                          f"{config.FUTURES_SYMBOL} futures, continuous back-adjusted, {y0}-{y1}",
                          datasets, f"{config.FUTURES_SYMBOL} {y0}-{y1}", last_candles=True)
    print_results(results, f"Out-of-sample (test) results, {config.FUTURES_SYMBOL}")
    print(f"\nAll tables and figures saved to '{out}/'.")


def main_nq_ticks(out):
    """LEGACY pipeline. Real NQ trades (one month): every trade, 1-minute and hourly candles."""
    print(f"Loading tick files from {config.ticks_dir()} ...")
    ticks = data.load_tick_files(config.ticks_dir())
    datasets, daily = data.tick_datasets(ticks)
    print(f"  {len(ticks):,} trades, {ticks.index.min()} -> {ticks.index.max()}, {len(daily)} sessions: "
          + ", ".join(f"{k} {len(v):,}" for k, v in datasets.items()))

    seed = config.SEEDS[0]
    results, summary, importances = run_frequencies(datasets, seed, split_days=daily.index)
    save_detailed_outputs(out, results, summary, importances, datasets["1-min"]["close"],
                          f"1-minute close ({len(ticks):,} trades in total)",
                          f"{config.FUTURES_SYMBOL} trades (CME via IBKR), "
                          f"{daily.index.min():%Y-%m-%d} to {daily.index.max():%Y-%m-%d}",
                          datasets, f"{config.FUTURES_SYMBOL} trades, one month",
                          last_candles=False, daily=daily)
    print_results(results, f"Out-of-sample (test) results, {config.FUTURES_SYMBOL} trades")
    print(f"\nAll tables and figures saved to '{out}/'.")


def main_synthetic(out):
    """GBM null experiment over several seeds (legacy pipeline, kept as is)."""
    main_seed = config.SEEDS[0]

    all_results, all_summaries = [], []
    for seed in config.SEEDS:
        print(f"\n=== Seed {seed} ===")
        results, summary, importances, ticks, datasets = run_seed(seed)
        all_results.append(results)
        all_summaries.append(summary)

        if seed == main_seed:
            # Detailed tables and figures for one representative path
            save_detailed_outputs(out, results, summary, importances, ticks["price"],
                                  f"Tick (trade) prices: {len(ticks):,} ticks at random times",
                                  "Simulated GBM price path (one path, all frequencies derived from it)",
                                  datasets, f"seed {main_seed}", last_candles=False)
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

    print_results(main_results, f"Out-of-sample (test) results, seed {main_seed}")
    print(f"\nTest accuracy across {len(config.SEEDS)} seeds; improvement over the baseline in pp:")
    print(seed_stats[["frequency", "model", "accuracy_mean", "accuracy_std",
                      "acc_vs_baseline_pp_mean", "acc_vs_baseline_pp_std",
                      "n_seeds_beating_baseline", "balanced_accuracy_mean"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nAll tables and figures saved to '{out}/'.")


SOURCES = ["synthetic", "nq", "nq_ticks", "fixture"]
FREQS = ["tick", "1min", "5min", "15min", "1h", "4h", "session"]


def parse_args(argv=None):
    import experiment
    import features
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-source", choices=SOURCES, default=None,
                   help="default: config.DATA_SOURCE")
    p.add_argument("--legacy", action="store_true", help="run the original pipeline (nq, nq_ticks)")
    p.add_argument("--target", choices=[t for t in features.TARGETS if t != "next_trade_direction"],
                   default="next_close_direction")
    p.add_argument("--frequency", nargs="+", choices=FREQS, default=["1min", "5min", "15min", "1h", "4h", "session"])
    p.add_argument("--models", nargs="+", choices=experiment.MODELS, default=["logreg", "xgboost"])
    p.add_argument("--seed", type=int, default=config.SEEDS[0])
    p.add_argument("--scheme", choices=["holdout", "expanding", "rolling"], default="holdout")
    p.add_argument("--folds", type=int, default=4, help="test blocks for expanding/rolling")
    p.add_argument("--rolling-train-sessions", type=int, default=750)
    p.add_argument("--years", type=int, nargs=2, default=None, metavar=("FIRST", "LAST"))
    p.add_argument("--spike-policy", choices=["quarantine", "none", "legacy_remove"], default="quarantine")
    p.add_argument("--extra-features", nargs="*", default=[], choices=["volume", "range", "rv", "time"])
    p.add_argument("--context-features", action="store_true",
                   help="add completed-session returns to intraday rows (as-of join)")
    p.add_argument("--regimes", action="store_true", help="paper-inspired change-point regime models")
    p.add_argument("--regime-min-rows", type=int, default=2000)
    p.add_argument("--latency", default="1s", help="inference latency added to the decision time")
    p.add_argument("--max-train-rows", type=int, default=0, help="0 = all (most recent rows kept)")
    p.add_argument("--min-rows", type=int, default=30, help="minimum rows per split")
    p.add_argument("--confirmatory", action="store_true",
                   help="include the reserved forward holdout (use once, choices fixed in advance)")
    p.add_argument("--smoke", action="store_true",
                   help="quick offline run on the synthetic fixture; writes to results/smoke")
    p.add_argument("--fixture-signal", type=float, default=0.5, help="bounce size of the fixture (0 = null)")
    p.add_argument("--tag", default="")
    a = p.parse_args(argv)
    a.data_source = a.data_source or config.DATA_SOURCE
    if a.data_source not in SOURCES:  # never fall back silently
        p.error(f"config.DATA_SOURCE={a.data_source!r} is not one of {SOURCES}")
    if a.smoke:
        a.data_source, a.smoke_fixture = "fixture", True
        a.max_train_rows = a.max_train_rows or 20000
        if "--frequency" not in (argv or __import__("sys").argv):
            a.frequency = ["1min", "5min", "1h"]
    else:
        a.smoke_fixture = False
    if "tick" in a.frequency and a.data_source != "nq_ticks":
        p.error("--frequency tick needs --data-source nq_ticks")
    if a.max_train_rows < 0 or a.folds < 1 or a.min_rows < 1:
        p.error("--max-train-rows >= 0, --folds >= 1 and --min-rows >= 1 required")
    return a


def main(argv=None):
    args = parse_args(argv)
    if args.data_source == "synthetic" or args.legacy:
        import runlog
        if args.data_source == "fixture":
            raise SystemExit("--legacy does not support the fixture")
        out = runlog.new_run_dir(config.RESULTS_DIR, args.data_source, tag="legacy")
        runlog.write_run_json(out, {"args": vars(args), "config": runlog.config_snapshot(config),
                                    "pipeline": "legacy"})
        return {"synthetic": main_synthetic, "nq": main_nq, "nq_ticks": main_nq_ticks}[args.data_source](str(out))
    import experiment
    return experiment.run(args)


if __name__ == "__main__":
    main()
