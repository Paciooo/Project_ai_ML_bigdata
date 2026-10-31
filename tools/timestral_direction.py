"""
Zero-shot UP/DOWN prediction with Mistral's Timestral on the NQ test set.

Timestral is a pretrained time-series forecasting model (no training on our
data). For each test bar it receives the last --context closes, forecasts the
next close, and we predict UP if the forecast median is above the current
close, DOWN otherwise. The result is scored exactly like the other models:
same data, same cleaning, same calendar split, same test period, same
majority baseline (taken from the training period).

Forecasting every 1-minute test bar (1.3 million) would take hours on a CPU,
so a random sample of --n-samples test bars is scored per frequency (all of
them when there are fewer). The baseline is scored on the same sample.

Setup (once):
  pip install git+https://github.com/mistralai/timestral.git   # needs torch >= 2.4
  The first run downloads the weights from Hugging Face (5M: 21 MB, 209M: 837 MB).

Usage (from the project folder, with DATA_SOURCE = "nq" data available):
  python tools/timestral_direction.py                              # Timestral-5M, all frequencies
  python tools/timestral_direction.py --model mistralai/Timestral-209M --n-samples 500
  python tools/timestral_direction.py --frequency daily

Output: results/nq/timestral_results.csv
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project modules
import config  # noqa: E402
import data  # noqa: E402
import evaluation  # noqa: E402
import features  # noqa: E402


def load_nq_datasets():
    """Same data and cleaning as main.py (DATA_SOURCE = "nq")."""
    y0, y1 = config.FUTURES_YEARS
    minutes = data.load_futures_minutes(config.FUTURES_DATA_DIR, config.FUTURES_SYMBOL, y0, y1)
    minutes, _ = data.clean_minute_bars(minutes, config.SPIKE_THRESHOLD)
    return data.minute_datasets(minutes)


def score_frequency(forecaster, name, df, val_start, test_start, args, rng):
    """Timestral vs majority baseline on a sample of the test bars of one frequency."""
    X, y, _ = features.build_dataset(df, config.RETURN_LAGS, config.MOMENTUM_WINDOWS,
                                     config.VOLATILITY_WINDOWS)
    splits = features.calendar_split(X, y, val_start, test_start, config.SPLIT_GAP)
    y_train, y_test = splits["train"][1], splits["test"][1]
    majority = int(y_train.mean() >= 0.5)

    # Test bars with a full context window behind them
    close = df["close"].to_numpy(dtype=np.float64)
    pos = df.index.get_indexer(y_test.index)
    ok = pos >= args.context - 1
    y_test, pos = y_test[ok], pos[ok]
    if len(y_test) > args.n_samples:
        pick = np.sort(rng.choice(len(y_test), size=args.n_samples, replace=False))
        y_test, pos = y_test.iloc[pick], pos[pick]

    preds, ties, t0 = [], 0, time.time()
    for i in range(0, len(pos), args.batch):
        p = pos[i:i + args.batch]
        ctx = np.stack([close[j - args.context + 1:j + 1] for j in p])   # (batch, context)
        median = forecaster.forecast(ctx, prediction_length=1).median[:, 0, 0].cpu().numpy()
        last = ctx[:, -1]
        preds.append((median > last).astype(int))
        ties += int((median == last).sum())
        print(f"  [{name}] {min(i + args.batch, len(pos)):,}/{len(pos):,} forecasts", end="\r")
    print()
    pred = np.concatenate(preds)

    class Fixed:  # lets evaluation.evaluate score fixed predictions
        def __init__(self, values):
            self.values = values

        def predict(self, _):
            return self.values

    tim = evaluation.evaluate(Fixed(pred), None, y_test.to_numpy())
    base = evaluation.evaluate(Fixed(np.full(len(y_test), majority)), None, y_test.to_numpy())
    return {
        "frequency": name, "model": args.model, "context": args.context, "n_scored": len(y_test),
        "accuracy": tim["accuracy"], "acc_ci95": tim["acc_ci95"],
        "baseline_accuracy": base["accuracy"],
        "acc_vs_baseline_pp": 100 * (tim["accuracy"] - base["accuracy"]),
        "balanced_accuracy": tim["balanced_accuracy"], "f1": tim["f1"],
        "pred_up_share": tim["pred_up_share"], "median_equal_last": ties / len(y_test),
        "seconds": round(time.time() - t0, 1),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="mistralai/Timestral-5M",
                   help="Hugging Face repo: Timestral-5M, -28M, -123M or -209M")
    p.add_argument("--frequency", choices=["1-min", "hourly", "daily", "all"], default="all")
    p.add_argument("--context", type=int, default=512, help="past closes given to the model")
    p.add_argument("--n-samples", type=int, default=2000, help="test bars scored per frequency")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--device", default="cpu", help='"cuda" if you have an NVIDIA GPU')
    args = p.parse_args()

    from timestral import TimestralForecaster  # imported here so --help works without it

    print(f"Loading {config.FUTURES_SYMBOL} data ...")
    datasets = load_nq_datasets()
    val_start, test_start = features.calendar_boundaries(
        datasets["daily"].index, config.TRAIN_FRAC, config.VAL_FRAC)
    print(f"Test period starts {test_start.date()}. Loading {args.model} ...")
    forecaster = TimestralForecaster.from_pretrained(args.model, device=args.device)

    rng = np.random.default_rng(config.SEEDS[0])
    names = list(datasets) if args.frequency == "all" else [args.frequency]
    rows = [score_frequency(forecaster, n, datasets[n], val_start, test_start, args, rng) for n in names]
    res = pd.DataFrame(rows)

    out = Path(config.RESULTS_DIR) / "nq"
    out.mkdir(parents=True, exist_ok=True)
    f = out / "timestral_results.csv"
    res.to_csv(f, mode="a", header=not f.exists(), index=False, float_format="%.4f")
    print(res[["frequency", "n_scored", "accuracy", "acc_ci95", "baseline_accuracy", "acc_vs_baseline_pp",
               "balanced_accuracy", "pred_up_share"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nAppended to {os.path.normpath(f)}")


if __name__ == "__main__":
    main()
