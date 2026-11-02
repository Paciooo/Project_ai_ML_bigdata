# Run guide and implementation status (audit of 2026-11-02)

This file is meant to be enough for another session to carry on without the
earlier conversation. Run every command from the project folder in PowerShell.
The tested environment is pinned in `requirements-lock.txt`. All defaults
are in `config.py`, and every run saves its full configuration to `run.json`.

## 1. What is complete

Correctness fixes (all under regression test: `python -m pytest tests -q`, 50 tests):

| Area | Done | Where |
|---|---|---|
| Target naming | `next_close_direction` (default, the old target); `next_candle_body_direction` (optional); `next_trade_direction` (raw ticks only) | `features.py` |
| Flat outcomes | Kept, never dropped before the shift. Two metrics: `acc_nonflat` (legacy, conditional on a move, with `coverage_nonflat`) and `acc_all` (flat counts as a miss). The P&L uses every outcome | `features.py`, `evaluation.py` |
| Timing | Every row records `bar_start`, `bar_end`, `available_at`, `decision_at` (+ `--latency`, default 1 s), `label_start`/`label_end` and `session_id`. Tests check the order | `features.build_timed_dataset` |
| Sources | TradeStation: Europe/Rome, end-stamped. IB: UTC event time, left-closed minutes. DST-invalid stamps dropped and counted | `timeframes.py` |
| Sessions | CME, America/Chicago, open 17:00 CT. The close and the halt changed over time (inferred from the data: 16:15 close until 2015-09-18, halt until 2021-06-25). Bars outside the schedule are dropped and counted | `timeframes.SCHEDULE` |
| Time frames | 1min, 5min, 15min, 1h, 4h, session, all anchored to the session. Each bar is available at its scheduled end. The partial last 4h bin and early-close bins are flagged and never used as a row or a label. A label is used only if the next bar is the next scheduled interval in the same session (no outages, gaps or overnight) | `timeframes.aggregate`, `next_label_eligible` |
| Bad prints | Causal quarantine instead of the old future-dependent removal: \|1-min return\| > 2% flags the bar (prefix-invariant). Rows whose feature window or label touches a flagged bar are excluded and counted. The legacy rule is kept only for sensitivity (`--spike-policy legacy_remove`) | `quality.py` |
| Validation | Bars: finite and positive prices, OHLC consistency, volume ≥ 0, unique and increasing timestamps, year coverage, missing ranges. Ticks: equal timestamps and identical records are allowed, nothing is deduplicated | `quality.py` |
| Splits | Whole sessions per split. Labels crossing a boundary are purged. Fractions are validated, a too-small split raises an error, and the legacy `calendar_split` rejects gap ≥ split size | `features.purged_split` |
| Walk-forward | `--scheme expanding` or `rolling`, `--folds N`. Models, thresholds and regime settings are refitted on each fold | `features.walk_forward_folds` |
| Baselines and metrics | Majority, persistence and reversal (a zero last return falls back to the majority). Accuracy, balanced accuracy, macro F1, confusion matrix, Brier score, log loss and calibration table. Paired block bootstrap of model minus majority over 1/5/20-session blocks; the verdict is "inconclusive" below 60 sessions. Results by year | `evaluation.py` |
| Backtest | Offline only. Entry at the next bar's open after the decision (never the signal close), exit at the next close, one contract, $20/pt, 0.25 tick. Cost scenarios `zero_cost`/`low`/`base`/`high` (assumptions). No-trade threshold chosen on validation. Flat outcomes are traded at full cost. Approximate roll days are excluded at session frequency. Reports gross/net, turnover, drawdown, daily-P&L bootstrap and a no-trade row | `backtest.py` |
| Reproducibility | CLI (`main.py --help`). An unknown source is an error, never a silent fallback. Every run gets a unique folder `results/<source>/runs/<UTC>_<id>_<tag>/` with `run.json` (config, git commit + dirty flag, package versions, source hashes, conventions, folds), per-origin predictions, fitted models (`models/*.joblib`) and `trial_log.csv` (every run marked exploratory) | `runlog.py`, `experiment.py` |
| Holdout protocol | Sessions from `CONFIRMATORY_HOLDOUT_START` (2025-06-02) are excluded unless `--confirmatory` is passed. All data up to 2025-05-30 is development evidence | `config.py` |
| IBKR ticks | Typed results and errors, matched by request ID. A repeated 10187 makes the day **incomplete**, never "no data". Whole-second pagination: no loss or duplication, and 1000+ trades in one second are kept. Manifest per day, coverage check against the 1-min bars before a day is final, atomic writes, resume from checkpoint. Old files without a manifest are reported UNVERIFIED and left untouched | `tools/download_ticks_ibkr.py`, `tools/ib_common.py` |
| IBKR bars | `reqHistoricalData` per day, manifest. CONTFUT uses an empty endDateTime. Daily closes are flagged as possible settlement prices | `tools/download_bars_ibkr.py` |
| check_ticks | Reads the manifest (verified vs UNVERIFIED); gap check is schedule-aware | `tools/check_ticks.py` |
| Timestral | `HF_HUB_DISABLE_XET=1` is set before the Hugging Face import (your own value wins). Common origin manifest per frequency (the sample does not depend on `--frequency`). Arguments validated, paired comparison vs rules and `--baseline-run` models, time, device and revision saved | `tools/timestral_direction.py` |
| Paper adaptation | Pettitt test + binary segmentation fitted on each training fold only. A small MLP routes each session from the previous completed session. Per-group models with pooled fallback. Change-points (1/3), minimum segment length (20/60) and lags (3/5) chosen on validation. Labelled as an adaptation of Oh & Kim (2002), not a replication | `regimes.py`, `--regimes` |
| Extras | `--extra-features volume range rv time` (ablation); `--context-features` (completed-session returns via an as-of join, tested so that an unfinished session is never used) | `features.py` |

## 2. What remains unverified

* **Real IBKR behaviour:** tested only against a mock (`tests/fake_ib.py`).
  Not yet checked live:
  * IB really completes the last second of a tick page;
  * the error codes and the request-ID matching;
  * 1-minute bar volume agrees with tick volume within 2%.

  The first real day will tell. If coverage fails on every day, look at
  `volume_ratio` in `manifest.json` before changing `--volume-tolerance`.
* **Your 25 tick files (NQZ6):** downloaded with the old script, which had no
  manifest and could drop trades inside seconds with more than 1000 trades.
  They are UNVERIFIED. `--recheck-unverified` downloads them again and
  replaces a file only if coverage passes.
* **Real Timestral weights:** Hugging Face is blocked in the cloud session.
  Only the interface was tested (the installed `timestral` 1.0.0 with a tiny
  random checkpoint). Real-weight runs must be done on your PC.
* **Schedule dates and roll dates:** inferred from the data or approximated.
  Roll dates are not in the data, so the continuous series is
  back-adjusted and its prices are not tradable contract prices.
* **Fills:** bar data only. The next open is a proxy for the first fill;
  there are no bid/ask quotes, and costs are assumptions, not your account's
  fees.
* **The daily close** is the last trade, not the CME settlement price.

## 3. Results: legacy vs final

* **LEGACY (before this audit, do not mix with the new runs):**
  * the README tables "Results (test set 2021-07-14 to 2025-05-30)";
  * the old Timestral table;
  * the old images in `results/nq/*.png` and `results/synthetic/`.

  Those runs dropped flat moves, removed spikes using the next bar, used
  calendar-resampled hourly candles and had no cost model.
* **Interim (cloud, superseded):** `results/nq/runs/20261102T053438Z_286e22_holdout_all_tf`.
  It was made before the last quarantine fix (label bar suspect). Don't use
  it for slides. Its findings were the same as run A's: 1-min +0.9 pp, not
  tradable.
* **FINAL-CODE runs:** any `results/<source>/runs/*` folder created with the
  current commit (check `git.commit` in `run.json`).

Run status (cloud container, 2026-11-02; this container is temporary and
`results/` is not in Git, so these folders do NOT reach your PC):

| Run | Status | Folder |
|---|---|---|
| A: holdout, all time frames, logreg + xgboost | **finished**. Small summaries copied into Git: `reports/cloud_run_A_holdout_close/` | `results/nq/runs/20261102T054428Z_2ac714_A_holdout_close` (cloud only) |
| B-G (body target, walk-forward, ablation, spike sensitivity, MLP, regimes) | **cancelled**, not started | none |

Run A used the final pipeline code. `run.json` shows commit 510c239 + dirty,
because the code was committed right after the run. The only later changes
are saving fitted models and the summary plot, which do not change the
results. Rerun A on your PC if you want an identical commit hash. The rest
of the minimal set (B, C, Timestral) has not been run with the final code.

## 4. Setup (PowerShell)

```powershell
cd C:\Users\aless\Desktop\py\Project_ai_ML_bigdata   # your project folder
git stash push config.py      # only if you changed config.py locally: keeps your edits
git pull
git stash pop                 # only if you stashed; resolve config.py by hand if asked
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-lock.txt     # exact tested versions (Python 3.13); if pip fails: pip install -r requirements.txt
python -m pytest tests -q                # expected: 50 passed, or 49 passed + 1 skipped without timestral
```

futures-data is expected next to the project (`..\futures-data`). Otherwise:
`$env:FUTURES_DATA_DIR="C:\Users\aless\Desktop\py\futures-data"`.
Optional: `pip install -r requirements-timestral.txt` and
`pip install -r requirements-ibkr.txt`.

## 5. Smoke tests (offline, under a minute, write only to `results\smoke\`)

```powershell
python main.py --smoke
python tools/timestral_direction.py --smoke
```

## 6. Minimal experiment set for the 10-slide presentation

Make a logs folder once: `New-Item -ItemType Directory -Force logs`. Run one
command at a time and wait for "Saved to ..." before starting the next. That
avoids duplicate runs and keeps the 4 CPU cores free.

```powershell
# A (main slide): next-close direction, 6 time frames, identical test sessions
python main.py --data-source nq --models logreg xgboost --max-train-rows 2000000 --tag A_holdout_close 2>&1 | Tee-Object logs\A.log
# B (is it tradable?): direction of the next candle's body, decided before it opens
python main.py --data-source nq --target next_candle_body_direction --models logreg xgboost --max-train-rows 2000000 --tag B_holdout_body 2>&1 | Tee-Object logs\B.log
# C (robustness): expanding walk-forward, 4 test blocks
python main.py --data-source nq --scheme expanding --folds 4 --frequency 1min 15min 1h session --models logreg xgboost --max-train-rows 1000000 --tag C_walkforward 2>&1 | Tee-Object logs\C.log
# Timestral-5M on common origins, paired with run A's models (replace <A folder>)
python tools/timestral_direction.py --model mistralai/Timestral-5M --frequency 1min 1h session --n-samples 2000 --baseline-run results\nq\runs\<A folder> 2>&1 | Tee-Object logs\T5M.log
```

The synthetic null experiment already exists (legacy, still valid as a null
check). Rerun it only if you want it in the new run-folder format:
`python main.py --data-source synthetic`.

Optional, only if time allows (not needed for 10 slides):

```powershell
python main.py --data-source nq --years 2014 2025 --frequency 1min 1h --regimes --models logreg xgboost mlp_ens --max-train-rows 300000 --tag G_regimes 2>&1 | Tee-Object logs\G.log
python main.py --data-source nq --frequency 1min 5min 1h --extra-features volume range rv time --context-features --models logreg xgboost --max-train-rows 2000000 --tag D_ablation 2>&1 | Tee-Object logs\D.log
python main.py --data-source nq --frequency 1min session --spike-policy legacy_remove --models logreg --tag E_spike_legacy 2>&1 | Tee-Object logs\E.log
python main.py --data-source nq_ticks --frequency tick 1min --models logreg xgboost --tag ticks 2>&1 | Tee-Object logs\ticks.log
```

Suggested slides:
1. Question and data
2. Timing/leakage design
3. Synthetic null result
4. Accuracy vs majority by time frame (A, `timeframe_comparison.png`)
5. Year-by-year stability (A, `by_year.csv`)
6. Close-to-close vs body target (A vs B)
7. Backtest after costs (A/B `backtest.csv`)
8. Walk-forward (C)
9. Timestral
10. Conclusions and limitations

## 7. Checking progress without duplicate runs

```powershell
Get-Process python -ErrorAction SilentlyContinue        # is a run still going?
Get-ChildItem results\nq\runs | Sort-Object LastWriteTime | Select-Object Name, LastWriteTime
Test-Path results\nq\runs\<folder>\backtest.csv          # True = that run finished
Get-Content logs\A.log -Tail 5
```

`run.json` is written when a run starts. `metrics.csv`, `backtest.csv` and
`timeframe_comparison.png` are written only at the end. A folder without
`backtest.csv` and no python process running is an interrupted run.

## 8. Resuming and preserving results

* **Experiments:** these are not resumable mid-run. Rerun the same command;
  it creates a NEW folder and never overwrites or deletes an earlier one.
  Ignore (or delete by hand) folders without `backtest.csv`.
  `trial_log.csv` keeps every run.
* **Tick and bar downloads:** resumable. Rerun the same command and it
  continues from `manifest.json` and `<day>.partial.parquet`. Days marked
  `complete`/`no_trades` are skipped, and `incomplete`/`failed_coverage`
  days are retried.
* **Never needed:** `git restore`, `git reset` or deleting `results\`.
  `results\` and `data_raw\` are not tracked by Git.

## 9. What to send back for interpretation (small files only)

From each finished run folder:
* `metrics.csv`, `paired_vs_majority.csv`, `backtest.csv`, `by_year.csv`,
  `dataset_info.csv`, `folds.csv`
* `run.json`
* `timeframe_comparison.png`
* the console log (`logs\A.log`, etc.)

For Timestral: `timestral_results.csv` from its run folder.

Do NOT send the `predictions_*.parquet` files or `models\` (large).

## 10. Reading the numbers

* An accuracy edge counts only if the 20-session block-bootstrap interval
  in `paired_vs_majority.csv` excludes zero, on test data. Even then it is
  exploratory, because this period has been inspected before.
* Profitability needs a positive net P&L under the `base` and `high` cost
  scenarios, with a daily-P&L interval above zero. Accuracy alone says
  nothing about profit.
* Run A findings (NQ, final code, test 2021-07-15 → 2025-05-30, 1,002 sessions;
  files in `reports/cloud_run_A_holdout_close/`):
  * **1min:** logreg +0.86 pp, xgboost +0.88 pp vs majority (20-session
    block CI about +0.70 to +1.05). 5min: +0.6 to +0.7 pp. The reversal
    rule alone gives +0.69 (1min) and +0.47 (5min): this is the bid-ask
    bounce.
  * **Not tradable:** from the next open, the 1-min reversal rule loses
    $0.79M at ZERO cost. With base costs every 1-min/5-min strategy is
    deeply negative (e.g. reversal −$17.9M over 1.37M trades).
  * **15min:** no detectable edge. **1h:** logreg −0.65 pp. **4h:**
    xgboost −2.1 pp. **Session:** all intervals include 0.
  * **Session-frequency net P&L** is positive for logreg/xgboost under the
    base costs (+$57k / +$44k), but the daily-P&L interval includes 0:
    no evidence.
  * **Conclusion:** no evidence of a profitable edge. 1,002 test sessions;
    exploratory, because this period has been inspected before.
