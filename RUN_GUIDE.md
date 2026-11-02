# Run guide: verified local results and audit repairs

Updated 2 November 2026. The planned experiments are complete. Do not rerun
training just to reproduce the presentation tables.

## Current evidence

The canonical results are indexed in [reports/README.md](reports/README.md).
Local runs A, B, C and NQ Timestral have completed. All four Timestral sizes
(5M, 28M, 123M, 209M) completed on nearest-expiry VIX futures. Raw-input labels,
saved classifier probabilities and cached Timestral directions were checked
locally. The verification details are in
`reports/verification/local_verification.json`.

The cloud A replica is superseded. Its XGBoost numbers differ from local A;
use only the local A snapshot in slides. Different environments/source states
mean the reason for the discrepancy is not uniquely established.

## Corrected economic policy

- Decision time = completed input bar availability + configured latency.
- Entry = first observed one-minute open at or after the decision, inside
  the target interval and target session. A suspect candidate is rejected.
- Exit = target bar close, strictly after entry.
- With the default one-second latency, one-minute targets have no observable
  entry before exit at the available price resolution. Their P&L is NaN and
  `economic_status=not_evaluable_no_entry_before_exit`, not zero-profit evidence.
- Other bars use delayed minute-open proxies. This is an approximation with
  assumed commissions/spread/slippage, not executable quote evidence.
- Thresholds are chosen on validation using the corrected execution policy
  and stored per row/fold. Missing thresholds raise an error.
- P&L is grouped by the target session. Drawdown is end-of-day, including the
  initial zero-equity point. Approximate roll exclusions remain a limitation.
- Body direction is an outcome definition. With positive latency the decision
  can occur after that body starts; it is not described as a pre-open decision.

The original prediction probabilities and classification scores were preserved.
Pre-repair files are saved inside each local run's `audit_backups/` directory.
Original training provenance stays in `run.json`; repairs are in `rescore.json`.

## Local setup and checks

From PowerShell in this project folder, using the existing `.venv`:

```powershell
$env:FUTURES_DATA_DIR = "C:\Users\aless\Desktop\py\futures-data"
python -m pip install -r requirements.txt
python -m pytest tests -q
python main.py --smoke
python tools/timestral_direction.py --smoke
```

Use the reported test count from the latest check in `docs/AUDIT_FIXES.md`.
`requirements-lock.txt` records the earlier Linux training environment, not
an instruction to replace the working Windows environment unnecessarily.

## Repair/verify existing local runs without refitting classifiers

```powershell
python -u tools/repair_results.py
```

This needs the original yearly minute Parquet files, saved fitted models,
VIX CSVs and cached weights. It backs up and updates local economic outputs,
reconstructs validation probabilities from the saved models, verifies labels
and Timestral forecasts, then refreshes the small report snapshots.
It does not place orders, modify raw inputs, or refit classifiers.

## Optional fresh experiments

The following are reproduction commands, not unfinished work:

```powershell
python -u main.py --data-source nq --models logreg xgboost --max-train-rows 2000000 --tag A_holdout_close
python -u main.py --data-source nq --target next_candle_body_direction --models logreg xgboost --max-train-rows 2000000 --tag B_holdout_body
python -u main.py --data-source nq --scheme expanding --folds 4 --frequency 1min 15min 1h session --models logreg xgboost --max-train-rows 1000000 --tag C_walkforward
python -u tools/coinflip_vix.py --simulations 10000 --timestral-models 5M 28M 123M 209M
```

Fresh experiments create unique folders. A/B/C classification should reproduce
in the same environment; economic outputs follow the corrected policy and
must not be compared against old P&L as if the execution assumptions matched.
For NQ Timestral, use `--baseline-run` with a completed local A folder.
Do not use `--confirmatory` casually: the displayed historical tests have
already been inspected and remain exploratory.

## What to share

The small canonical snapshots in `reports/` are sufficient to inspect the
reported tables and provenance remotely. `metrics.csv`, `paired_vs_majority.csv`,
`paired_vs_reversal.csv`, `backtest.csv`, `folds.csv`, `run.json` and
`rescore.json` distinguish training from repaired execution.
`local_timestral_NQ_5M/timestral_results.csv` verifies the NQ Timestral table.
Do not upload model weights, raw licensed data or all large prediction files
merely to include these summaries.

## Scientific conclusions and limits

There is a small exploratory classification advantage at short NQ horizons.
Reversal performance is consistent with microstructure effects, but does not
identify bid-ask bounce as the sole mechanism. VIX majority accuracy is 59.59%,
logistic regression 59.81% with an interval including zero improvement.
All four Timestral sizes score below the VIX majority baseline in this test.
Higher raw accuracy across instruments does not imply higher skill: compare
balanced accuracy and each instrument's own baselines.
VIX futures, spot VIX and realized S&P 500 volatility are distinct targets.
Use "profitability not established", not "proved untradable".

The presentation outline remains limited to ten slides in
`docs/MONTE_CARLO_VIX_RESULTS.md`.
