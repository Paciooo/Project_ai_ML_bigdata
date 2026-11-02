# Current project context for Claude Code

Read this file first. It is the CURRENT handoff, updated 2 November 2026.
Do not treat old prompts/reviews in docs/history/ as outstanding tasks.
The final presentation must fit in **at most ten slides**.

## State

Implementation, local experiments and the subsequent correctness repair are
complete. Last full check: **70 tests passed** on Windows/Python 3.12.
Do not launch full training, downloads, additional model searches or brokerage
connections just to review the existing outcome. No orders are submitted.

Use these documents, in order:
1. README.md -- concise project map.
2. docs/AUDIT_FIXES.md -- defects fixed and local verification.
3. reports/README.md -- authoritative result folders.
4. docs/FINAL_RESULTS_SUMMARY.md and docs/MONTE_CARLO_VIX_RESULTS.md.
5. RUN_GUIDE.md -- optional reproduction commands and limitations.

## Authoritative runs (keep these exact folders)

| Evidence | Full local run under results/ | Small Git snapshot |
|---|---|---|
| NQ A, next-close | nq/runs/20261102T060506Z_892e22_A_holdout_close | reports/local_run_A_holdout_close/ |
| NQ B, candle-body | nq/runs/20261102T060756Z_a695aa_B_holdout_body | reports/local_run_B_holdout_body/ |
| NQ C, expanding | nq/runs/20261102T061204Z_cdd340_C_walkforward | reports/local_run_C_walkforward/ |
| NQ Timestral-5M | nq/runs/20261102T061421Z_ef43c1_timestral_Timestral-5M | reports/local_timestral_NQ_5M/ |
| VIX / coin controls | extensions/runs/20261102T065506Z_d1d861_coinflip_vix_front | reports/coinflip_vix_front/ |

Old cloud A, incomplete VIX runs, smoke output, obsolete logs and temporary
files were removed. The old cloud A is only available in earlier Git history;
never mix its XGBoost numbers with local A. Cross-run differences involved
environment/source-state differences; the cause was not isolated.
Keep the referenced audit_backups/ copies: they preserve pre-repair evidence.
The earlier synthetic null has one small snapshot in reports/synthetic_null_legacy/.

## Correctness repair already completed

- Execution uses the first observed clean minute open at/after decision and
  inside the target interval/session, with exit strictly after entry.
- One-minute targets with the default one-second latency have no observable
  minute-open entry before exit. Economic status is not evaluable and P&L is
  NaN, not a fake zero and not proof of untradability.
- Fold-specific thresholds are stored per prediction row, not DataFrame.attrs.
  Missing thresholds fail. Validation thresholds were reselected using SAVED
  models under the corrected execution policy; classifiers were not refitted.
- End-of-day drawdown includes initial zero equity. P&L is grouped by target
  session. Corrected outputs are in backtest.csv and rescore.json.
- Original classification labels/probabilities/directions remain unchanged.
- Body direction is an outcome definition, not a pre-open timing guarantee.

## Verification actually performed locally

reports/verification/local_verification.json records raw-data/model hashes.
It covers NQ source labels, 56 saved classifier/fold probability checks,
NQ Timestral replay at three frequencies, both local VIX classifiers and
all four real Timestral checkpoints. VIX labels were rebuilt from raw CSVs.
The NQ Timestral 48.00% etc. table was checked directly, not inferred from
coinflip summaries. Raw inputs/weights were never modified.

Raw datasets, full prediction files and weights are intentionally not in Git.
A remote checkout may inspect the small reports and verification evidence,
but cannot honestly claim to have independently rerun absent local files.
Source hashes in run.json describe the historical training state. rescore.json
describes later execution repairs. Historical intermediate-path references
are provenance, not current dependencies; reused coin draws survive in the
complete extension's CSV. Do not rewrite original provenance to hide changes.

## Interpretation to retain

- NQ A: XGBoost +0.89 pp at 1min and +0.74 pp at 5min versus majority.
- The reversal pattern is consistent with microstructure, not proof that
  bid-ask bounce is the only explanation. Paired-versus-reversal tables exist.
- VIX nearest expiry: majority DOWN 59.59%, logistic 59.81%, improvement
  interval [-0.97,+1.29] pp. Logistic balanced accuracy is only 50.91%.
- VIX Timestral accuracies: 5M 50.00%, 28M 48.28%, 123M 52.05%, 209M 51.83%.
  These are the SAME 928 origins with 64 observations of the SAME expiry.
- Other maturities are not separate predictors. Spot CSV is a reference audit.
  VIX futures direction, spot VIX and realized SPX volatility are different.
- Coin simulation: 10,000 conditional random policies on fixed outcomes,
  fair and matched expected UP share. Not a future-market confidence interval.
- No learned strategy has a positive lower daily-P&L interval under both
  base/high costs in the corrected simulations. Profitability is unproven.
  All historical tests shown are exploratory, with no fresh confirmatory claim.

## Scope and working conventions

Keep the project small. Prefer reviewing/rescoring saved evidence to adding
features or new models. Preserve raw data, weights and the user's config.py
choice. Do not auto-push or commit: PUSH_AUDIT_FIXES.md contains manual steps.
Do not recreate removed logs/tmp or duplicate old reports as a default action.
If new verification reveals a material defect, fix it and state which results
change; passing tests alone is not proof of financial validity.
