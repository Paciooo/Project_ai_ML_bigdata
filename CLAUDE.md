# Project context for Claude Code

Read README.md first. Scope: NQ futures (E-mini Nasdaq-100, $20 per point)
only; VIX was removed from the project.

## State

* Branch `research/nasdaq-multi-target-forecasting` (this worktree): multi-target
  study `multitarget.py` -> run `20261112T050812Z_82953a_final`, report
  `multitarget_report.py` -> `reports/multitarget_final/`, deck
  `presentation_multitarget/` (8 slides, built from its deck_data.json).
  Commission there is $2.50 per round trip (user's spec), $5 shown alongside.
  Exploratory verdict: realized volatility predictable (log-RV R² vs
  persistence +40% 5 min, +52% 1 h; vs HAR-OLS +6.8% / +26.1%); for returns no
  improvement demonstrated with the models, data and horizons tried (never
  write "returns are not predictable"); direction +0.69 pp but loses after
  costs. Selection is not blind; verify with `verify_multitarget.py`.
  Two documented deviations (majority bug fixed without refit; volatility
  selection score changed after test numbers were printed): RESEARCH_MULTITARGET.md.
* Original direction study below is unchanged.

* Final study: `study.py` -> intraday run `20261111T070520Z_949daa_final`
  (commit `ff445f3`) + daily run `20261111T103357Z_02bd49_daily_final`
  (commit `0970132`), joined by `tools/merge_runs.py` into
  `20261111T103500Z_final_with_daily`. Summaries in `reports/nq_study_final/`,
  deck in `presentation/` (built from `presentation/deck_data.json`).
* `verify.py` re-derives the test metrics (accuracy, trades, P&L, win rate,
  profit factor, session-end drawdown, equity, costs) from the saved
  predictions: 8,858 checks, 0 problems on the merged run. It does not rebuild
  data, refit models or check slide text; do not call it a full audit.
* The merged run folder also holds `walk_forward_status.csv`
  (`tools/wf_threshold_status.py`): daily walk-forward years 2-3 had too few
  validation sessions for a threshold and are "not evaluable" (old code traded
  them unfiltered: published 4/4 +$201.5k, corrected 2/2 evaluable +$118.7k).
* Verdict: no deployable strategy demonstrated; none of the 5
  validation-selected candidates passes the criteria. Do not describe any
  result as profitable or validated beyond what the selection rule in
  `report.py` states, and do not attribute daily profits to "being long in a
  rising market" (not shown). Descriptive test configurations with a positive
  interval are multiple comparisons, not confirmations.
* Deck: cover + 8 slides (limit 10); appendix is a separate file
  (`NQ_final_report_appendix.pptx/.pdf`).

## Rules

* Never rerun expensive jobs without telling the user; one run at a time.
* Never delete `results/`, `data_raw/` or the local futures-data folder.
* Costs are $5 per round trip, zero slippage, zero latency, 100% fills at the
  next open (idealised); keep saying so wherever P&L is shown. $12.50 / $23
  are assumed scenarios, not realistic or verified execution.
* The 2021-2025 test period is exploratory (seen in earlier phases).
* The user works on Windows PowerShell and prefers replies in Italian.
