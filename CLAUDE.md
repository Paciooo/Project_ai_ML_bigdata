# Project context for Claude Code

Read README.md first. Scope: NQ futures (E-mini Nasdaq-100, $20 per point)
only; VIX was removed from the project.

## State

* Final study: `study.py` -> intraday run `20261111T070520Z_949daa_final`
  (commit `ff445f3`) + daily run `20261111T103357Z_02bd49_daily_final`
  (commit `0970132`), joined by `tools/merge_runs.py` into
  `20261111T103500Z_final_with_daily`. Summaries in `reports/nq_study_final/`,
  deck in `presentation/` (built from `presentation/deck_data.json`).
* `verify.py` recomputes every test metric independently: 527 checks, 0
  problems on the merged run.
* Verdict: no deployable strategy. Do not describe any result as profitable
  or validated beyond what the selection rule in `report.py` states.

## Rules

* Never rerun expensive jobs without telling the user; one run at a time.
* Never delete `results/`, `data_raw/` or the local futures-data folder.
* Costs are $5 per round trip, zero slippage, idealised fills at the next
  open; keep saying so wherever P&L is shown.
* The 2021-2025 test period is exploratory (seen in earlier phases).
* The user works on Windows PowerShell and prefers replies in Italian.
