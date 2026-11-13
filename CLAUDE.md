# Project context for Claude Code

Read README.md first. Scope: NQ futures (E-mini Nasdaq-100) only; VIX was
removed from the project.

## State

* Main project (branch `main`): multi-target study. `multitarget.py` -> run
  `20261112T050812Z_82953a_final`; `multitarget_report.py` ->
  `reports/multitarget_final/`; deck `presentation/` (cover + 8 slides, the
  last on reproducibility, built from its deck_data.json; slides 5-7 compare
  five models per task, slide 8 shows only the best model overall by
  validation z); check with
  `verify_multitarget.py` (recomputes the numbers, not the bootstraps, data
  or models; do not call it a full audit).
* The study is statistical only: no trading rules, P&L, costs or equity
  curves anywhere in the main project, at the user's request.
* Exploratory verdict: realized volatility predictable (log-RV R² vs
  persistence +40% 5 min, +52% 1 h; vs HAR-OLS +6.8% / +26.1%); for returns no
  improvement demonstrated with the models, data and horizons tried (never
  write "returns are not predictable"); direction +0.69 pp over the majority
  class, concentrated in 2021-2023 (intervals include zero in 2024-2025).
* Selection is not blind (volatility score revised after test numbers were
  printed); the majority-class bug was fixed without refit; the Timestral
  volatility calibration on validation is past-to-future only. See README.
* The earlier direction-and-trading study (study.py run, report.py,
  verify.py, NQ decks, IBKR tools) lives on branch `archive/nq-direction-study`.

## Rules

* Never rerun expensive jobs without telling the user; one run at a time.
* Never delete `results/`, `data_raw/` or the local futures-data folder; never
  commit NQ data, saved forecasts or model weights.
* The 2021-2025 test period is exploratory (seen in earlier phases).
* The executive summary (slide 2) is read by other students and compiled by
  the professor: it must stand alone.
* The user works on Windows PowerShell and prefers replies in Italian.
