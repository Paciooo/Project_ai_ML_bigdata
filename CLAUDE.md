# Project context for Claude Code

Read README.md first. Scope: NQ futures (E-mini Nasdaq-100) only; VIX was
removed from the project.

## State

* Main project (branch `main`): multi-target study. `multitarget.py` -> run
  `20261112T050812Z_82953a_final`; `multitarget_report.py` ->
  `reports/multitarget_final/`; `presentation/build_deck.js` writes the main
  deck (cover + 9 slides: slide 4 task x timeframe map, slides 6-8 five models
  per task, slide 9 strongest evidence, slide 10 reproducibility + references)
  and the appendix A1-A5. Check with `verify_multitarget.py` (recomputes the
  numbers, not the bootstraps, data or models; do not call it a full audit).
  `tools/sample_checks.py` and `tools/timestral_same_sample.py` feed A3/A4.
* Wording: z = strength of evidence against each task's OWN naive benchmark,
  not a "best model overall"; volatility "error cut" = reduction of the
  SQUARED log-RV error (not RMSE; vs HAR-OLS stated separately); the ranking
  is not pre-registered (volatility score revised after test numbers);
  statistical accuracy only, economic value not studied; no option strategies.
* Known sample issues (README "Sample checks"): 304 of 1,002 daily test
  sessions lie in the 50-session flagged-bar window and are excluded (50 of
  them only because of a flag measured against a dropped 16:59 CT bar); 56
  more are excluded for label eligibility; 642 kept (ends 2025-01-30). Daily
  results = filtered sample, exploratory: never "volatility predictable on
  every horizon" without that caveat; the daily issue is documented, NOT
  resolved. Flags are computed before the schedule filter: describe them with
  `suspect_flag_details` (the values that generated them), not with
  filtered-sequence returns; timing/persistence are clues, not proof.
  Incomplete intraday candles are accepted; the < 0.4-point figure is a
  rescoring of saved TEST forecasts only (training, features and a refit not
  evaluated). Fixing either needs a new run: propose it, do not run it
  unasked.
* Prices are additively back-adjusted (TradeStation): percentage returns are
  compressed (vs ^NDX: std ratio 0.42 in 2006-2012, 0.90 in the test;
  re-levelled correlation 0.998; re-levelling = retrospective estimate from
  Yahoo, not the contract prices), README "Sample checks" C, appendix A6,
  `tools/yahoo_compare.py`. Effect on relative performance NOT measured
  (never write "should change little"); a fix (ratio adjustment) needs a new
  run.
* Yahoo day-by-day comparison: the reconciled daily close-to-close returns
  of all 16 flagged sessions and of the 292 excluded test sessions with a
  quote agree with ^NDX / NQ=F within 0.5 pp. "Confirmed" = closing returns
  only (range not in the rule): it suggests the quarantine also removes
  plausible sessions, it does not verify minutes or RV. Never call Yahoo a
  certification of data quality. A
  daily filter based on this comparison (about 921 test sessions) was proposed; the
  user chose "documentation only" (2026-11-13): no new run.
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
