# What Can Machine Learning Predict in NASDAQ Markets?

University project (Waseda University, *AI, Machine Learning and Big Data for
Finance and Management*; assignment: data analysis with AI, machine learning
and/or big data, judged on originality, usefulness and feasibility). Author:
Alessandro Pacifici.

**Problem.** Markets are hard to forecast, but not every characteristic of
the market need be equally hard. **Research question:** which characteristic
of the next NASDAQ futures candle — its return, its direction or its realized
volatility — can machine learning predict, and on which candle length?

**Originality.** Experimental: one pipeline, one data set and one test period
for three targets and five horizons (5 minutes to 1 day), so "what is
predictable" is compared like for like across tasks. Applied: machine
learning and a new time-series foundation model (Mistral AI's Timestral) are
set against the classical HAR benchmark for realized volatility (Corsi, 2009).

**Usefulness and implications.** Realized-volatility forecasts are inputs to
risk limits and position sizing; the study shows on which horizons machine
learning adds to HAR and where it adds nothing. The study is purely
statistical: it contains no trading rules, P&L, costs or option strategies,
and it does not show that any forecast is profitable.

* Deck (cover + 9 slides): [presentation/NASDAQ_multitarget_report.pptx](presentation/NASDAQ_multitarget_report.pptx)
  ([PDF](presentation/NASDAQ_multitarget_report.pdf)); technical appendix A1–A5
  (reproduction, selection and statistics, sample checks, Timestral on the
  same sample, data profile): [presentation/NASDAQ_multitarget_appendix.pptx](presentation/NASDAQ_multitarget_appendix.pptx)
  ([PDF](presentation/NASDAQ_multitarget_appendix.pdf)).
* Result tables and charts: [reports/multitarget_final/](reports/multitarget_final/).
* Earlier phases of the project (next-candle direction with long/short
  trading tests, IBKR tick tools, the first decks) are kept in Git on the
  branch **`archive/nq-direction-study`**; they are no longer part of the
  main project.

**All results are exploratory.** The 2021–2025 test period was examined in
earlier phases of the project, and the volatility selection score was revised
after a first report had printed test numbers (see "Selection"). Nothing
here is a blind, confirmatory test.

## Results (run `20261112T050812Z_82953a_final`)

**Answer (exploratory): realized volatility is the most predictable
characteristic in this study, on every intraday candle length tested (5
minutes to 1 hour); on daily candles it beats persistence too, but only on a
filtered sample that under-represents turbulent periods (see "Sample
checks"), so the daily result is not a robust conclusion. For next-candle
returns no improvement over forecasting zero was demonstrated with the
models, data and horizons tried. For next-candle direction a small accuracy
gain was measured on 5-minute candles only, and it is not stable across the
test years.**

Metric conventions. For volatility, the "error cut" (an out-of-sample R²) is
the reduction of the **squared log-RV forecast error** relative to a
benchmark: 40.0% vs persistence for the 5-minute DNN means its squared error
is 40% smaller. This is not the RMSE reduction (RMSE 0.901 vs 1.163, i.e.
22.5% lower) and not the gain over HAR-OLS (6.8% smaller squared error than
HAR-OLS). For returns, R² is relative to the zero forecast; for direction,
the gain is accuracy minus the majority-class accuracy, in percentage points.

### What is predictable on which horizon (`timeframe_map.csv`, slide 4)

In each cell, the trained model with the highest validation evidence z for
that task and candle length, scored on the test (95% interval):

| Task (test) | 5 min | 15 min | 30 min | 1 hour | 1 day |
|---|---|---|---|---|---|
| Return: R² vs zero (%) | −0.04 [−0.10, +0.02] | −0.01 [−0.07, +0.04] | +0.00 [−0.02, +0.02] | −0.01 [−0.06, +0.04] | −1.95 [−6.11, +1.73] |
| Direction: accuracy gain vs majority (pp) | **+0.69 [+0.40, +0.98]** | +0.25 [−0.07, +0.57] | +0.16 [−0.12, +0.38] | +0.26 [−0.08, +0.55] | −0.78 [−1.52, +0.00] |
| Volatility: squared log-RV error cut vs persistence (%) | **40.0 [39.5, 40.4]** | **36.5 [35.1, 37.7]** | **42.6 [40.8, 44.3]** | **52.0 [50.0, 54.1]** | **18.7 [9.7, 24.5]** |
| Volatility: squared log-RV error cut vs HAR-OLS (%) | **6.8 [6.2, 7.3]** | **10.3 [8.8, 11.6]** | **16.7 [14.5, 18.7]** | **26.1 [23.8, 28.5]** | 7.6 [−0.9, 14.6] |

Bold: interval above zero. The gain over HAR-OLS grows with candle length up
to one hour; on daily candles the interval includes zero.

> **Daily caveat.** The 1-day column is a filtered sample: 642 of the 1,002
> daily test sessions. 304 sessions lie within 50 sessions after a session
> with a flagged 1-minute bar and are excluded by the data-quality filter
> (another 56 for other reasons, below), so turbulent periods are
> under-represented. The daily results are exploratory and hold for this
> filtered sample only; they say nothing robust about the unfiltered daily
> series.

### Five models per task (slides 6–8, `comparison_by_task.csv`)

One model per family on the task's best validation timeframe (5 minutes for
all three): a linear model (Ridge, or logistic regression for direction),
random forest, XGBoost, the best network architecture on validation, and
Timestral (frozen weights; for volatility with the supervised calibration).
Rows are ranked by validation z; the values are test results. Timestral is
scored on its fixed random sample of 20,000 test candles.

| Task | #1 | #2 | #3 | #4 | #5 |
|---|---|---|---|---|---|
| Return (R² vs zero) | DNN 256-128-64-32 −0.04% | XGBoost +0.00% | Ridge −0.03% | Random forest −0.05% | Timestral −2.55% |
| Direction (pp vs majority) | DNN 256-128-64-32 +0.69 | Random forest +0.62 | Logistic +0.50 | XGBoost +0.58 | Timestral −0.59 |
| Volatility (error cut vs persistence / vs HAR-OLS) | DNN 128-64-32 40.0% / 6.8% | XGBoost 42.0% / 9.9% | Random forest 37.1% / 2.3% | Ridge 35.7% / 0.1% | Timestral + calibration 12.2% / −35.9% |

* **Return:** no model improves on the zero forecast on any candle length; the
  5-minute network's forecasts span less than ±0.5 bp against a typical move
  of 7.7 bp. Consistent with near-random-walk returns at these horizons, but
  not a proof of it.
* **Direction:** the 5-minute gain of +0.69 pp is about 7 more right calls per
  1,000 candles, and comes from the early test years: +1.18 / +1.33 / +0.82
  pp in 2021 / 2022 / 2023 (intervals above zero), +0.02 and −0.12 pp in 2024
  and 2025 (intervals include zero; chart 2, `direction_by_year`).
* **Volatility:** HAR-OLS already captures volatility clustering, mean
  reversion and the intraday pattern (36% error cut vs persistence at 5
  minutes); what machine learning adds is the gain over HAR-OLS. It concerns
  realized, not implied, volatility (Andersen & Bollerslev, 1998).
* **Deeper networks:** no consistent gain; the best depth differs by task and
  timeframe, and differences between networks are small next to the gap to
  the benchmark.

### Strongest evidence (slide 9)

Ranking rule: (1) every model × timeframe is scored on validation by its
improvement over its task's naive benchmark; (2) z = that improvement / its
block-bootstrap standard error; (3) the highest z wins within each task.
Across tasks, z only says where the evidence of beating the **task's own**
naive benchmark is strongest; it is not a common scale of usefulness and does
not show that one task's model is "better" than another's. Task winners:
volatility DNN 128-64-32 at 5 minutes (z 44.0), direction DNN 256-128-64-32
at 5 minutes (z 11.9), return DNN 256-128-64-32 at 5 minutes (z 3.1). The
rule was applied as computed on validation; it is not a pre-registered
choice, because the volatility score itself was revised after a first look
at test numbers. On the test XGBoost (#2 by validation z) cuts the
volatility error slightly more (42.0% vs 40.0%); the ranking was not
re-chosen on the test.

**Statistical vs economic performance.** The metrics above are statistical
and are reproducible from the saved forecasts. Whether a statistical gain is
economically useful was not studied: commissions, bid-ask spread, slippage
and latency would decide that, and a small gain such as +0.69 pp in
direction may not survive them.

## Data analysis (test sample, `data_profile.csv`)

| Candle | Rows (test) | UP / DOWN / doji | Return s.d. (bp) | ACF return | ACF log RV |
|---|---|---|---|---|---|
| 5 min | 273,340 | 49.9% / 48.3% / 1.8% | 7.7 | −0.005 | 0.75 |
| 15 min | 90,189 | 50.7% / 48.3% / 1.0% | 13.3 | +0.003 | 0.84 |
| 30 min | 44,395 | 51.1% / 48.1% / 0.8% | 18.5 | −0.002 | 0.84 |
| 1 hour | 21,507 | 51.5% / 48.0% / 0.5% | 26.2 | −0.000 | 0.83 |
| 1 day | 642 | 54.2% / 45.8% / 0.0% | 116.9 | −0.001 | 0.81 |

ACF = lag-1 autocorrelation (intraday: pairs within a session; daily:
consecutive sessions). Candle direction is close to a coin flip, consecutive
returns are practically uncorrelated, while log realized variance is strongly
persistent. This matches the results: volatility is the predictable target.
The same statistics for the validation block are in `data_profile.csv`.

### Deviations found while reporting

1. **Direction majority bug.** The first run computed the majority class over
   all training rows, counting dojis as "not up", so for 5, 15 and 30 minutes
   it chose DOWN, a weak baseline that inflated the gains to about +2 pp.
   Fixed in `multitarget.majority_direction` (non-doji candles only).
   `tools/mt_fix_majority.py` recomputed the constant from the training
   data, without refitting any model; the previous file is kept as
   `dataset_info_before_majority_fix.json`.
2. **Volatility selection score revised after seeing test numbers** (see
   "Selection").
3. **Timestral volatility calibration** (review of 2026-11-12): the first
   report scored the calibrated Timestral variant on the same validation rows
   used to fit its calibration. A first fix used 2-fold cross-fitting, but
   one fold calibrated on the LATER half and scored the earlier one, which
   is not causal. Its validation score is now past-to-future only (below):
   5 min selection z 8.19 → 8.09 (2-fold) → 6.37, log-RV R² vs persistence
   +18.4% → +18.2% → +18.3% [+12.3, +22.7] on the 16,036 scorable rows
   (daily −32.1% → −39.5%). The ranking of the selected configurations, the
   five configurations and every test number are unchanged.

## Sample checks (`tools/sample_checks.py` → `sample_checks.json`, appendix A3)

Measured from the 1-minute data and the saved forecasts, without training.

**A. Why the daily test ends on 2025-01-30.** All 85 sessions from 2025-01-31
to 2025-05-30 are dropped. 84 of them fall within 50 sessions after a session
containing a flagged 1-minute bar: `study.build` keeps a row only if none of
the last 50 candles contains one (`clean50`), and `quality.flag_suspect`
flags any 1-minute bar whose log return against the previous bar exceeds 2%
in absolute value. The flags are on 2025-02-03, 2025-04-07 and 2025-04-09;
2025-01-31 itself is dropped because its label session (2025-02-03) contains
the flag.

*Which return generated each flag (diagnostic corrected).* The flag is
computed in `experiment.load_minutes` on the unfiltered 1-minute sequence,
before `timeframes.add_sessions` drops the bars outside the CME schedule. The
first version of this diagnostic recomputed the returns after that filter, so
for 5 of the 16 in-schedule flags it showed returns close to zero against a
different predecessor. `load_minutes` now records, for every flag, the return
that triggered it, the predecessor's time and price
(`quality.suspect_details`, report key `suspect_flag_details`); the tool uses
those values and reports the filtered-sequence returns separately
(`filtered_*`). Flags and sample are unchanged. On the original sequence all
21 flags (16 in schedule, 5 outside it and dropped) exceed 2%.

| In-schedule flags | n | Verified facts | Clues (not proof) |
|---|---|---|---|
| Out-of-schedule predecessor | 5 (2010-03-17, 2013-03-12, 2017-03-24, 2019-03-14, 2023-03-16) | the predecessor is a 16:59 CT bar (daily break, dropped) 989–991 points below the bars just before and after it; the flagged bar returns to the earlier level (+7% to +27% vs that bar, under 0.05% vs the last kept bar) | same time and offset each time, in March: looks like a data artefact; cause not established |
| Move persists 30 minutes later | 9 (2020-03-09, 2020-03-23, 2022-02-28, 2022-07-13, 2022-10-13, 2022-11-10, 2025-02-03, 2025-04-07, 2025-04-09) | −4.5% to +2.6% vs an in-schedule predecessor, still beyond 2% in the same direction 30 in-schedule minutes later | 5 follow a weekend gap; the 2022 ones fall at 8:30 ET (US data releases); 2025-04-09 at 13:19 ET. Consistent with genuine moves, not proof of them |
| Unresolved | 2 (2020-03-02, 2020-03-16) | −2.4% and −3.0%, partly retraced within 30 minutes (−1.7%, −1.6%) | weekend reopening / 15-minute gap; nothing decisive |

*Impact on the daily sample.* 304 of the 1,002 daily test sessions (train
150, validation 115) lie in 50-session windows after an in-schedule flag and
all 304 are excluded; 50 of them only because of the 2023-03-16 flag, whose
predecessor is the dropped 16:59 CT bar, 254 because of flags whose move
persists. Another 56 test sessions are excluded because their label is not
the next session (`timeframes.next_label_eligible`; the conditions overlap), so 642
are kept. The **daily sample under-represents turbulent periods**: the daily
results are exploratory and refer to this filtered sample. Intraday the
window is 50 candles, a negligible loss. Changing the filter changes the
sample and needs a new daily run; this was not done, so the daily limitation
is documented, not resolved (see "Future experiments").

**B. Incomplete intraday candles.** `timeframes.aggregate` stores the number
of 1-minute bars in each candle (`n_minutes`) but does not use it: a
full-length 5-minute candle with 4 of its 5 minutes is accepted as complete
(`tests/test_timing_and_splits.py` documents this). TradeStation writes no
bar for a minute without trades; scheduled halts and the overnight break lie
outside the candles. Share of full-length candles with missing minutes:

| Candle | All years | Test | Test, 08:30–15:00 CT | Test rows whose label candle is incomplete |
|---|---|---|---|---|
| 5 min | 18.0% | 0.23% | 0.003% | 601 of 273,340 |
| 15 min | 25.8% | 0.43% | 0.008% | 385 of 90,189 |
| 30 min | 30.6% | 0.65% | 0.008% | 279 of 44,395 |
| 1 hour | 34.3% | 1.01% | 0.000% | 201 of 21,507 |

Missing minutes are concentrated overnight in 2006–2017 (54% of 5-minute
candles in 2006, 0.02% in 2025), consistent with thin trading; in the test a
few sessions with long gaps coincide with quarterly contract-roll days
(e.g. 2021-09-09, 2022-09-08), consistent with dataset gaps.

*Scope of the sensitivity check.* The saved test forecasts were rescored
without the rows whose label candle is incomplete (point estimates; HAR-OLS,
the 5-minute networks and XGBoost; return, direction, volatility): the
rescored metrics move by at most 0.4 points (e.g. 5-minute DNN volatility
39.99% → 40.00%, 1-hour XGBoost 51.95% → 52.34%). This check does **not**
measure the effect of incomplete candles in the training data, their effect
on the features of the rows that were kept, or the result of re-training on
corrected data. The conclusions are unchanged in this check on the test
sample; the impact on the historical training data was not evaluated.

**C. External comparison with Yahoo Finance: daily returns of the excluded
sessions, and back-adjusted percentage returns.** `tools/yahoo_compare.py` builds daily candles from the
1-minute bars (same session calendar as the study) and compares them with
Yahoo Finance daily data: NQ=F (front-month futures, not back-adjusted) and
^NDX (cash index, close at 16:00 ET). Only summary statistics are saved
(`yahoo_check.json`, appendix A6); the Yahoo files stay in the gitignored
`data_raw/yahoo/`.

*Measured.* The TradeStation series is back-adjusted **additively**: every
roll gap is added to all earlier prices. Its level is about 2,200 points
above the unadjusted futures price in 2006–2022, about 770 in 2024 and 170 in
2025 (median of Yahoo NQ=F minus futures-data per year). Additive adjustment
keeps point changes, not percentages, so a log return computed on the
adjusted price is scaled by roughly unadjusted/adjusted price, a factor that
changes slowly over time. Daily log returns, futures-data at 15:00 CT
(= 16:00 ET) against ^NDX:

| Period | Std ratio (as used) | Beta (as used) | Std ratio, re-levelled | Correlation, re-levelled |
|---|---|---|---|---|
| 2006–2012 | 0.42 | 0.42 | 0.98 | 0.996 |
| 2013–2020 | 0.74 | 0.74 | 1.00 | 0.999 |
| Train + validation (to 2021-07-14) | 0.59 | 0.56 | 0.99 | 0.998 |
| Test (2021-07-15 → 2025-05-30) | 0.90 | 0.89 | 0.99 | 0.998 |

"Re-levelled" adds back an estimated offset: the centred 21-day rolling
median of Yahoo NQ=F minus futures-data. This is a retrospective
reconciliation built from Yahoo itself; it does not recover the original
prices of the individual contracts and is not a fully independent
validation. After it, the daily returns are close to the cash index (whole
period: correlation 0.998, beta 0.99, mean absolute difference 0.05
percentage points), which is consistent with the gap in return scale coming
from the additive adjustment. Against Yahoo NQ=F
itself the match is looser (correlation 0.96) because NQ=F has its own roll
artefacts (e.g. 2020-03-23: −6.0% against +0.2% for ^NDX).

*Consequences (not measured).* Returns and realized volatility in the study
are in "adjusted-percentage" units: by the estimate above, 2006–2012 returns
are about 0.4× and test returns about 0.9× the percentage moves of the
traded contract, and log-RV is shifted by about 2·log of that factor (about
−1.7 in 2006–2012, −0.3 to 0 in the test). The sign of a within-contract
change is not affected by an additive shift. The factor changes over time
and differs between training and test, and the 2% flag threshold
corresponds to about 4–5% moves in 2006–2012. **The effect on the relative
performance (models vs benchmarks) was not measured**: comparing a model and
its benchmark on the same series does not guarantee that their difference
stays the same after the correction. None of this was re-estimated: it needs the series rebuilt with ratio
(multiplicative) adjustment or per-contract returns, and a new run (see
"Future experiments"). This check does not explain the 16:59 CT bars about
990 points off (A above): that is not the size of the level offset on those
dates (about 1,950–2,340 points).

*Day by day: do the daily returns of the excluded sessions agree with Yahoo?*
A daily session is called **confirmed** when its re-levelled close-to-close
return is within 0.5 percentage points of ^NDX (15:00 CT close) or of Yahoo
NQ=F (session close); the typical gap to ^NDX is 0.03 pp (95th percentile
0.16 pp). Sessions without a Yahoo quote are counted apart. "Confirmed"
refers to closing returns only: the daily range is computed and reported but
is not part of the rule, so a session with a wrong high or low would still be
confirmed if its closes agree. The check does not validate the whole daily
candle, the single minutes or the realized volatility.

* All 16 sessions with an in-schedule flagged minute are confirmed; their
  daily range is also close to NQ=F's (descriptive, not part of the rule)
  (e.g. 2023-03-16, the 16:59 CT case: +2.68% vs
  ^NDX +2.66%, range 3.40% vs 3.32%; 2020-03-16, unresolved above: −13.4% vs
  −13.0%; 2025-04-09: +11.1% vs +11.4%). The 16:59 CT bars are outside the
  schedule and dropped; on the other flagged days the cash index shows a
  large move too.

| Daily sessions | With a Yahoo quote | Confirmed | Median gap to ^NDX |
|---|---|---|---|
| Test, in the 50-session windows (excluded) | 292 of 304 | 100% | 0.034 pp |
| Test, outside the windows | 658 of 698 | 99.5% | 0.030 pp |
| All splits, in the windows | 551 of 569 | 99.6% | 0.032 pp |
| All splits, outside the windows | 4,214 of 4,442 | 99.7% | 0.032 pp |

* Only 13 sessions in 2006–2025 disagree (3 in the test: 2023-09-15,
  2023-12-18, 2024-12-23). Almost all fall on a quarterly expiry Friday or
  the Monday after it, consistent with the contract switch in the continuous
  series rather than with bad prices; 2020-03-18/19 were futures limit-down
  days. None of them is among the excluded sessions.

**Conclusion.** The reconciled daily returns of the excluded sessions agree
with the Yahoo references within the chosen tolerance. This suggests that
the quarantine also removes plausible sessions, but it does not verify the
correctness of the single minutes or of the realized volatility; Yahoo is
not a certification of data quality. A daily filter based on this check (a flagged minute excludes a
session only if Yahoo does not confirm its daily candle) would currently
remove no session and keep about 921 of the 1,002 daily test sessions
(the 81 whose label is not the next session stay out). This was **not**
applied: it needs a new daily run (see "Future experiments"), so the daily
limitation remains documented, not resolved. The check uses only end-of-day
data and cannot validate single minutes; Yahoo NQ=F and ^NDX have their own
errors (one reference is enough to confirm a day for that reason).

## Timestral on the same sample (`tools/timestral_same_sample.py`, appendix A4)

On exactly the same random test candles (saved forecasts, 95% block
intervals in `timestral_same_sample.csv`):

| Candle | Return R²: Timestral / DNN | Direction pp, Timestral: body / close-to-close | Direction pp, DNN (body) | Volatility error cut vs persistence: Timestral / DNN | Correlation with next log RV: Timestral spread / current RV |
|---|---|---|---|---|---|
| 5 min | −2.55% / +0.20% | −0.59 / −0.66 | +0.51 | +12.2% / +39.6% | 0.76 / 0.76 |
| 15 min | −2.23% / −0.10% | +0.01 / −0.08 | +1.09 | −37.1% / +34.8% | 0.74 / 0.84 |
| 30 min | −1.46% / +0.02% | −1.76 / −1.88 | +0.14 | −56.2% / +41.3% | 0.71 / 0.84 |
| 1 hour | −1.59% / +0.00% | −2.28 / −2.36 | +0.02 | −74.3% / +41.8% | 0.64 / 0.83 |
| 1 day | −1.83% / +0.17% | −1.87 / −1.56 | +0.00 | −19.3% / +18.7% | 0.75 / 0.82 |

* **Target:** Timestral forecasts the next close (direction = median vs last
  close) while the study scores the candle body. Scored on the close-to-close
  direction it does no better, so the mismatch does not explain its result.
* **Sample:** the trained models keep their advantage on Timestral's own
  20,000-candle sample (daily: all 642 sessions), so the gap is not a
  sampling artefact.
* **Volatility:** Timestral's volatility is indirect, the q10–q90 spread of
  its price forecast mapped to log RV by a linear calibration fitted on
  validation; the trained models forecast log RV directly. The raw spread
  correlates with next log RV slightly less than the current RV does, and the
  calibrated forecast is worse than persistence beyond 5 minutes.
* **Scope:** one small model (5M parameters), frozen weights, a 512-close
  context, one market. These results do not generalise to time-series
  foundation models in general (Mistral AI, 2026).

## Data

* Instrument: **NQ**, E-mini Nasdaq-100 futures (CME), back-adjusted
  continuous TradeStation 1-minute series (sessions before the 2025-06-02
  confirmatory hold-out), $20 per index point. The adjustment is additive:
  percentage returns are compressed by up to about 60% in early years and
  about 10% in the test (estimated against Yahoo Finance, "Sample checks" C).
  No QQQ, MNQ, cash index or VIX data enters features or targets; Yahoo
  daily data (^NDX, NQ=F) are used only for that external check.
* Timeframes: 5 min, 15 min, 30 min, 1 hour, 1 day (one CME session,
  17:00–16:00 CT), all rebuilt from the 1-minute bars inside each session
  (`timeframes.aggregate`; `30min` was added to `FREQUENCIES` for this study).
  1-minute candles are excluded: the realized variance of a 1-minute candle
  from 1-minute data would be a single squared return.
* Same chronological boundaries for every timeframe and task; labels that
  cross a split boundary are purged; intraday labels never cross a session.
  Actual sample dates and rows (signal sessions, `dataset_info.json`):

| Timeframe | Train | Validation | Test |
|---|---|---|---|
| 5 min | 2006-01-10 → 2017-08-24, 769,058 | 2017-08-25 → 2021-07-14, 269,167 | 2021-07-15 → 2025-05-30, 273,340 |
| 15 min | 2006-01-10 → 2017-08-24, 263,691 | 2017-08-25 → 2021-07-14, 88,336 | 2021-07-15 → 2025-05-30, 90,189 |
| 30 min | 2006-01-10 → 2017-08-24, 132,606 | 2017-08-25 → 2021-07-14, 44,513 | 2021-07-15 → 2025-05-30, 44,395 |
| 1 hour | 2006-01-10 → 2017-08-24, 64,813 | 2017-08-25 → 2021-07-14, 21,642 | 2021-07-15 → 2025-05-30, 21,507 |
| 1 day | 2006-03-13 → 2017-08-23, 1,991 | 2017-08-25 → 2021-07-13, 822 | 2021-07-15 → **2025-01-30**, 642 |

  The daily sample starts later because the 50-candle moving-average
  feature needs 50 sessions of history. It ends on 2025-01-30 because of the
  flagged-bar filter (measured cause in "Sample checks").
* The test period was already inspected for the direction task in earlier
  project phases, so the test results are **exploratory**, not a clean
  confirmation.

## Targets (all for candle t+1, decided at the close of candle t)

| Task | Target | Benchmark | Metrics |
|---|---|---|---|
| Return | r = log(C(t+1)/C(t)) in basis points | zero return (also: training mean) | MAE, RMSE, out-of-sample R² vs zero, sign accuracy, correlation |
| Direction | UP if C(t+1) > O(t+1), DOWN if lower; doji kept and counted as a miss | majority class of the non-doji training candles | accuracy, balanced accuracy, precision, recall, F1, confusion matrix |
| Volatility | RV(t+1) = Σ squared 1-minute log returns inside candle t+1 (first term log(C/O) of its first minute) | persistence RV(t+1) = RV(t); HAR-OLS (also: rolling mean of 22 candles) | MAE/RMSE of log RV, R² vs persistence and vs HAR-OLS, QLIKE on the variance level, correlation; RV direction (UP if RV(t+1) > RV(t)): accuracy, balanced accuracy, precision, recall, F1 vs majority and vs HAR-OLS |

Squared returns are a variance proxy; volatility is its square root. Models
forecast log RV; the level forecast used for QLIKE is
`c · exp(log forecast)`, with `c` estimated on the training residuals
(Timestral: on the calibration rows).

## Features (known at the close of candle t)

Past returns (lags 1, 2, 3, 5), momentum (5, 10), rolling volatility (5, 10,
20), relative volume, range, body, wicks, RSI(14), moving-average gaps (20,
50), time of day (sin/cos), and log realized variance of the last 1, 5 and 22
candles plus, intraday, the mean RV of the next candle's time slot over the 5
previous sessions. HAR-OLS uses only the realized-variance features.

## Models

See slide 5 and `reports/multitarget_final/models.csv` (every fit, with
epochs / best iteration / chosen hyper-parameter).

* Benchmarks: zero return, training mean, majority class, persistence,
  rolling mean of 22 candles.
* Statistical: HAR-OLS (volatility), Ridge (α ∈ {0.1, 10, 1000} chosen on
  validation; return, volatility), logistic regression (C ∈ {0.01, 1} chosen on
  validation; direction).
* Machine learning: random forest (200 trees, depth 8), XGBoost (depth 4,
  learning rate 0.03, early stopping on validation).
* Deep learning: feed-forward networks 64-32-16 (the repository's original
  architecture), 64-32, 128-64-32, 256-128-64-32; ReLU, AdamW, weight decay
  1e-4, early stopping on the validation loss (patience 4). Return networks
  are trained on a winsorised target (0.1% / 99.9% training quantiles) and
  scored on the raw one.
* **Mistral AI — Timestral-5M** (`mistralai/Timestral-5M`, Mistral's
  pretrained time-series foundation model, local weights, no API, no cost; it
  is not a language model and does not read text). Role in the repository:
  `tools/timestral_direction.py` (unchanged) ran it
  zero-shot for next-close direction. Here two variants are distinguished:
  * **Timestral-5M, frozen weights** (return, direction; zero-shot): from the
    last 512 closes of the same timeframe it forecasts quantiles of the next
    close. Return forecast = log(median / last close). Direction = sign of
    (median − last close), i.e. a next-**close** forecast, while the
    classification target is the **body** of the next candle (close vs
    open); the mismatch is part of what is being tested.
  * **Timestral-5M + supervised volatility calibration** (volatility; not
    zero-shot): the q10–q90 spread gives a variance proxy, mapped to log RV
    by a linear regression fitted on **validation targets**. For the
    validation score used in selection the calibration runs only from past
    to future: the validation sample is cut into 5 consecutive blocks of
    origin sessions, and block k (k = 1..4) is scored with a regression
    fitted on every earlier row whose target session ends before block k
    starts (expanding window), so each calibration target is already known
    when the scored forecasts are made. Block 0 has no earlier calibration
    and is excluded from the score, for Timestral and for the benchmarks
    compared with it (about 16,000 of the 20,000 intraday rows and 657 of
    the 822 daily rows are scored). Test forecasts use the calibration
    fitted on the whole validation sample, whose targets all precede the
    first test origin. Coefficients and the calibration / evaluation date
    ranges of every block are in `timestral_calibration.json`.
  Both are scored on a fixed random sample of 20,000 validation and 20,000
  test origins per timeframe (all of them for daily candles), against the
  benchmarks on exactly those rows. Pretraining-data overlap with NQ is
  unknown. Timestral-28M, -123M and -209M (config and weights available in
  `data_raw/model_cache`) were **not tested in this
  study**. There is no Mistral chat/LLM API integration in the repository.
* Considered, not run: LSTM. The feature networks already reach their
  early-stopping optimum within a few epochs, and a sequence model would add
  cost without a reason in the validation results.

## Selection

`multitarget_report.py`. The criteria use validation data only, but the
selection is **not blind**:

* **Initial criterion (first report):** evidence score = z-statistic
  (20-session moving-block bootstrap) of the paired validation improvement
  over the naive benchmark: squared-error reduction vs zero (return), accuracy
  gain vs majority (direction), **QLIKE reduction vs persistence**
  (volatility).
* **Revision (after the first report had printed test numbers):** on
  validation >99% of persistence's QLIKE came from a handful of limit-locked
  candles in March 2020 with almost zero variance, so the volatility score
  was ~1 for every model. The volatility score was replaced by the
  **squared-error reduction of log RV vs persistence**. Return and direction
  scores are unchanged.
* **Criterion actually used:** z of the validation squared-error reduction
  vs zero (return), of the accuracy gain vs majority (direction), and of the
  squared-error reduction of log RV vs persistence (volatility). Each task
  gets its highest-z configuration; the two remaining slots go to the two
  tasks whose best configuration on a **different timeframe** has the highest
  z (at most two per task). Benchmarks are never selected. The five are then
  scored on the test block.

Because the revision came after test numbers were seen, and the test period
had been examined before, the selection and all test results are exploratory.

The deck uses these z scores in three ways: the five-model tables (one model
per family on each task's best validation timeframe), the timeframe map (the
highest-z trained model per task and timeframe) and the task winners on
slide 9. The earlier five-configuration selection (`selected_five.csv`, the
rule above with two extra slots on a different timeframe) still provides the
charts.

## Statistics

* Differences (accuracy gains, RV-direction gains vs majority or HAR-OLS,
  selection z): moving-block bootstrap with 20-session blocks of the paired
  per-row differences, 2,000 replicates.
* Relative improvements (R² vs zero, R² of log RV vs persistence and vs
  HAR-OLS, QLIKE reduction): model and benchmark losses are resampled
  **together** by session blocks and the ratio 1 − Σ model loss / Σ benchmark
  loss is recomputed in every replicate (`block_boot_ratio`). (The first
  report divided an interval of the loss difference by a fixed denominator;
  point estimates were unaffected, intervals moved by up to a few tenths of a
  point.)
* Return chart (slide 6): each decile's realized-mean interval is a
  20-session block bootstrap within the decile (first report: independent
  standard errors).
* Classification convention (`cls_stats`, scikit-learn's
  `zero_division=0`): precision = 0 without positive predictions, recall = 0
  without positive cases, F1 = 2TP / (2TP + FP + FN), and 0 when that
  denominator is 0. (First report: NaN; this changed only the
  RV-direction precision/F1 of the persistence benchmark, which never
  predicts UP.)


## Reproduce

Windows PowerShell from the repository root. Environment of the final run
(`run.json`): Python 3.12.3, Windows 11, CPU; exact versions in
`requirements-lock.txt` (numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1,
xgboost 3.4.1, torch 2.15.0+cpu, …) and `requirements-timestral.txt`
(timestral 1.0.0 at a pinned commit). Node.js 18+ for the deck (pptxgenjs
3.12.0 pinned in `presentation/package-lock.json`). Timestral-5M weights must
be in the Hugging Face cache (first download:
`python -c "from timestral import TimestralForecaster; TimestralForecaster.from_pretrained('mistralai/Timestral-5M')"`
with `HF_HUB_OFFLINE` unset; `multitarget.py` itself runs offline).

The NQ 1-minute data (TradeStation, 2006–2025) are licensed and **not in
Git**; neither are `results/` (saved forecasts) and `data_raw/` (model cache).
Point `FUTURES_DATA_DIR` to a local copy (default `..\futures-data`).

```powershell
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements-lock.txt -r requirements-timestral.txt
$env:FUTURES_DATA_DIR = "C:\path\to\futures-data"
python multitarget.py --tag final                       # full training, ~30 min on 16 CPU threads
python multitarget_report.py results\multitarget\runs\<run id> reports\multitarget_final
python tools\timestral_same_sample.py results\multitarget\runs\<run id> reports\multitarget_final
python tools\sample_checks.py results\multitarget\runs\<run id> reports\multitarget_final   # reads the 1-minute data, ~1 min
copy reports\multitarget_final\deck_data.json, reports\multitarget_final\chart_*.png, reports\multitarget_final\sample_checks.json, reports\multitarget_final\timestral_same_sample.json presentation\
python verify_multitarget.py results\multitarget\runs\<run id> reports\multitarget_final presentation
cd presentation
npm ci                                                  # installs pptxgenjs 3.12.0 from package-lock.json
node build_deck.js                                      # main deck + appendix
cd ..
python -m pytest -q                                     # from the repository root
```

Pipeline check in a minute:
`python multitarget.py --freqs 1h session --years 2015 2019 --quick --timestral-n 500`.

**What the checks cover.** `verify_multitarget.py` recomputes, from the saved
forecasts and without `multitarget_report.py`, every point estimate in the
three results tables (including the past-to-future Timestral calibration,
with a check that no calibration target falls after the origins it is used
for, and the HAR-OLS comparisons), the ranking rules, the five-model tables,
the timeframe map, the selected-five fields, every value of
`data_profile.csv`, the per-year direction gains and the deck data (run id,
rows, charts, appendix inputs, no trading fields, no VIX). It also checks
the content of `sample_checks.json`: every flag above the threshold and
equal to the log return against its recorded predecessor, the categories
re-derived from their rule, the counts and the daily accounting adding up,
the daily test rows equal to the saved forecasts, no saved daily forecast
within 50 sessions of an in-schedule flag, and the all-rows rescoring of the
incomplete-candle check. On the final run, with the presentation folder:
4,128 checks, 0 problems. The unit tests (`pytest`, 43 tests) cover
targets, timing, sessions, splits, bootstrap, the ranking rule, the causal
Timestral calibration and the flag provenance (a flag whose predecessor is
an out-of-schedule bar dropped later).

**What they do not cover.** The raw 1-minute data and the candle building
(measured by `tools/sample_checks.py`; the verifier checks that tool's output
for coherence but does not re-read the 1-minute prices); the training (models were
not re-fitted); the bootstrap intervals (only their order is checked). The
final run used an uncommitted version of `multitarget.py`: its hash in
`run.json` (`a448847f…`) matches no committed version. The known later
changes are the majority-class fix (applied to the saved outputs by
`tools/mt_fix_majority.py`) and docstrings; the other source files used by
the run match the committed versions (hashes in `run.json`). A bit-identical
re-run is therefore not guaranteed.

Outputs of `multitarget.py` (`results/multitarget/runs/<id>/`):
`predictions_<freq>.parquet` (validation and test rows, every model's
forecast), `models.csv`, `dataset_info.json`, `console.log`, `run.json`.
Outputs of `multitarget_report.py`: `results_return.csv`,
`results_direction.csv`, `results_volatility.csv` (every model × timeframe ×
split, plus the Timestral-sample rows), `data_profile.csv`,
`selection_validation_ranking.csv`, `comparison_by_task.csv`,
`timeframe_map.csv`, `selected_five.csv` (the earlier five-configuration
selection, source of the charts), `configurations_tested.csv`,
`timestral_calibration.json`, `chart_cfg<id>_*.png`, `deck_data.json`.
Tools: `sample_checks.json`, `timestral_same_sample.csv/.json`.

## Future experiments (proposed, not run; by benefit and cost)

1. **Daily sample fix** — high benefit, low cost (daily data set is small):
   for daily candles, let a flagged minute exclude a session only if the
   Yahoo comparison ("Sample checks" C) does not confirm its daily return
   (ideally extended to the daily range and to RV) (today:
   no session; test about 921 of 1,002 sessions instead of 642), keeping the
   old rule as the default for reproducibility; then re-run
   `multitarget.py --freqs session` as a new run, merge it with the intraday
   predictions of the saved run and compare with the saved daily results.
2. **Aligned Timestral targets** — medium benefit, low–medium cost: forecast
   open and close of t+1 (body direction) and the quantiles of the next
   candle's 1-minute returns, derive RV from Timestral's own path, and add
   Timestral-28M/123M/209M (weights already cached) on the same sample.
3. **Pre-registered confirmation** — high benefit, medium cost: freeze the
   models and the ranking rule now and score them once on data after
   2025-06-02 (the reserved hold-out), turning the exploratory result into a
   confirmatory one.
4. **Economic evaluation of volatility forecasts** — medium benefit, medium
   cost: e.g. volatility-targeted position sizing with realistic costs,
   reported separately from the statistical results.
5. **Incomplete-candle policy** — benefit unknown beyond the test rescoring
   (< 0.4 points there; training and features not evaluated), medium
   cost: use `n_minutes` to drop or flag candles in dataset-gap sessions, then
   a full re-run.
6. **Ratio-adjusted prices** — benefit unknown (the effect on relative
   performance was not measured; absolute return and volatility levels and
   the training history are scaled),
   medium–high cost: rebuild the continuous series with multiplicative
   adjustment (or compute returns within each contract), recompute the data
   quality flags, then a full re-run compared with the saved one.

## Repository map

| Path | Content |
|---|---|
| `multitarget.py` | builds the data for each timeframe, trains every model, runs Timestral, saves the forecasts |
| `multitarget_report.py` | tables, data profile, rankings, five-model tables, timeframe map, charts, `deck_data.json` (reads saved forecasts only) |
| `verify_multitarget.py` | independent recomputation of the reported numbers |
| `config.py`, `data.py`, `timeframes.py`, `quality.py`, `features.py`, `evaluation.py`, `runlog.py` | data loading, CME sessions and candles, data checks, features and splits, run folders |
| `study.py`, `experiment.py`, `backtest.py`, `strategy.py`, `regimes.py`, `plots.py` | shared pipeline code from the earlier direction study; `multitarget.py` uses `study.build` (data set) and `experiment.load_minutes` / `bars_for`; their trading functions are not used here |
| `tools/sample_checks.py`, `tools/timestral_same_sample.py` | sample checks (appendix A3) and Timestral on the same sample (appendix A4) |
| `tools/mt_fix_majority.py`, `tools/timestral_direction.py` | majority-class fix without refitting; original zero-shot Timestral direction tool |
| `tests/` | unit tests (`python -m pytest -q`) |
| `presentation/` | deck generator, deck data, charts, main deck and appendix (`.pptx`, `.pdf`) |
| `reports/multitarget_final/` | final tables, charts and check outputs |

The earlier direction-and-trading study (`study.py` run, `report.py`,
`verify.py`, the NQ decks, IBKR download tools) is on the Git branch
`archive/nq-direction-study`.

## References

* Andersen, T. G., & Bollerslev, T. (1998). Answering the skeptics: Yes,
  standard volatility models do provide accurate forecasts. *International
  Economic Review*, 39(4), 885–905.
* Corsi, F. (2009). A simple approximate long-memory model of realized
  volatility. *Journal of Financial Econometrics*, 7(2), 174–196.
* Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary
  observations. *Annals of Statistics*, 17(3), 1217–1241.
* Patton, A. J. (2011). Volatility forecast comparison using imperfect
  volatility proxies. *Journal of Econometrics*, 160(1), 246–256.
* Mistral AI (2026). Timestral: time series foundation models. Announcement
  https://mistral.ai/news/timestral; code https://github.com/mistralai/timestral
  (commit a534bdbd); weights https://huggingface.co/mistralai/Timestral-5M.
