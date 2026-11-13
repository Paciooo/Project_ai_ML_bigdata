# Can Machine Learning Trade the Next Nasdaq Candle?

University project (Waseda University, *AI, Machine Learning and Big Data for
Finance and Management*, option 1: data analysis). The project asks whether
machine-learning models can predict the direction of the next Nasdaq futures
candle, and whether that prediction can be traded profitably after
commissions.

**Final answer: no deployable strategy was demonstrated under the criteria
fixed in advance; none of the five validation-selected candidates passes
every robustness criterion.** The validation-selected models predict slightly better
than "always up" on 1 and 5-minute candles (best of them: +0.62 percentage
points, exploratory interval), but none of the five candidates selected on
validation (one per candle length) passes the out-of-sample, cost and
walk-forward tests. On daily candles most models call "up" every day and so
reproduce "always long" exactly.

Final deck (cover + 8 slides): [presentation/NQ_final_report.pptx](presentation/NQ_final_report.pptx)
([PDF](presentation/NQ_final_report.pdf)); appendix as a separate document:
[presentation/NQ_final_report_appendix.pptx](presentation/NQ_final_report_appendix.pptx)
([PDF](presentation/NQ_final_report_appendix.pdf)). Result tables:
[reports/nq_study_final/](reports/nq_study_final/).

## Instrument and data

| Item | Value |
|---|---|
| Instrument | CME E-mini Nasdaq-100 futures (**NQ**), $20 per index point, tick 0.25 = $5. Not MNQ ($2 per point), not QQQ, not the cash index |
| Data | TradeStation continuous NQ, 1-minute OHLCV bars, Jan 2006 – May 2025 (6,193,869 bars), additively back-adjusted (within-session point moves equal the traded contract's) |
| Candles studied | 1, 5, 15 and 60 minutes and daily (one CME session, 17:00–16:00 CT), built from the 1-minute bars inside CME sessions |
| Notional | $20 × index level, about $210k–$445k per contract in 2021–2025; P&L = points moved × $20 |
| Location | not in Git (licensed). Default `..\futures-data`, or set `FUTURES_DATA_DIR` |

## Prediction task and trades

* **Decision:** at the close of candle *t*, using candles up to *t* only.
* **Target:** body of candle *t+1*: bullish if close(t+1) > open(t+1),
  bearish if lower. Dojis (equal prices) stay in the data and count as misses.
* **Trade:** long (bullish call) or short (bearish call) one NQ contract at the
  **open** of *t+1*, exit at its **close**. P&L = ±(close − open) × $20.
  The traded move is exactly the predicted target (checked by `verify.py`).
* **Costs:** $5 per round trip ($2.50 to open + $2.50 to close), zero
  slippage, 100% fills. This is an **idealised** limit-order assumption:
  real limit orders are not guaranteed to fill, and filling at the next open
  with zero latency is optimistic, especially at 1 minute. Sensitivity runs
  use $0, $2.50, $12.50 and $23 per round trip; $12.50 and $23 are assumed
  scenarios (one tick of spread plus ¼ or 1 tick of slippage per side), not
  measured execution costs.
* **Latency:** the final run uses zero latency. With `--latency` > 0 the entry
  is the first clean minute after the delay, so the traded return is
  entry → close, not the predicted body; `study.py` flags this
  (`traded_return`, `n_delayed_entry`, `n_not_evaluable` in
  `dataset_info.json`, a warning in the log) instead of presenting it as the
  candle body. If a sample has no observable entry before the exit at all,
  its strategies get `economic_status = not_evaluable_no_entry_before_exit`
  with N/A P&L and are excluded from the selection (a strategy that has
  entries available but chooses not to trade stays evaluated with zero P&L).
  With zero latency, as in the final run, every row is executable.
* **Strategies:** Long & Short, Long only, Short only, each with and without
  a confidence filter |P(up) − 0.5| ≥ threshold chosen on validation data.

## Models

Each family has the repository's original configuration and optimised
variants (8 original price features, plus 11 new causal features: candle
body/range/wicks, volume, realised volatility, RSI, moving-average gaps,
time of day):

| Family | Original | Variants |
|---|---|---|
| Logistic regression | C = 1 | C = 1, C = 0.01 |
| Random forest | 200 trees, depth 6 | 200 trees, depth 10 |
| XGBoost | 200 trees, depth 3 | depth 3 / 5 with early stopping |
| Neural network (PyTorch MLP) | 64-32-16 | 64-32, 128-64-32, 256-128-64-32, dropout + batch norm, lower learning rate |

Plus rules: always long, always short, reversal of the last move.

## Validation protocol

* Chronological splits by trading session: train 2006 – Aug 2017,
  validation Aug 2017 – Jul 2021, test Jul 2021 – May 2025; labels that cross
  a boundary are purged. No shuffling across time.
* Scaling fitted on training data only; hyper-parameters, early stopping,
  strategy direction and thresholds chosen on validation only. A threshold
  needs at least 200 executable validation trades, at the run's cost per
  round trip; if none qualifies the filtered strategy is reported as not
  evaluable (no silent fallback to "no filter").
* Test scored once. Yearly expanding walk-forward refits for the candidate
  strategies. 20-session block-bootstrap intervals for accuracy differences,
  5-session blocks for daily P&L.
* Selection rule fixed before reading the results (see `report.py`): the
  strategy with the highest validation net P&L is the candidate; it is
  *deployable* only if test net P&L > 0 at $5 **and** $12.50, the 95% interval
  of daily net P&L is above zero, and at least 3 of 4 walk-forward years are
  profitable (a not-evaluable year does not count as profitable).
* Caveat: the 2021–2025 test period was examined with the original models in
  earlier phases of the project, so the results are exploratory. Data after
  May 2025 was not available.
* Daily candles: early-close sessions are not used as targets and the same
  50-candle quarantine after bad prints applies (50 days), so the daily
  sample has 3,457 sessions and its test ends on 2025-01-30 (642 sessions).
  Networks use batches of 128 when fewer than 20,000 rows are available.

## Results

Intraday run `20261111T070520Z_949daa_final` (commit `ff445f3`) plus daily
run `20261111T103357Z_02bd49_daily_final` (commit `0970132`, same settings),
joined into one folder by `tools/merge_runs.py`.

Test accuracy on all candles, original → optimised:

| Model | 1 min | 5 min | 15 min | 1 hour | 1 day |
|---|---|---|---|---|---|
| Logistic regression | 48.63 → 48.67% | 50.42 → 50.43% | 50.73 → 50.77% | 51.09 → 50.50% | 51.71 → 51.71% |
| Random forest | 48.68 → 48.69% | 50.49 → 50.47% | 50.88 → 50.79% | 51.11 → 51.69% | 54.21 → 54.21% |
| XGBoost | 48.63 → 48.63% | 50.50 → 50.46% | 50.67 → 50.96% | 50.77 → 51.76% | 51.56 → 51.56% |
| Neural network | 48.66 → 48.71% | 50.33 → **50.52%** | 50.94 → 50.85% | 51.57 → 51.74% | 54.21 → 54.21% |

Daily 54.21% is the share of up days: those models call "up" every day.

Best of the validation-selected models (one per family and candle length):
MLP 128-64-32 on 5-minute candles, 51.45% on non-doji candles, +0.62 pp over
"always up" (95% CI +0.32 to +0.92). Across all 15 configurations, MLP
256-128-64-32 on 5 minutes shows +0.73 pp on the test, descriptively; it was
not selected on validation and is not promoted. The intervals are
exploratory and not adjusted for the search over models. From 15 minutes on,
no validation-selected model is reliably better than "always up". Deeper
networks gave no consistent gain; early stopping halted every network after
5–17 epochs.

Trading (test period, $5 per round trip):

* The unfiltered Long & Short versions of the five validation-selected
  candidates all lose after costs (1 min −$7.09M, 5 min −$1.25M, 15 min
  −$498.8k, 1 hour −$74.9k, 1 day −$42.1k); intraday their gross edge is
  below the $5 commission (e.g. 5-minute reversal rule: +$391k gross, −$975k
  net over 273,340 trades, $1.43 gross per trade). Not every configuration
  loses: descriptively, 1-hour unfiltered short-only earns +$29.4k
  (original logistic), +$25.7k (random forest d10) and +$3.7k (original MLP),
  and the filtered 1-minute candidate +$30.7k. These are results of a search
  over many configurations, not validated strategies.
* Validation-chosen candidates (filtered Long & Short): 1 min +$30.7k,
  5 min −$57.4k, 15 min −$2.4k, 1 hour −$7.6k, 1 day −$25.9k. None of these
  five has a daily-P&L interval above zero, and none is profitable in 3 of 4
  walk-forward years. The walk-forward covers these five candidates only,
  not all 407 configurations.
* Descriptively, 3 of the 407 test configurations have a daily-P&L interval
  above zero (1-minute MLP 256-128-64-32 short-only filtered, +$28.5k, lower
  bound about $6.68 a day; 5-minute original MLP Long & Short and long-only
  filtered, the same 345 trades, +$34.1k, lower bound about $3.69). They come
  from a search over many configurations and are not independent
  confirmations.
* Pre-specified choice: MLP 256-128-64-32 with dropout and batch norm, daily
  candles, Long & Short, |P(up) − 0.5| ≥ 0.01: +$164.5k on validation,
  **−$25.9k on test** (575 trades, win rate 50.6%, −$40.09 gross and
  −$45.09 net per trade, profit factor 0.97, max drawdown −$102.4k).
* Long vs short (same model, no filter): long-only +$16.1k, short-only
  −$58.2k, Long & Short −$42.1k; the model calls "up" 81% of the days.
  Simply staying long every day earned +$73.0k; the daily models that call
  "up" every day (random forest d10, original MLP) earn exactly that.
* Daily walk-forward (appendix A7). Code up to commit `35be806` reported
  4 of 4 positive years, +$201.5k, but in years 2 and 3 the validation block
  had only 185 and 70 sessions, fewer than the 200 trades a threshold needs,
  and the code silently traded them unfiltered. Corrected: 2 of 4 years
  evaluable, both positive, +$118.7k; always long on the same sessions
  −$12.5k. Over the four published years the gross P&L was +$204.5k:
  +$67.1k "direction mix" (what a random policy with the same numbers of
  longs and shorts would gross on the same sessions) and +$137.4k residual;
  always long on those sessions earned +$73.0k. This neither proves an edge
  (one path, one configuration chosen among 407, idealised fills; the same
  model fitted once loses −$25.9k on the test) nor supports "it only rode a
  rising market": in 2021-22 always long lost −$29.7k while the refitted
  model earned +$86.7k.
* Profit factor = sum of winning trades ÷ |sum of losing trades|, both after
  commissions. Gross per trade = average P&L per trade before commission.
  Maximum drawdown is measured on session-end equity (intra-session dips
  are not included).
* Accuracy alone does not set profitability: with symmetric moves of size W
  the expected P&L per trade is (2p − 1)·W − $5, so 51% earns 0.02·W − $5.

## Reproduce (Windows PowerShell, from the project folder)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:FUTURES_DATA_DIR = "C:\path\to\futures-data"
python -m pytest tests -q
```

```powershell
python study.py                                        # all models, strategies, walk-forward (~15 min, 16 CPU cores, no GPU)
# this deck: intraday run + daily-only run joined into one folder
#   python study.py --freqs 1min 5min 15min 1h ; python study.py --freqs session
#   python tools\merge_runs.py results\nq_study\runs\<new id> <intraday run> <daily run>
python tools\wf_threshold_status.py results\nq_study\runs\<run id>   # runs made before the threshold fix: required, report.py stops without it
python report.py results\nq_study\runs\<run id> presentation\deck_data.json
python verify.py results\nq_study\runs\<run id> presentation\deck_data.json
```

`verify.py` recomputes, from the saved test predictions and without the
strategy code, the test accuracy, trades, longs/shorts, gross and net P&L,
win rate, profit factor, session-end drawdown, equity curves and cost
sensitivity; it checks probabilities and P&L for non-finite values, entry
timing (when the run saved it) and the deck data. It does not rebuild candles
or features from the raw data, refit models, re-run the bootstrap or check
the slide text. On the final run: 8,858 checks, 0 problems (the final run
predates the saved entry times, so entry timing rests on the tests).

```powershell
cd presentation
npm install
node build_deck.js NQ_final_report.pptx                 # deck from deck_data.json (Node.js 18+)
node build_deck.js NQ_final_report_appendix.pptx        # appendix, separate document
```

`python study.py --freqs 15min --years 2016 2019 --quick` runs a fast
pipeline check. Runs are written to `results/` (not in Git); the small
summaries of the final run are copied to `reports/nq_study_final/`.
`report.py` also writes `equity_curves.png` into the run folder.

## Repository map

```text
study.py        next-candle study: features, models, strategies, walk-forward
strategy.py     positions, P&L with the NQ multiplier, trading metrics
report.py       pre-specified selection, summaries, chart, deck data
verify.py       independent recomputation of every reported test metric
presentation/   deck builder (pptxgenjs), deck data, final .pptx and .pdf
reports/        small result snapshot of the final run
experiment.py, features.py, timeframes.py, quality.py, backtest.py,
evaluation.py, data.py, config.py, runlog.py
                shared data pipeline: sessions, candles, causal cleaning,
                purged splits, execution timing (main.py: earlier runs A-C)
tools/          merge_runs.py (joins runs over different candle lengths);
                wf_threshold_status.py (walk-forward threshold status of older runs);
                IBKR tick / bar downloaders (optional, not used in the study)
tests/          regression tests (timing, splits, execution, strategies)
```

Earlier project phases (VIX experiment, coin-flip Monte Carlo, Timestral
foundation models, runs A-C with close-to-close targets) are preserved in
the Git history up to commit `fd3473c`.
