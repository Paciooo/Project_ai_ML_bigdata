# Can Machine Learning Trade the Next Nasdaq Candle?

University project (Waseda University, *AI, Machine Learning and Big Data for
Finance and Management*, option 1: data analysis). The project asks whether
machine-learning models can predict the direction of the next Nasdaq futures
candle, and whether that prediction can be traded profitably after
commissions.

**Final answer: no deployable strategy was found.** The models predict
slightly better than "always up" on 1 and 5-minute candles (best: +0.62
percentage points), but no long/short, long-only or short-only strategy
passes the out-of-sample, cost and walk-forward tests fixed in advance. On
daily candles the models mostly collapse to "always long"; the profits there
come from the Nasdaq's upward drift, not from prediction.

Final deck: [presentation/NQ_final_report.pptx](presentation/NQ_final_report.pptx)
([PDF](presentation/NQ_final_report.pdf)). Result tables:
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
  with zero delay is optimistic, especially at 1 minute. Sensitivity runs use
  $0, $2.50, $12.50 and $23 per round trip ($12.50 and $23 add one tick of
  spread and ¼ or 1 tick of slippage per side).
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
  strategy direction and thresholds chosen on validation only.
* Test scored once. Yearly expanding walk-forward refits for the candidate
  strategies. 20-session block-bootstrap intervals for accuracy differences,
  5-session blocks for daily P&L.
* Selection rule fixed before reading the results (see `report.py`): the
  strategy with the highest validation net P&L is the candidate; it is
  *deployable* only if test net P&L > 0 at $5 **and** $12.50, the 95% interval
  of daily net P&L is above zero, and at least 3 of 4 walk-forward years are
  profitable.
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

Best predictive model: MLP 128-64-32 on 5-minute candles, 51.45% on non-doji
candles, +0.62 pp over "always up" (95% CI +0.32 to +0.92). From 15 minutes
on, no model is reliably better than "always up". Deeper networks gave no
consistent gain; early stopping halted every network after 5–17 epochs.

Trading (test period, $5 per round trip):

* Every unfiltered intraday strategy loses: costs exceed the gross edge
  (5-minute reversal rule: +$391k gross, −$975k net over 273,340 trades,
  i.e. $1.43 gross per trade against $5 of commission).
* Validation-chosen candidates (filtered Long & Short): 1 min +$30.7k,
  5 min −$57.4k, 15 min −$2.4k, 1 hour −$7.6k, 1 day −$25.9k. None has a
  daily-P&L interval above zero. Only the daily candidate is profitable in
  3+ walk-forward years (4 of 4, +$201.5k), with yearly-refitted models that
  are mostly long in a rising market.
* Pre-specified choice: MLP 256-128-64-32 with dropout and batch norm, daily
  candles, Long & Short, |P(up) − 0.5| ≥ 0.01: +$164.5k on validation,
  **−$25.9k on test** (575 trades, win rate 50.6%, −$40.09 gross and
  −$45.09 net per trade, profit factor 0.97, max drawdown −$102.4k).
* Long vs short (same model, no filter): long-only +$16.1k, short-only
  −$58.2k, Long & Short −$42.1k; the model calls "up" 81% of the days.
  Simply staying long every day earned +$73.0k.
* Profit factor = sum of winning trades ÷ |sum of losing trades|, both after
  commissions. Gross per trade = average P&L per trade before commission.

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
python report.py results\nq_study\runs\<run id> presentation\deck_data.json
python verify.py results\nq_study\runs\<run id> presentation\deck_data.json
```

```powershell
cd presentation
npm install
node build_deck.js NQ_final_report.pptx                 # deck from deck_data.json (Node.js 18+)
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
                IBKR tick / bar downloaders (optional, not used in the study)
tests/          regression tests (timing, splits, execution, strategies)
```

Earlier project phases (VIX experiment, coin-flip Monte Carlo, Timestral
foundation models, runs A-C with close-to-close targets) are preserved in
the Git history up to commit `fd3473c`.
