"""
Offline backtest of timestamped out-of-sample direction signals (no orders).

Rules, fixed before any evaluation:
  * decision at `decision_at` (bar close + inference latency), using only that
    bar and earlier ones;
  * entry at `entry_price`: the first price after the decision. With bar
    data this is the next bar's OPEN, a proxy: with latency the true first
    fill may come later, so slippage is charged on top. Never the signal
    bar's close (it was an input) and never an unavailable future price;
  * exit at the next bar's close; every signal is an independent one-bar
    round trip of one contract (conservative: no netting of consecutive
    signals, so turnover is 2 contracts per trade);
  * no trade when |P(up) - 0.5| < threshold (threshold chosen on validation
    only); a future flat outcome is NOT an entry filter: it is a trade with
    zero gross P&L and full costs;
  * rows without an eligible next bar (session end, outage, partial bar,
    quarantined data) are not traded, so no position crosses a session
    close. For the continuous back-adjusted series the point difference
    within one session equals the traded contract's (additive adjustment);
    session-frequency trades crossing an approximate roll date are excluded.

Costs per round trip (all configurable, all assumptions unless the user
supplies account costs): 2 x commission+fees per side, one full bid/ask
spread (fills at trade prices treated as mid; half spread per side), and
2 x slippage. Trade prices do not tell whether a print was at the bid or the
ask, and same-second historical ticks do not establish sub-second fills;
results are therefore indicative, with cost stress scenarios.
"""

import numpy as np
import pandas as pd

NQ_MULTIPLIER = 20.0        # USD per index point
NQ_TICK = 0.25              # index points

# Cost scenarios (USD per side commission+exchange fees; spread and slippage in ticks).
# These are ASSUMPTIONS, not quotes from an account.
COST_SCENARIOS = {
    "zero_cost": dict(commission_per_side=0.0, spread_ticks=0.0, slippage_ticks_per_side=0.0),
    "low": dict(commission_per_side=1.0, spread_ticks=1.0, slippage_ticks_per_side=0.0),
    "base": dict(commission_per_side=2.5, spread_ticks=1.0, slippage_ticks_per_side=0.25),
    "high": dict(commission_per_side=4.0, spread_ticks=1.0, slippage_ticks_per_side=1.0),
}


def round_trip_cost_usd(commission_per_side, spread_ticks, slippage_ticks_per_side,
                        multiplier=NQ_MULTIPLIER, tick=NQ_TICK):
    return (2 * commission_per_side + spread_ticks * tick * multiplier
            + 2 * slippage_ticks_per_side * tick * multiplier)


def positions(proba_up, threshold=0.0, pred=None):
    """+1 long, -1 short, 0 flat. With probabilities: trade only if |p-0.5| >= threshold."""
    if proba_up is None:
        return np.asarray(pred, dtype=int)
    p = np.asarray(proba_up, dtype=float)
    pos = np.where(p >= 0.5, 1, -1)
    return np.where(np.abs(p - 0.5) >= threshold, pos, 0)


def run_backtest(meta, pos, costs, multiplier=NQ_MULTIPLIER, tick=NQ_TICK, exclude=None):
    """Per-trade and per-session P&L in USD for one contract per signal."""
    pos = np.asarray(pos, dtype=int).copy()
    if exclude is not None:
        pos[np.asarray(exclude, bool)] = 0
    entry, exit_ = meta["entry_price"].to_numpy(float), meta["exit_price"].to_numpy(float)
    if np.isnan(entry[pos != 0]).any() or np.isnan(exit_[pos != 0]).any():
        raise ValueError("a traded row has no entry/exit price")
    gross = np.where(pos != 0, pos * (exit_ - entry) * multiplier, 0.0)
    cost = np.where(pos != 0, round_trip_cost_usd(**costs, multiplier=multiplier, tick=tick), 0.0)
    trades = pd.DataFrame({"session_id": meta["session_id"].to_numpy(), "pos": pos,
                           "gross_usd": gross, "cost_usd": cost, "net_usd": gross - cost})
    return trades


def summarize(trades, capital_usd=None):
    """Gross/net P&L, turnover, trade stats, drawdown, daily statistics."""
    t = trades[trades["pos"] != 0]
    daily = trades.groupby("session_id")["net_usd"].sum()
    eq = daily.cumsum()
    dd = (eq - eq.cummax()).min() if len(eq) else 0.0
    wins, losses = t.loc[t["net_usd"] > 0, "net_usd"], t.loc[t["net_usd"] <= 0, "net_usd"]
    out = {
        "n_trades": int(len(t)), "exposure": float((trades["pos"] != 0).mean()),
        "turnover_contracts": int(2 * len(t)),
        "gross_usd": float(t["gross_usd"].sum()), "cost_usd": float(t["cost_usd"].sum()),
        "net_usd": float(t["net_usd"].sum()),
        "expectancy_net_usd": float(t["net_usd"].mean()) if len(t) else 0.0,
        "expectancy_gross_usd": float(t["gross_usd"].mean()) if len(t) else 0.0,
        "win_rate_net": float((t["net_usd"] > 0).mean()) if len(t) else np.nan,
        "mean_win_usd": float(wins.mean()) if len(wins) else np.nan,
        "mean_loss_usd": float(losses.mean()) if len(losses) else np.nan,
        "max_drawdown_usd": float(dd), "n_sessions": int(len(daily)),
        "mean_daily_net_usd": float(daily.mean()) if len(daily) else 0.0,
        "std_daily_net_usd": float(daily.std()) if len(daily) > 1 else np.nan,
    }
    if capital_usd:  # percentages only with a stated capital assumption
        out["capital_usd_assumed"] = capital_usd
        out["net_return_pct_on_capital"] = 100 * out["net_usd"] / capital_usd
    return out


def choose_threshold(meta_val, proba_val, costs, grid=(0.0, 0.01, 0.02, 0.05, 0.1)):
    """No-trade threshold maximising validation net P&L (validation only)."""
    best, best_net = 0.0, -np.inf
    for th in grid:
        net = run_backtest(meta_val, positions(proba_val, th), costs)["net_usd"].sum()
        if net > best_net:
            best, best_net = th, net
    return best


def approx_roll_sessions(session_ids):
    """Approximate CME equity-index roll days: the Thursday 8 days before the
    third Friday of Mar/Jun/Sep/Dec (the continuous series' exact roll dates
    are not in the data, so this is an estimate)."""
    s = pd.DatetimeIndex(pd.unique(session_ids))
    years = range(s.year.min(), s.year.max() + 1) if len(s) else []
    rolls = []
    for y in years:
        for m in (3, 6, 9, 12):
            first = pd.Timestamp(year=y, month=m, day=1)
            third_fri = first + pd.Timedelta(days=(4 - first.dayofweek) % 7 + 14)
            rolls.append(third_fri - pd.Timedelta(days=8))
    return pd.DatetimeIndex(rolls)
