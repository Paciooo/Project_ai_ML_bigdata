"""
Next-candle trading strategies for NQ futures and their performance metrics.

Instrument: CME E-mini Nasdaq-100 futures (NQ), $20 per index point, tick
0.25 points = $5. Prices come from the TradeStation continuous, additively
back-adjusted series: within one session the point difference equals the
traded contract's, so one-candle round trips are measured correctly.

Trade per signal (one contract): enter at the OPEN of candle t+1, exit at the
CLOSE of candle t+1. Gross P&L = position x (close - open) x $20. Costs are a
fixed amount per completed round trip (default $5.00: $2.50 to open + $2.50 to
close) with zero slippage: an idealised limit-order fill assumption, NOT a
realistic simulation of limit-order fills (which are not guaranteed).

Strategies (from P(up) of a model):
  long_short  p >= 0.5 -> long, else short
  long_only   p >= 0.5 -> long, else flat
  short_only  p <  0.5 -> short, else flat
  A confidence threshold t trades only when |p - 0.5| >= t (chosen on
  validation data only).
"""

import numpy as np
import pandas as pd

import evaluation

NQ_MULTIPLIER = 20.0
NQ_TICK = 0.25
DEFAULT_RT_COST = 5.0            # $2.50 open + $2.50 close, per contract
CAPITAL_USD = 100_000.0          # only to express drawdown in percent
STRATEGIES = ("long_short", "long_only", "short_only")


def positions(p_up, strategy, threshold=0.0):
    """+1 long / -1 short / 0 flat per row. Rejects probabilities that are not
    finite or outside [0, 1] and thresholds outside [0, 0.5]."""
    p = np.asarray(p_up, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must be finite and within [0, 1]")
    th = float(threshold)
    if not np.isfinite(th) or th < 0 or th > 0.5:
        raise ValueError(f"threshold must be finite and within [0, 0.5], got {threshold!r}")
    side = np.where(p >= 0.5, 1, -1)
    active = np.abs(p - 0.5) >= th
    if strategy == "long_only":
        active &= side > 0
    elif strategy == "short_only":
        active &= side < 0
    elif strategy != "long_short":
        raise ValueError(f"unknown strategy {strategy!r}")
    return np.where(active, side, 0)


def trade_pnl(pos, move_points, rt_cost=DEFAULT_RT_COST, multiplier=NQ_MULTIPLIER):
    """Per-row gross and net P&L (0 where flat). move_points = close(t+1) - open(t+1)."""
    pos = np.asarray(pos, dtype=int)
    gross = pos * np.asarray(move_points, dtype=float) * multiplier
    cost = np.where(pos != 0, rt_cost, 0.0)
    return gross, gross - cost


EVALUATED = "evaluated"
NOT_EVALUABLE = "not_evaluable_no_entry_before_exit"


def summarize(pos, move_points, sessions, rt_cost=DEFAULT_RT_COST, capital=CAPITAL_USD,
              bootstrap=True, executable=None):
    """Trading metrics for one strategy on one sample (rows in time order).

    `executable` = rows with an observable entry before the exit. If there is
    none, the sample has no tradable opportunity: economic metrics are NaN and
    economic_status = NOT_EVALUABLE. A strategy that has opportunities but
    chooses not to trade is EVALUATED with zero trades and zero P&L.
    """
    pos = np.asarray(pos, dtype=int)
    if executable is not None:
        ex = np.asarray(executable, bool)
        if ex.shape != pos.shape:
            raise ValueError("executable mask must align with positions")
        if (pos[~ex] != 0).any():
            raise ValueError("positions on rows without an executable entry")
        if not ex.any():
            return {"economic_status": NOT_EVALUABLE, "n_executable": 0, "n_trades": 0, "n_long": 0, "n_short": 0,
                    **{k: np.nan for k in ("exposure", "gross_usd", "costs_usd", "net_usd", "win_rate", "avg_win_usd",
                                           "avg_loss_usd", "avg_net_per_trade_usd", "profit_factor", "max_drawdown_usd",
                                           "max_drawdown_pct_capital", "sharpe_daily_ann", "sortino_daily_ann",
                                           "long_net_usd", "short_net_usd", "long_win_rate", "short_win_rate",
                                           "top1pct_trades_share_of_gross", "gross_from_direction_mix_usd",
                                           "gross_from_timing_usd")},
                    "n_sessions": int(len(pd.unique(np.asarray(sessions))))}
    gross, net = trade_pnl(pos, move_points, rt_cost)
    traded = pos != 0
    t_net, t_gross = net[traded], gross[traded]
    daily = pd.Series(net).groupby(np.asarray(sessions)).sum().sort_index()
    eq = np.r_[0.0, daily.cumsum().to_numpy()]
    dd = float((eq - np.maximum.accumulate(eq)).min())
    wins, losses = t_net[t_net > 0], t_net[t_net <= 0]
    std = daily.std()
    downside = np.sqrt(np.mean(np.minimum(daily.to_numpy(), 0) ** 2)) if len(daily) else np.nan
    longs, shorts = pos > 0, pos < 0
    out = {
        "economic_status": EVALUATED,
        "n_executable": int(len(pos) if executable is None else np.asarray(executable, bool).sum()),
        "n_trades": int(traded.sum()), "n_long": int(longs.sum()), "n_short": int(shorts.sum()),
        "exposure": float(traded.mean()) if len(pos) else 0.0,
        "gross_usd": float(t_gross.sum()), "costs_usd": float(rt_cost * traded.sum()),
        "net_usd": float(t_net.sum()),
        "win_rate": float((t_net > 0).mean()) if traded.any() else np.nan,
        "avg_win_usd": float(wins.mean()) if len(wins) else np.nan,
        "avg_loss_usd": float(losses.mean()) if len(losses) else np.nan,
        "avg_net_per_trade_usd": float(t_net.mean()) if traded.any() else np.nan,
        "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else np.nan,
        "max_drawdown_usd": dd,                  # on session-end (daily) equity, from 0
        "max_drawdown_pct_capital": 100 * dd / capital,
        "n_sessions": int(len(daily)),
        "sharpe_daily_ann": float(daily.mean() / std * np.sqrt(252)) if std and std > 0 else np.nan,
        "sortino_daily_ann": float(daily.mean() / downside * np.sqrt(252)) if downside and downside > 0 else np.nan,
        "long_net_usd": float(net[longs].sum()), "short_net_usd": float(net[shorts].sum()),
        "long_win_rate": float((net[longs] > 0).mean()) if longs.any() else np.nan,
        "short_win_rate": float((net[shorts] > 0).mean()) if shorts.any() else np.nan,
        # concentration: share of net P&L from the best 1% of trades
        "top1pct_trades_share_of_gross": (float(np.sort(t_gross)[::-1][:max(1, len(t_gross) // 100)].sum()
                                          / t_gross.sum()) if t_gross.sum() > 0 else np.nan),
    }
    # market-direction vs timing decomposition of GROSS P&L: a random policy
    # with the same number of longs/shorts on the same rows earns, in
    # expectation, (n_long - n_short) * mean move * $20.
    mv = np.asarray(move_points, dtype=float)
    out["gross_from_direction_mix_usd"] = float((longs.sum() - shorts.sum()) * mv.mean() * NQ_MULTIPLIER)
    out["gross_from_timing_usd"] = out["gross_usd"] - out["gross_from_direction_mix_usd"]
    if bootstrap and len(daily) > 10:
        est, lo, hi = evaluation.block_bootstrap(net, np.asarray(sessions), block_len=5, stat="sum_per_group")
        out["mean_daily_net_usd"], out["daily_net_ci_lo"], out["daily_net_ci_hi"] = est, lo, hi
    return out


def daily_equity(pos, move_points, sessions, rt_cost=DEFAULT_RT_COST):
    _, net = trade_pnl(pos, move_points, rt_cost)
    return pd.Series(net).groupby(np.asarray(sessions)).sum().sort_index().cumsum()


THRESHOLD_OK = "ok"
THRESHOLD_NONE = "no_threshold_reaches_min_trades"


def choose_threshold(p_up, move_points, strategy, grid, rt_cost=DEFAULT_RT_COST, min_trades=200,
                     executable=None):
    """Validation-only threshold maximising net P&L (ties -> smaller threshold).

    Only executable entries count, for both the P&L and `min_trades`
    (`executable` = boolean mask, None = every row executable). Returns
    (threshold, THRESHOLD_OK), or (None, THRESHOLD_NONE) when no threshold in
    the grid reaches `min_trades`; there is no silent fallback to zero.
    """
    p = np.asarray(p_up, dtype=float)
    ex = np.ones(len(p), bool) if executable is None else np.asarray(executable, bool)
    if ex.shape != p.shape:
        raise ValueError("executable mask must align with probabilities")
    if not len(grid):
        raise ValueError("empty threshold grid")
    best, best_net = None, -np.inf
    for th in grid:
        pos = positions(p, strategy, th) * ex
        if (pos != 0).sum() < min_trades:
            continue
        net = trade_pnl(pos, move_points, rt_cost)[1].sum()
        if net > best_net + 1e-9:
            best, best_net = th, net
    return (best, THRESHOLD_OK) if best is not None else (None, THRESHOLD_NONE)
