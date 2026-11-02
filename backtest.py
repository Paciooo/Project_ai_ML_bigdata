"""
Offline backtest of timestamped out-of-sample direction signals (no orders).

Rules, fixed before any evaluation:
  * decision at `decision_at` (bar close + inference latency), using only that
    bar and earlier ones;
  * entry at the first observed ONE-MINUTE bar open at or after the decision
    and inside the label interval. Exit must be strictly later than entry.
    Missing, suspect or too-late entries are not traded. At one-minute
    prediction frequency a positive latency normally leaves no observable
    entry before the outcome closes: economic performance is not evaluable.
    Opens are still bar-price proxies, not guaranteed executable bid/ask fills;
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
    threshold = np.asarray(threshold, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must be finite and between zero and one")
    if not np.isfinite(threshold).all() or (threshold < 0).any():
        raise ValueError("thresholds must be finite and nonnegative")
    if threshold.ndim and threshold.shape != p.shape:
        raise ValueError("per-row thresholds must align with probabilities")
    pos = np.where(p >= 0.5, 1, -1)
    return np.where(np.abs(p - 0.5) >= threshold, pos, 0)


def attach_execution(meta, minutes):
    """Attach conservative observed-minute entries without changing labels/predictions.

    Select the first observed minute at or after both the decision and the
    label start. Missing minute timestamps advance the search to the next
    observed minute. If that selected candidate has an invalid/suspect price,
    is partial, belongs to the wrong session or falls outside the label
    interval, reject it without searching for a later candidate. Its entry
    must precede label end and its minute must finish no later than label end.
    """
    required = {"decision_at", "label_start", "label_end"}
    if not required <= set(meta):
        raise ValueError(f"missing timing columns: {required - set(meta)}")
    cols = [c for c in ("bar_start", "bar_end", "open", "session_id", "suspect", "is_partial") if c in minutes]
    m = minutes[cols].sort_values("bar_start", kind="stable").reset_index(drop=True)
    if m.empty or m.bar_start.duplicated().any():
        raise ValueError("execution minutes must be nonempty and have unique starts")
    out = meta.copy()
    start = pd.DatetimeIndex(m.bar_start).as_unit("ns")
    decision = pd.DatetimeIndex(out.decision_at).as_unit("ns")
    label_start = pd.DatetimeIndex(out.label_start).as_unit("ns")
    label_end = pd.DatetimeIndex(out.label_end).as_unit("ns")
    earliest = np.maximum(decision.asi8, label_start.asi8)
    ix = np.searchsorted(start.asi8, earliest, side="left")
    in_range = ix < len(m)
    safe = np.minimum(ix, len(m) - 1)
    picked = m.iloc[safe]
    entry = start[safe]
    minute_end = pd.DatetimeIndex(picked.bar_end).as_unit("ns")
    valid = in_range & (entry.asi8 >= earliest) & (entry.asi8 < label_end.asi8)
    valid &= minute_end.asi8 <= label_end.asi8
    price = picked.open.to_numpy(float)
    valid &= np.isfinite(price) & (price > 0)
    if "suspect" in picked:
        valid &= ~picked.suspect.to_numpy(bool)
    if "is_partial" in picked:
        valid &= ~picked.is_partial.to_numpy(bool)
    if "label_session_id" in out:
        valid &= picked.session_id.to_numpy() == out.label_session_id.to_numpy()
    out["entry_at"] = pd.Series(entry, index=out.index).where(valid)
    out["exit_at"] = out.label_end
    out["entry_price"] = np.where(valid, price, np.nan)
    out["execution_eligible"] = valid
    out["execution_reason"] = np.where(valid, "observed_minute_open_after_decision",
                                         "no_clean_minute_entry_before_label_end")
    return out


def frame_positions(preds, model):
    """Thresholds are data columns, never pandas attrs or a silent zero fallback."""
    if f"proba_{model}" not in preds:
        return positions(None, pred=preds[f"pred_{model}"])
    col = f"threshold_{model}"
    if col not in preds:
        raise ValueError(f"missing {col}; restore fold thresholds before scoring")
    return positions(preds[f"proba_{model}"], preds[col])


def run_backtest(meta, pos, costs, multiplier=NQ_MULTIPLIER, tick=NQ_TICK, exclude=None):
    """Per-trade and per-session P&L in USD for one contract per signal."""
    pos = np.asarray(pos, dtype=int).copy()
    if exclude is not None:
        pos[np.asarray(exclude, bool)] = 0
    required = {"decision_at", "entry_at", "exit_at", "execution_eligible"}
    if not required <= set(meta):
        raise ValueError("backtest requires explicit execution timing; attach execution first")
    eligible = meta.execution_eligible.to_numpy(bool)
    time_ok = ((meta.entry_at >= meta.decision_at) & (meta.entry_at < meta.exit_at)).to_numpy()
    if (eligible & ~time_ok).any():
        raise ValueError("an eligible entry precedes its decision or is not before exit")
    requested = pos.copy()
    pos[~eligible] = 0
    entry, exit_ = meta["entry_price"].to_numpy(float), meta["exit_price"].to_numpy(float)
    if np.isnan(entry[pos != 0]).any() or np.isnan(exit_[pos != 0]).any():
        raise ValueError("a traded row has no entry/exit price")
    gross = np.where(pos != 0, pos * (exit_ - entry) * multiplier, 0.0)
    cost = np.where(pos != 0, round_trip_cost_usd(**costs, multiplier=multiplier, tick=tick), 0.0)
    if not np.isfinite(entry[pos != 0]).all() or not np.isfinite(exit_[pos != 0]).all():
        raise ValueError("nonfinite traded price")
    pnl_session = meta.get("label_session_id", meta["session_id"])
    trades = pd.DataFrame({"session_id": pnl_session.to_numpy(), "pos": pos,
                           "requested_pos": requested, "execution_eligible": eligible,
                           "gross_usd": gross, "cost_usd": cost, "net_usd": gross - cost})
    return trades


def summarize(trades, capital_usd=None):
    """Gross/net P&L, turnover, trade stats, drawdown, daily statistics."""
    t = trades[trades["pos"] != 0]
    daily = trades.groupby("session_id")["net_usd"].sum()
    eq = pd.Series(np.r_[0.0, daily.cumsum().to_numpy()])
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
        "max_drawdown_eod_usd": float(dd),
        "drawdown_basis": "end-of-day equity including initial zero",
        "n_execution_eligible": int(trades["execution_eligible"].sum()),
        "execution_coverage": float(trades["execution_eligible"].mean()) if len(trades) else 0.0,
        "mean_daily_net_usd": float(daily.mean()) if len(daily) else 0.0,
        "std_daily_net_usd": float(daily.std()) if len(daily) > 1 else np.nan,
    }
    if capital_usd:  # percentages only with a stated capital assumption
        out["capital_usd_assumed"] = capital_usd
        out["net_return_pct_on_capital"] = 100 * out["net_usd"] / capital_usd
    out["economic_status"] = "illustrative_bar_price_proxy" if out["n_execution_eligible"] else "not_evaluable_no_entry_before_exit"
    if not out["n_execution_eligible"]:
        for key in ("gross_usd", "cost_usd", "net_usd", "expectancy_net_usd", "expectancy_gross_usd",
                    "max_drawdown_usd", "max_drawdown_eod_usd", "mean_daily_net_usd",
                    "std_daily_net_usd", "net_return_pct_on_capital"):
            if key in out:
                out[key] = np.nan
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
