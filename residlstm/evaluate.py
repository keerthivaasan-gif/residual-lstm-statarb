"""STEP 7 — evaluate honestly.

Headline metrics (net, out-of-sample): annualized Sharpe, max drawdown, annual
turnover, and the decay curve (Sharpe vs decision lag of 1d / 1wk -- a flat decay
means a fragile signal).  Plus the two pieces the plan singles out:

* **Deflated Sharpe Ratio** (Bailey & López de Prado).  You try many configs
  (L, k, deciles, cadence), so a high in-sample Sharpe is partly luck.  DSR is the
  probability the *true* Sharpe is > 0 after penalizing for the number of trials
  and for non-normal returns.  This is the line between a curve-fit and a result.

* **Sanity gate:** net Sharpe > 3 => assume a bug.  Re-audit RMT causality
  (Step 2), survivorship (Step 0), or label leakage (Step 4) before believing it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from config import CFG


def sharpe(returns: pd.Series, cfg=CFG) -> float:
    r = returns.dropna()
    if r.std() == 0 or len(r) < 2:
        return 0.0
    return np.sqrt(cfg.ann_factor) * r.mean() / r.std()


def max_drawdown(returns: pd.Series) -> float:
    curve = (1 + returns.fillna(0)).cumprod()
    peak = curve.cummax()
    return float((curve / peak - 1).min())


def annual_turnover(turnover: pd.Series, cfg=CFG) -> float:
    return float(turnover.mean() * cfg.ann_factor)


def decay_curve(signal: pd.DataFrame, forward_returns: pd.DataFrame,
                lags=(1, 5), cfg=CFG) -> dict:
    """Sharpe of a simple z-weighted book as the decision lag grows.  Flat (no
    decay) => the signal is not exploiting fast mean-reversion and is fragile."""
    from residlstm.portfolio import build_book
    out = {}
    for lag in lags:
        tmp = type(cfg)(**cfg.to_dict())
        tmp.exec_lag = lag
        book = build_book(signal, forward_returns, cfg=tmp)
        out[f"lag_{lag}d"] = sharpe(book.gross_ret, cfg)
    return out


def deflated_sharpe_ratio(returns: pd.Series, n_trials: int, cfg=CFG) -> float:
    """Probability the true Sharpe > 0 given ``n_trials`` configurations tried.

    Implements the Bailey-Lopez de Prado DSR: deflate the observed SR by the
    expected maximum SR under the null of ``n_trials`` independent trials, then
    map through the SR's sampling distribution (with skew/kurtosis correction).
    """
    r = returns.dropna()
    n = len(r)
    if n < 10 or r.std() == 0:
        return float("nan")

    sr = r.mean() / r.std()                       # per-period (not annualized) SR
    skew = stats.skew(r)
    kurt = stats.kurtosis(r, fisher=False)        # non-excess kurtosis

    # expected maximum SR under the null across n_trials (variance of trial SRs ~1/n)
    e_max = np.sqrt(1.0 / n) * (
        (1 - np.euler_gamma) * stats.norm.ppf(1 - 1.0 / n_trials)
        + np.euler_gamma * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    )

    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4.0 * sr ** 2)
    if denom <= 0:
        return float("nan")
    z = (sr - e_max) * np.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z))


@dataclass
class Report:
    name: str
    net_sharpe: float
    gross_sharpe: float
    max_dd: float
    ann_turnover: float
    dsr: float
    sweep_sharpe: dict = field(default_factory=dict)
    decay: dict = field(default_factory=dict)
    bug_warning: bool = False


def evaluate_strategy(name, book, net_ret, sweep, signal, forward_returns,
                      cfg=CFG) -> Report:
    """Assemble the full honest scorecard for one model's signal."""
    ns = sharpe(net_ret, cfg)
    rep = Report(
        name=name,
        net_sharpe=ns,
        gross_sharpe=sharpe(book.gross_ret, cfg),
        max_dd=max_drawdown(net_ret),
        ann_turnover=annual_turnover(book.turnover, cfg),
        dsr=deflated_sharpe_ratio(net_ret, cfg.n_trials, cfg),
        sweep_sharpe={c: sharpe(sweep[c], cfg) for c in sweep.columns},
        decay=decay_curve(signal, forward_returns, cfg=cfg),
        bug_warning=ns > 3.0,
    )
    return rep


def report_table(reports) -> pd.DataFrame:
    """Head-to-head table -- the deliverable."""
    rows = []
    for r in reports:
        row = {
            "model": r.name,
            "net_sharpe": round(r.net_sharpe, 3),
            "gross_sharpe": round(r.gross_sharpe, 3),
            "max_dd": round(r.max_dd, 3),
            "ann_turnover": round(r.ann_turnover, 1),
            "deflated_SR(P>0)": round(r.dsr, 3),
        }
        for c, v in r.sweep_sharpe.items():
            row[f"net_SR_{c}"] = round(v, 3)
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")
