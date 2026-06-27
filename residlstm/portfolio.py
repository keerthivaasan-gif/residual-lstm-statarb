"""STEP 6 — convert a signal panel into a market-neutral, traded P&L.

Pipeline (all on a date x ticker signal panel, uniform across models):

  1. SIZE        -- cross-sectionally z-score the signal each day; long the top
                    decile, short the bottom decile (weight proportional to z
                    within the deciles, gross-normalized to 1).
  2. NEUTRALIZE  -- dollar-neutral (subtract the cross-sectional mean) then
                    beta-neutral (regress out market beta); optional sector-neutral.
                    Beta-neutrality matters for a residual strategy and is the
                    plan's [EXTENDS] over the tutorial's dollar-neutral-only.
  3. EXECUTE     -- signal at close of t trades at t+1 (``exec_lag``); no same-bar
                    fills.  Gross P&L = position(t) . return(t+1).

Returns a tidy object with daily gross returns, positions, and turnover; the cost
layer (costs.py) then nets it down.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import CFG


@dataclass
class Book:
    positions: pd.DataFrame   # date x ticker target weights (post-neutralization)
    gross_ret: pd.Series      # daily gross strategy return
    turnover: pd.Series       # daily one-way turnover (sum |dw|)


def _decile_weights(row: pd.Series, decile: float) -> pd.Series:
    s = row.dropna()
    if len(s) < 10:
        return pd.Series(0.0, index=row.index)
    z = (s - s.mean()) / (s.std() or 1.0)
    n = max(1, int(len(z) * decile))
    longs = z.nlargest(n)
    shorts = z.nsmallest(n)
    w = pd.Series(0.0, index=row.index)
    if longs.sum() != 0:
        w[longs.index] = longs / longs.abs().sum()
    if shorts.abs().sum() != 0:
        w[shorts.index] = shorts / shorts.abs().sum()
    return w


def _proportional_weights(row: pd.Series, cap: float) -> pd.Series:
    """Weight proportional to the cross-sectional z-score, demeaned (dollar-neutral),
    capped per name, then gross-normalized to 1."""
    s = row.dropna()
    if len(s) < 10:
        return pd.Series(0.0, index=row.index)
    z = (s - s.mean()) / (s.std() or 1.0)
    gross = z.abs().sum()
    if gross == 0:
        return pd.Series(0.0, index=row.index)
    w = z / gross
    w = w.clip(-cap, cap)
    g = w.abs().sum()
    if g > 0:
        w = w / g
    return w.reindex(row.index).fillna(0.0)


def _row_weights(row: pd.Series, cfg) -> pd.Series:
    if cfg.weighting == "decile":
        return _decile_weights(row, cfg.decile)
    return _proportional_weights(row, cfg.per_name_cap)


def _neutralize(w: pd.Series, beta: pd.Series | None, sector: pd.Series | None,
                cfg=CFG) -> pd.Series:
    held = w[w != 0].index
    if len(held) == 0:
        return w
    w = w.copy()
    # dollar-neutral: remove the mean of the *held* book
    w[held] = w[held] - w[held].mean()
    # beta-neutral: regress weights on beta, keep residual
    if cfg.beta_neutral and beta is not None:
        b = beta.reindex(held).fillna(beta.mean())
        denom = (b * b).sum()
        if denom > 0:
            w[held] = w[held] - (w[held] * b).sum() / denom * b
    # sector-neutral: demean within each sector
    if cfg.sector_neutral and sector is not None:
        sec = sector.reindex(held)
        for s in sec.dropna().unique():
            members = sec[sec == s].index
            w[members] = w[members] - w[members].mean()
    # renormalize gross to 1
    gross = w[held].abs().sum()
    if gross > 0:
        w[held] = w[held] / gross
    return w


def build_book(signal: pd.DataFrame, forward_returns: pd.DataFrame,
               beta: pd.Series | None = None, sector: pd.Series | None = None,
               cfg=CFG) -> Book:
    """Turn a signal panel into a neutralized, lagged, traded book."""
    positions = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    for d in signal.index:
        w = _row_weights(signal.loc[d], cfg)
        w = _neutralize(w, beta, sector, cfg)
        positions.loc[d] = w

    # execution lag: hold position from t, earn return at t+exec_lag
    held = positions.shift(cfg.exec_lag)
    gross_ret = (held * forward_returns).sum(axis=1)

    turnover = held.diff().abs().sum(axis=1).fillna(held.abs().sum(axis=1))
    return Book(positions=held.fillna(0.0), gross_ret=gross_ret.fillna(0.0),
                turnover=turnover.fillna(0.0))
