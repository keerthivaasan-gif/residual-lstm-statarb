"""STEP 2 (+ the look-ahead GAP fix) — RMT-cleaned idiosyncratic residuals.

The plan's hardest correctness point:

    > The eigenvectors and regression betas must themselves be estimated on a
    > trailing window ending at t-1 only.  Fitting PCA on the full sample leaks
    > the future into *every* residual -- a look-ahead violation that shift(1)
    > does NOT catch.

So this module never sees the whole sample at once when it builds the loadings.
For each date ``t`` it:

  1. takes the return window ending the day *before* t  -> rows [t-W, t-1],
  2. standardizes and forms the correlation matrix,
  3. eigendecomposes it and finds the eigenvalues above the Marchenko-Pastur
     noise edge (RMT cleaning) -- those are the market+sector modes,
  4. projects day-t's standardized return onto that signal subspace and keeps the
     orthogonal complement: the idiosyncratic residual.

Therefore ``residual[t]`` depends only on returns up to and including ``t`` and
loadings estimated strictly before ``t``.  ``tests/test_rmt_causality.py`` proves
this is bit-identical whether computed on the full panel or the panel truncated
right after ``t``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import CFG


def marchenko_pastur_edge(n_vars: int, n_obs: int, var: float = 1.0) -> float:
    """Upper edge of the MP noise spectrum for an n_vars x n_vars correlation
    matrix estimated from n_obs observations.  Eigenvalues above this are signal."""
    q = n_vars / n_obs
    return var * (1.0 + np.sqrt(q)) ** 2


def _signal_eigvecs(window: np.ndarray, top_k_cap: int):
    """Return the eigenvectors (columns) of the trailing-window correlation matrix
    whose eigenvalues exceed the MP edge -- i.e. the cleaned market+sector modes.

    Falls back to at least the largest eigenmode (the market), and never returns
    more than ``top_k_cap`` modes.
    """
    # standardize each name over the window (z-score) -> correlation, not covariance
    mu = window.mean(axis=0)
    sd = window.std(axis=0, ddof=1)
    sd = np.where(sd == 0, 1.0, sd)
    z = (window - mu) / sd

    n_obs, n_vars = z.shape
    corr = (z.T @ z) / (n_obs - 1)
    eigvals, eigvecs = np.linalg.eigh(corr)        # ascending
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    edge = marchenko_pastur_edge(n_vars, n_obs)
    k = int(np.sum(eigvals > edge))
    k = max(1, min(k, top_k_cap))                  # >=1 market mode, <= cap
    return eigvecs[:, :k], mu, sd


def compute_residuals(returns: pd.DataFrame, cfg=CFG) -> pd.DataFrame:
    """Causal RMT idiosyncratic residuals, same shape as ``returns``.

    ``returns`` is a date x ticker panel (NaN where a name is absent/delisted).
    """
    R = returns
    W, cap = cfg.rmt_window, cfg.rmt_top_k
    resid = pd.DataFrame(np.nan, index=R.index, columns=R.columns)
    vals = R.to_numpy()
    n_dates = len(R)

    for t in range(W, n_dates):
        win = vals[t - W:t]                        # rows [t-W, t-1]  <- ends BEFORE t
        today = vals[t]                            # day-t return (allowed)

        # names usable today: full window history AND present today
        ok = ~np.isnan(win).any(axis=0) & ~np.isnan(today)
        if ok.sum() < cap + 2:                     # not enough names for a clean fit
            continue

        win_ok = win[:, ok]
        eigvecs, mu, sd = _signal_eigvecs(win_ok, cap)

        z_today = (today[ok] - mu) / sd            # standardize today w/ window stats
        proj = eigvecs @ (eigvecs.T @ z_today)     # project onto signal subspace
        resid_z = z_today - proj                   # orthogonal complement
        resid_today = resid_z * sd                 # back to return units

        out = np.full(R.shape[1], np.nan)
        out[ok] = resid_today
        resid.iloc[t] = out

    return resid


def characteristic_neutralize(residuals: pd.DataFrame, returns: pd.DataFrame,
                              dollar_volume: pd.DataFrame, cfg=CFG) -> pd.DataFrame:
    """[#3] Conditional / characteristic factor cleaning on top of statistical RMT.

    A static RMT projection removes *unconditional* common modes. Real factor
    exposure is conditional on firm characteristics (size, momentum, volatility) --
    the insight behind IPCA / conditional latent-factor models (Pelger et al.).
    We approximate it cheaply and causally: each day we cross-sectionally regress
    the RMT residual on standardized trailing characteristics and keep the residual,
    so any residual variation still aligned with size/momentum/vol is removed.

    All characteristics use trailing data only, and the regression is same-day
    cross-sectional, so this adds no look-ahead.
    """
    mom_w, vol_w = cfg.char_mom_window, cfg.char_vol_window
    # causal characteristics (each is a date x ticker panel)
    size = np.log(dollar_volume.rolling(cfg.rmt_window, min_periods=20).mean())
    mom = returns.rolling(mom_w, min_periods=mom_w // 2).sum().shift(1)   # trailing return, lagged
    vol = returns.rolling(vol_w, min_periods=vol_w // 2).std().shift(1)

    out = pd.DataFrame(np.nan, index=residuals.index, columns=residuals.columns)
    R = residuals.to_numpy()
    chars = [size.to_numpy(), mom.to_numpy(), vol.to_numpy()]

    for t in range(len(residuals)):
        y = R[t]
        ok = ~np.isnan(y)
        for c in chars:
            ok &= ~np.isnan(c[t])
        if ok.sum() < 10:
            out.iloc[t] = y
            continue
        # design matrix: intercept + z-scored characteristics
        cols = [np.ones(ok.sum())]
        for c in chars:
            v = c[t][ok]
            v = (v - v.mean()) / (v.std() or 1.0)
            cols.append(v)
        X = np.column_stack(cols)
        yv = y[ok]
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
        resid = yv - X @ beta
        row = np.full(R.shape[1], np.nan)
        row[ok] = resid
        out.iloc[t] = row
    return out


def cumulative_residual(residuals: pd.DataFrame) -> pd.DataFrame:
    """Cumulative idiosyncratic process X_t = sum of residuals (the OU level the
    Avellaneda-Lee s-score is fit on).  NaNs are treated as zero increments."""
    return residuals.fillna(0.0).cumsum().where(residuals.notna())
