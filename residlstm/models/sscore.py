"""STEP 5 — the Avellaneda-Lee s-score baseline (the domain benchmark).

The linear, fixed-kappa story.  On the cumulative idiosyncratic process
X_t = sum of residuals we fit an OU / AR(1) on a trailing window ending t-1:

    X_{t+1} = a + b * X_t + eps

and read off the OU parameters
    kappa     = -log(b)                     mean-reversion speed
    m         = a / (1 - b)                  equilibrium level
    sigma_eq  = sqrt(var(eps) / (1 - b^2))   equilibrium dispersion
    s_score   = (X_t - m) / sigma_eq

The trading signal is ``-s_score``: short when the residual is rich (s high), long
when cheap (s low).  This is the linear special case the LSTM must beat -- OU is
the overdamped Langevin oscillator with fixed kappa; the LSTM lets kappa and the
drift become state-dependent.

``signal_panel`` returns a date x ticker panel of ``-s_score`` aligned to the
residual panel, computed causally (each day uses only its trailing window).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import CFG
from residlstm.features.rmt import cumulative_residual


def _ou_sscore(x: np.ndarray) -> float:
    """s-score for the last point of a trailing cumulative-residual window."""
    if np.isnan(x).any() or len(x) < 30:
        return np.nan
    x_lag, x_next = x[:-1], x[1:]
    # OLS  x_next = a + b x_lag
    b, a = np.polyfit(x_lag, x_next, 1)
    if not (0 < b < 1):              # not mean-reverting on this window
        return np.nan
    resid = x_next - (a + b * x_lag)
    var_eps = np.var(resid, ddof=1)
    m = a / (1.0 - b)
    sigma_eq = np.sqrt(var_eps / (1.0 - b * b))
    if sigma_eq == 0:
        return np.nan
    return (x[-1] - m) / sigma_eq


def estimate_kappa(x: np.ndarray) -> float:
    """OU mean-reversion speed kappa = -log(b) on a window (for sanity checks)."""
    if np.isnan(x).any() or len(x) < 30:
        return np.nan
    x_lag, x_next = x[:-1], x[1:]
    b, _ = np.polyfit(x_lag, x_next, 1)
    if not (0 < b < 1):
        return np.nan
    return -np.log(b)


def signal_panel(residuals: pd.DataFrame, cfg=CFG) -> pd.DataFrame:
    """date x ticker panel of the (-s_score) trading signal, computed causally."""
    X = cumulative_residual(residuals)
    W = cfg.rmt_window
    out = pd.DataFrame(np.nan, index=X.index, columns=X.columns)
    xvals = X.to_numpy()
    for t in range(W, len(X)):
        win = xvals[t - W:t]                 # cumulative residual up to t-1
        row = np.array([_ou_sscore(win[:, j]) for j in range(win.shape[1])])
        out.iloc[t] = -row                   # signal = -s_score (mean reversion)
    return out
