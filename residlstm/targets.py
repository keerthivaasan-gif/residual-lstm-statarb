"""STEP 3 — define the prediction target on the residual.

We predict the **k-day-forward residual change**, where k == ``cfg.label_horizon``
must equal the rebalance frequency (predicting 1-day moves but trading every 5
days is the silent killer the plan warns about).

    target[t] = sum(residual[t+1 .. t+k])          # k-day forward residual change

Three flavors:
  * "raw"    -- the forward change itself (preserves magnitude for sizing)
  * "zscore" -- cross-sectionally standardized each day (default; comparable names)
  * "sign"   -- direction only (classification framing)

LABEL OVERLAP: target[t] uses residuals through t+k, so targets on consecutive
days share up to k days of future information.  That overlap is exactly what
forces the purge/embargo in Step 4 -- ``label_horizon`` is threaded straight into
the CV splitter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import CFG


def make_targets(residuals: pd.DataFrame, cfg=CFG) -> pd.DataFrame:
    """k-day-forward residual change as a date x ticker panel, per ``target_kind``."""
    k = cfg.label_horizon
    # forward sum of residuals over (t, t+k]; reverse-rolling trick
    fwd = (residuals.fillna(0.0)[::-1]
           .rolling(k, min_periods=k).sum()[::-1]
           .shift(-1))                                   # shift so it starts at t+1
    fwd = fwd.where(residuals.notna())

    if cfg.target_kind == "raw":
        out = fwd
    elif cfg.target_kind == "zscore":
        mu = fwd.mean(axis=1)
        sd = fwd.std(axis=1).replace(0, np.nan)
        out = fwd.sub(mu, axis=0).div(sd, axis=0)
    elif cfg.target_kind == "sign":
        out = np.sign(fwd)
    else:
        raise ValueError(f"unknown target_kind {cfg.target_kind!r}")
    return out


def label_horizon(cfg=CFG) -> int:
    """The k that downstream purge/embargo logic must respect."""
    return cfg.label_horizon
