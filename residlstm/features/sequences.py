"""STEP 2 — turn the residual panel into a 3-D sequence tensor for the LSTM.

This is what justifies a sequence model in Step 5: instead of a flat feature
table we build, per (name, date), the last ``L`` standardized residuals plus
optional residual-vol and volume-ratio channels.  Shape:

    X : (n_samples, L, n_channels)
    y : (n_samples,)                      aligned target
    idx : DataFrame[date, ticker]         maps each sample back to the panel

Belt-and-braces ``shift(1)``
----------------------------
The residuals are already causal (Step 2 RMT), but per the plan we *also* lag the
feature window so the last residual in the sequence is from t-1, and the target is
the k-day-forward change measured from t.  Two independent guards against leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import CFG


def detrended_level(residuals: pd.DataFrame, cfg=CFG) -> pd.DataFrame:
    """Cumulative residual, locally z-scored over a trailing window.

    The cumulative residual integrates RMT estimation error into a slow random-walk
    drift, so its *raw* level barely predicts the forward reversion.  Subtracting a
    trailing rolling mean and dividing by a trailing rolling std de-drifts it -- the
    same trick the Avellaneda-Lee s-score uses when it re-estimates a local
    equilibrium ``m`` on each window.  This is what makes the level learnable.
    """
    from residlstm.features.rmt import cumulative_residual
    Y = cumulative_residual(residuals)
    W = cfg.detrend_window
    mu = Y.rolling(W, min_periods=W // 2).mean()
    sd = Y.rolling(W, min_periods=W // 2).std()
    return (Y - mu) / sd.replace(0, np.nan)


def build_sequences(residuals: pd.DataFrame, targets: pd.DataFrame,
                    volume_ratio: pd.DataFrame | None = None, cfg=CFG):
    """Return (X, y, idx) where idx is a DataFrame with columns [date, ticker].

    Channel 0 is the trailing path of the *detrended* cumulative residual (the
    drift-free mean-reversion signal).  A sample exists for (ticker, date t) when
    the window [t-L, t-1] is fully present and the target at t is present.
    """
    L = cfg.seq_len
    resid = residuals
    level = detrended_level(residuals, cfg)        # drift-free local deviation
    dates = resid.index
    tickers = resid.columns

    X_rows, y_rows, rec_date, rec_tic = [], [], [], []

    rv = volume_ratio
    for j, tic in enumerate(tickers):
        r = resid[tic].to_numpy()
        lv = level[tic].to_numpy()
        tgt = targets[tic].to_numpy()
        vr = rv[tic].to_numpy() if rv is not None else None
        for t in range(L, len(dates)):
            if np.isnan(tgt[t]):
                continue
            win = lv[t - L:t]                      # detrended level path, t-L .. t-1
            if np.isnan(win).any():
                continue
            channels = [win]
            if cfg.use_vol_channel:
                # trailing residual volatility as a slow-moving regime channel
                vol = np.array([np.nanstd(r[max(0, t - L - i):t - i]) or 0.0
                                for i in range(L)][::-1])
                channels.append(vol)
            if cfg.use_volume_channel and vr is not None:
                vwin = vr[t - L:t]
                vwin = np.nan_to_num(vwin, nan=1.0)
                channels.append(vwin)
            X_rows.append(np.stack(channels, axis=-1))   # (L, C)
            y_rows.append(tgt[t])
            rec_date.append(dates[t])
            rec_tic.append(tic)

    if not X_rows:
        raise ValueError("No sequences built -- check seq_len vs data length.")

    X = np.asarray(X_rows, dtype=np.float64)
    # channel 0 is already the trailing-z-scored (detrended) level; do not re-scale
    y = np.asarray(y_rows, dtype=np.float64)
    idx = pd.DataFrame({"date": rec_date, "ticker": rec_tic})
    return X, y, idx
