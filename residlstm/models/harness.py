"""Shared walk-forward prediction harness for the supervised panel models.

Both the trees baseline and the panel-LSTM are trained the same way: a single
model across the whole cross-section, refit per purged walk-forward window, used
to predict the held-out test days.  This routine wires any model exposing
``fit(X, y)`` / ``predict(X)`` through the Step-4 folds and assembles the
out-of-sample predictions back into a date x ticker *signal panel* -- the uniform
currency the Step-6 portfolio layer consumes (same shape the s-score emits).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import CFG
from residlstm.cv import purged_walk_forward


def walk_forward_signal(make_model, X, y, idx, panel_dates, panel_tickers, cfg=CFG):
    """Train/predict across purged folds; return a date x ticker OOS signal panel.

    Parameters
    ----------
    make_model : callable -> model with .fit(X,y)/.predict(X) (fresh per fold)
    X, y, idx  : sequence tensor, targets, and [date, ticker] index (from build_sequences)
    panel_dates, panel_tickers : axes of the output panel
    """
    date_to_pos = {d: i for i, d in enumerate(panel_dates)}
    sample_pos = idx["date"].map(date_to_pos).to_numpy()

    out = pd.DataFrame(np.nan, index=panel_dates, columns=panel_tickers)
    n_days = len(panel_dates)
    n_fit = 0

    for fold in purged_walk_forward(n_days, cfg):
        train_set = set(fold.train_days.tolist())
        test_set = set(fold.test_days.tolist())
        tr = np.array([p in train_set for p in sample_pos])
        te = np.array([p in test_set for p in sample_pos])
        if tr.sum() < 50 or te.sum() == 0:
            continue

        model = make_model()
        model.fit(X[tr], y[tr])
        preds = np.asarray(model.predict(X[te])).reshape(-1)
        n_fit += 1

        te_idx = idx.loc[te]
        for p, tic, val in zip(te_idx["date"], te_idx["ticker"], preds):
            out.at[p, tic] = val

    if n_fit == 0:
        raise RuntimeError("No folds trained -- check data length vs cv_* settings.")
    return out
