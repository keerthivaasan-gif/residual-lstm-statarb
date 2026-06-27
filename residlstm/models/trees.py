"""STEP 5 — the nonlinear tabular baseline (the scientific control).

This is the middle rung of the three-baseline ladder:

    s-score   = linear + domain structure (fixed-kappa OU)
    trees     = NONLINEAR but NO sequence memory  <-- this file
    panel-LSTM= nonlinear + sequence memory

The trees model sees the residual lags as a *flat* feature vector (it cannot use
their order), so comparing LSTM-vs-trees isolates the value of *sequence memory*
specifically, not generic nonlinearity.  If the LSTM beats the s-score but not the
trees, the win was nonlinearity, not memory -- and the LSTM is not justified.

Backend: scikit-learn ``HistGradientBoostingRegressor`` by default (no libomp,
the stand-in the project's requirements already endorse); real ``xgboost`` if
installed and selected.
"""
from __future__ import annotations

import numpy as np

from config import CFG


class _FlatGBM:
    """Wraps a gradient-boosting regressor to accept the (n, L, C) tensor by
    flattening it to (n, L*C) -- deliberately discarding temporal structure."""

    def __init__(self, backend="sklearn"):
        if backend == "xgboost":
            import xgboost as xgb  # noqa: F401
            self.model = xgb.XGBRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
            )
        else:
            from sklearn.ensemble import HistGradientBoostingRegressor
            self.model = HistGradientBoostingRegressor(
                max_depth=4, learning_rate=0.05, max_iter=300,
                l2_regularization=1.0,
            )

    @staticmethod
    def _flat(X):
        return X.reshape(X.shape[0], -1)

    def fit(self, X, y):
        self.model.fit(self._flat(X), y)
        return self

    def predict(self, X):
        return self.model.predict(self._flat(X))


def make_trees_model(cfg=CFG):
    """Factory the walk-forward harness calls once per fold."""
    backend = getattr(cfg, "trees_backend", "sklearn")
    return lambda: _FlatGBM(backend)
