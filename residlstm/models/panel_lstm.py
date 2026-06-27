"""STEP 5 — the panel-LSTM (the model the whole project is testing).

Why an LSTM is justified here (not overkill): the input is a *sequence with
genuine memory* -- the residual path -- which is precisely the carve-out the plan
names for LSTMs.  OU is the overdamped Langevin oscillator with fixed kappa; the
LSTM is the data-driven generalization where kappa and the drift become
state-dependent.  We train ONE panel model across the whole cross-section (not one
per name): per-stock models have a few hundred points and overfit; the panel sees
thousands of sequences and learns the general reversion dynamics.

Backend strategy
----------------
* If TensorFlow/Keras is installed -> a Conv1D -> BiLSTM stack (mirrors the
  architecture in project2/lstm_model.py), Huber loss, early stopping.
* If not -> a scikit-learn ``MLPRegressor`` on the flattened sequence, so the
  head-to-head still runs end-to-end.  This fallback is NOT a sequence model; it
  is clearly logged as a stand-in so results stay honest.
"""
from __future__ import annotations

import warnings

import numpy as np

from config import CFG

try:                                   # optional heavy dependency
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import Huber
    from tensorflow.keras.callbacks import EarlyStopping
    KERAS_AVAILABLE = True
except Exception:                      # pragma: no cover - depends on environment
    KERAS_AVAILABLE = False


class _KerasLSTM:
    """Conv1D -> stacked BiLSTM regression head on the (n, L, C) tensor."""

    def __init__(self, seq_len, n_channels, epochs=40, batch=256):
        self.epochs, self.batch = epochs, batch
        model = Sequential([
            Conv1D(32, 3, padding="causal", activation="relu",
                   input_shape=(seq_len, n_channels)),
            Bidirectional(LSTM(48, return_sequences=True)),
            Bidirectional(LSTM(48)),
            Dense(32, activation="relu"),
            Dense(1),
        ])
        model.compile(optimizer=Adam(1e-3), loss=Huber(), metrics=["mae"])
        self.model = model

    def fit(self, X, y):
        es = EarlyStopping(patience=5, restore_best_weights=True,
                           monitor="val_loss")
        self.model.fit(X, y, validation_split=0.15, epochs=self.epochs,
                       batch_size=self.batch, verbose=0, callbacks=[es])
        return self

    def predict(self, X):
        return self.model.predict(X, verbose=0).reshape(-1)


class _MLPFallback:
    """scikit-learn MLP on the flattened sequence (no real temporal memory)."""

    def __init__(self):
        from sklearn.neural_network import MLPRegressor
        self.model = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu",
                                  alpha=1e-3, max_iter=300, early_stopping=True)

    def fit(self, X, y):
        self.model.fit(X.reshape(X.shape[0], -1), y)
        return self

    def predict(self, X):
        return self.model.predict(X.reshape(X.shape[0], -1))


def make_lstm_model(seq_len, n_channels, cfg=CFG):
    """Factory the walk-forward harness calls once per fold."""
    if KERAS_AVAILABLE:
        return lambda: _KerasLSTM(seq_len, n_channels)
    warnings.warn(
        "TensorFlow not installed -> using a scikit-learn MLP fallback for the "
        "'LSTM' slot. This is NOT a sequence model; install tensorflow to run the "
        "real Conv1D-BiLSTM. Results are labeled 'lstm(mlp-fallback)'.",
        stacklevel=2,
    )
    return lambda: _MLPFallback()


def backend_name() -> str:
    return "lstm(keras)" if KERAS_AVAILABLE else "lstm(mlp-fallback)"
