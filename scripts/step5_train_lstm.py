"""STEP 5 (LSTM) — the panel-LSTM as a swap-in signal block.

    python scripts/step5_train_lstm.py

One panel model across the whole cross-section, refit per purged walk-forward
window.  Uses the real Conv1D-BiLSTM if TensorFlow is installed, otherwise a
clearly-labeled scikit-learn MLP fallback (so this still runs end-to-end).
Writes outputs/signals/lstm.pkl.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CFG, OUTPUTS
from residlstm.features.sequences import build_sequences
from residlstm.models.harness import walk_forward_signal
from residlstm.models.panel_lstm import make_lstm_model, backend_name

SIGDIR = OUTPUTS / "signals"
SIGDIR.mkdir(exist_ok=True)


def _volume_ratio(dollar_volume):
    adv = dollar_volume.rolling(CFG.rmt_window, min_periods=20).mean()
    return (dollar_volume / adv).clip(0, 5)


def main():
    with open(OUTPUTS / "panel.pkl", "rb") as f:
        panel = pickle.load(f)
    with open(OUTPUTS / "residuals.pkl", "rb") as f:
        residuals = pickle.load(f)
    with open(OUTPUTS / "targets.pkl", "rb") as f:
        targets = pickle.load(f)

    print(f"[step5-lstm] backend = {backend_name()}")
    vr = _volume_ratio(panel["dollar_volume"])
    X, y, idx = build_sequences(residuals, targets, vr, CFG)
    n_channels = X.shape[2]
    print(f"[step5-lstm] sequence tensor X={X.shape}; training panel model...")

    sig = walk_forward_signal(
        make_lstm_model(CFG.seq_len, n_channels, CFG), X, y, idx,
        residuals.index, residuals.columns, CFG)

    with open(SIGDIR / "lstm.pkl", "wb") as f:
        pickle.dump(sig, f)
    print(f"[step5-lstm] wrote {SIGDIR / 'lstm.pkl'}")


if __name__ == "__main__":
    main()
