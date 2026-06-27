"""STEP 5 (baselines) — s-score and trees, BEFORE touching the LSTM.

    python scripts/step5_run_baselines.py

The plan's strongest advice: establish baseline signals on identical residuals
first.  Writes outputs/signals/sscore.pkl and outputs/signals/trees.pkl.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CFG, OUTPUTS
from residlstm.models.sscore import signal_panel
from residlstm.models.trees import make_trees_model
from residlstm.models.harness import walk_forward_signal
from residlstm.features.sequences import build_sequences

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

    # --- 1) s-score: causal, no training needed ----------------------------
    print("[step5] fitting Avellaneda-Lee s-score baseline...")
    sscore_sig = signal_panel(residuals, CFG)
    with open(SIGDIR / "sscore.pkl", "wb") as f:
        pickle.dump(sscore_sig, f)

    # --- 2) trees: nonlinear, no sequence memory ---------------------------
    print("[step5] building sequences + training trees baseline (walk-forward)...")
    vr = _volume_ratio(panel["dollar_volume"])
    X, y, idx = build_sequences(residuals, targets, vr, CFG)
    print(f"        sequence tensor X={X.shape}, samples={len(y)}")
    trees_sig = walk_forward_signal(
        make_trees_model(CFG), X, y, idx,
        residuals.index, residuals.columns, CFG)
    with open(SIGDIR / "trees.pkl", "wb") as f:
        pickle.dump(trees_sig, f)

    print(f"[step5] wrote {SIGDIR/'sscore.pkl'} and {SIGDIR/'trees.pkl'}")


if __name__ == "__main__":
    main()
