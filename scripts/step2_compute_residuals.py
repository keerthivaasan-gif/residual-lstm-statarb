"""STEP 2 — compute causal RMT idiosyncratic residuals.

    python scripts/step2_compute_residuals.py

Reads outputs/panel.pkl, writes outputs/residuals.pkl.  On synthetic data it also
reports the correlation between the recovered residuals and the simulator's known
ground-truth residuals -- a high value confirms the RMT block works and is causal.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CFG, OUTPUTS
from residlstm.features.rmt import compute_residuals, characteristic_neutralize


def main():
    with open(OUTPUTS / "panel.pkl", "rb") as f:
        panel = pickle.load(f)

    print("[step2] computing causal RMT residuals (trailing window ends at t-1)...")
    residuals = compute_residuals(panel["returns_u"], CFG)

    if CFG.conditional_factors:
        print("[step2] applying conditional characteristic cleaning (size/mom/vol)...")
        residuals = characteristic_neutralize(
            residuals, panel["returns_u"], panel["dollar_volume"], CFG)

    truth = panel.get("true_residuals")
    if truth is not None:
        a = residuals.to_numpy().ravel()
        b = truth.reindex_like(residuals).to_numpy().ravel()
        ok = ~np.isnan(a) & ~np.isnan(b)
        corr = np.corrcoef(a[ok], b[ok])[0, 1]
        print(f"[step2] recovered-vs-true residual corr = {corr:.3f} "
              f"(should be high; confirms factor removal works)")

    with open(OUTPUTS / "residuals.pkl", "wb") as f:
        pickle.dump(residuals, f)
    print(f"[step2] wrote {OUTPUTS / 'residuals.pkl'}")


if __name__ == "__main__":
    main()
