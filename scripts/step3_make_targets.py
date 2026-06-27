"""STEP 3 — build the k-day-forward residual target.

    python scripts/step3_make_targets.py

Reads outputs/residuals.pkl, writes outputs/targets.pkl.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CFG, OUTPUTS
from residlstm.targets import make_targets


def main():
    with open(OUTPUTS / "residuals.pkl", "rb") as f:
        residuals = pickle.load(f)

    print(f"[step3] target = {CFG.label_horizon}-day forward residual change "
          f"({CFG.target_kind}); rebalance must match k={CFG.label_horizon}")
    targets = make_targets(residuals, CFG)

    with open(OUTPUTS / "targets.pkl", "wb") as f:
        pickle.dump(targets, f)
    print(f"[step3] wrote {OUTPUTS / 'targets.pkl'} "
          f"({int(targets.notna().sum().sum())} labeled cells)")


if __name__ == "__main__":
    main()
