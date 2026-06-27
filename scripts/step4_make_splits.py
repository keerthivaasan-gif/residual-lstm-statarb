"""STEP 4 — materialize the purged, embargoed walk-forward folds.

    python scripts/step4_make_splits.py

Reads outputs/residuals.pkl for the date axis, writes outputs/folds.pkl, and
prints the fold geometry so you can eyeball the purge gap and embargo.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CFG, OUTPUTS
from residlstm.cv import purged_walk_forward


def main():
    with open(OUTPUTS / "residuals.pkl", "rb") as f:
        residuals = pickle.load(f)
    dates = residuals.index

    folds = list(purged_walk_forward(len(dates), CFG))
    print(f"[step4] {len(folds)} purged walk-forward folds "
          f"(train={CFG.cv_train_days}d, test={CFG.cv_test_days}d, "
          f"purge={CFG.label_horizon}d, embargo={CFG.cv_embargo}d)")
    for i, fl in enumerate(folds[:3]):
        print(f"        fold {i}: train {fl.train_days[0]}..{fl.train_days[-1]} | "
              f"test {fl.test_days[0]}..{fl.test_days[-1]}")

    with open(OUTPUTS / "folds.pkl", "wb") as f:
        pickle.dump([(fl.train_days, fl.test_days) for fl in folds], f)
    print(f"[step4] wrote {OUTPUTS / 'folds.pkl'}")


if __name__ == "__main__":
    main()
