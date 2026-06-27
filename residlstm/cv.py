"""STEP 4 (+ the purge/embargo GAP fix) — purged, embargoed walk-forward CV.

The plan's correction to the tutorial:

    > walk_forward_split advances by test_size with no overlap *between folds*,
    > but it does NOT purge training samples whose label window overlaps the test
    > window, nor embargo the days just after.  With overlapping labels this leaks.

So each fold here:

  * PURGE  -- a training day t is dropped if its forward-label window [t, t+k]
    intersects the test window.  Concretely the train block ends ``label_horizon``
    days before the test block opens.
  * EMBARGO -- after the test block we skip ``embargo`` days before the next
    train block begins, so post-test information cannot bleed back.

This operates on the *time axis* (positional day indices).  Panel models then keep
only the samples whose date falls in the returned train / test day sets.
(López de Prado, *Advances in Financial ML*, Ch. 7.)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import CFG


@dataclass
class Fold:
    train_days: np.ndarray   # positional indices into the date axis
    test_days: np.ndarray


def purged_walk_forward(n_days: int, cfg=CFG):
    """Yield ``Fold`` objects over ``n_days`` time steps with purge + embargo."""
    train_n = cfg.cv_train_days
    test_n = cfg.cv_test_days
    k = cfg.label_horizon
    embargo = cfg.cv_embargo

    start = 0
    while start + train_n + test_n <= n_days:
        tr0, tr1 = start, start + train_n
        te0, te1 = tr1, tr1 + test_n

        # PURGE: training labels must close before the test window opens
        tr1_purged = tr1 - k
        train_days = np.arange(tr0, max(tr0, tr1_purged))
        test_days = np.arange(te0, te1)

        yield Fold(train_days=train_days, test_days=test_days)

        # EMBARGO: skip `embargo` days after the test block before the next train
        start = te1 + embargo


def split_dates(date_index: pd.DatetimeIndex, cfg=CFG):
    """Convenience: same folds but expressed as (train_dates, test_dates)."""
    for fold in purged_walk_forward(len(date_index), cfg):
        yield (date_index[fold.train_days], date_index[fold.test_days])
