"""Step 4 gate — no training label window may intersect a test window.

With a k-day label, a training sample at day t "sees" residuals through t+k.  The
purged splitter must therefore drop any training day whose [t, t+k] window reaches
into the test block, and must leave an embargo gap after the test block.
"""
import numpy as np

from config import Config
from residlstm.cv import purged_walk_forward


def test_purge_gap_respects_label_horizon():
    cfg = Config(cv_train_days=200, cv_test_days=40, label_horizon=5, cv_embargo=3)
    n_days = 600
    for fold in purged_walk_forward(n_days, cfg):
        if len(fold.train_days) == 0:
            continue
        last_train = fold.train_days[-1]
        first_test = fold.test_days[0]
        # the last training day's label window must close before the test opens
        assert last_train + cfg.label_horizon <= first_test, (
            f"label window of train day {last_train} leaks into test "
            f"starting {first_test}"
        )


def test_no_train_test_overlap():
    cfg = Config(cv_train_days=200, cv_test_days=40, label_horizon=5, cv_embargo=3)
    for fold in purged_walk_forward(600, cfg):
        assert set(fold.train_days).isdisjoint(set(fold.test_days))


def test_embargo_between_folds():
    cfg = Config(cv_train_days=150, cv_test_days=30, label_horizon=5, cv_embargo=7)
    folds = list(purged_walk_forward(700, cfg))
    for prev, nxt in zip(folds, folds[1:]):
        gap = nxt.train_days[0] - prev.test_days[-1]
        assert gap >= cfg.cv_embargo, f"embargo too small: gap={gap}"


def test_folds_are_time_ordered():
    cfg = Config(cv_train_days=150, cv_test_days=30, label_horizon=5, cv_embargo=7)
    folds = list(purged_walk_forward(700, cfg))
    assert len(folds) >= 2
    for prev, nxt in zip(folds, folds[1:]):
        assert nxt.test_days[0] > prev.test_days[0]
