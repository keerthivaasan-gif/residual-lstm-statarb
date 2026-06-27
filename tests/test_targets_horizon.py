"""Step 3 gate — the target is the k-day FORWARD residual change, aligned to t.

target[t] must equal sum(residual[t+1 .. t+k]) and must use no information at or
before t (it is purely forward), so that it pairs correctly with the purge logic.
"""
import numpy as np
import pandas as pd
import pytest

from config import Config
from residlstm.targets import make_targets


def _toy_residuals():
    dates = pd.bdate_range("2020-01-01", periods=20)
    # one name with a simple ramp so the forward sum is easy to check
    r = pd.DataFrame({"A": np.arange(1, 21, dtype=float)}, index=dates)
    return r


def test_forward_sum_matches_definition():
    cfg = Config(label_horizon=3, target_kind="raw")
    r = _toy_residuals()
    tgt = make_targets(r, cfg)
    # at t=5 (value 6), target = residuals at t+1,t+2,t+3 = 7+8+9 = 24
    assert tgt["A"].iloc[5] == pytest.approx(24.0)


def test_target_is_forward_only():
    cfg = Config(label_horizon=3, target_kind="raw")
    r = _toy_residuals()
    tgt = make_targets(r, cfg)
    # the last k targets cannot be formed (no future) -> NaN
    assert tgt["A"].iloc[-cfg.label_horizon:].isna().all()


def test_zscore_is_cross_sectional():
    cfg = Config(label_horizon=2, target_kind="zscore")
    dates = pd.bdate_range("2020-01-01", periods=10)
    r = pd.DataFrame(np.random.default_rng(0).standard_normal((10, 5)),
                     index=dates, columns=list("ABCDE"))
    tgt = make_targets(r, cfg)
    row = tgt.iloc[3].dropna()
    if len(row) > 1:
        assert abs(row.mean()) < 1e-9  # demeaned cross-sectionally
