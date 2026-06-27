"""BUILD-ORDER #1 GATE — the causal-RMT no-look-ahead test.

The plan's hardest correctness claim: the residual at date t must use *no data
after t-1* for its loadings, and no data after t at all.  We prove it directly:
the residual at t computed on the FULL panel must be bit-identical to the residual
at t computed on the panel TRUNCATED right after t.  If RMT (e.g. a full-sample
PCA) leaked the future, these would differ.
"""
import numpy as np
import pandas as pd
import pytest

from config import Config
from residlstm.data.synthetic import simulate
from residlstm.features.rmt import compute_residuals, marchenko_pastur_edge


@pytest.fixture(scope="module")
def panel():
    cfg = Config(sim_days=800, n_names=50, rmt_window=252, sim_delist_frac=0.0)
    return cfg, simulate(cfg)


def test_residual_is_causal(panel):
    cfg, sim = panel
    returns = sim.returns
    full = compute_residuals(returns, cfg)

    # pick several test dates well past the warmup window
    test_positions = [cfg.rmt_window + 5, cfg.rmt_window + 50, len(returns) - 1]
    for t in test_positions:
        truncated = compute_residuals(returns.iloc[: t + 1], cfg)
        a = full.iloc[t].to_numpy()
        b = truncated.iloc[t].to_numpy()
        ok = ~np.isnan(a) & ~np.isnan(b)
        assert ok.sum() > 0, f"no residuals at t={t}"
        np.testing.assert_allclose(
            a[ok], b[ok], rtol=1e-9, atol=1e-12,
            err_msg=f"residual at t={t} changed when future data was removed "
                    f"-> look-ahead leak in the RMT step",
        )


def test_warmup_is_nan(panel):
    cfg, sim = panel
    resid = compute_residuals(sim.returns, cfg)
    # before the first full trailing window there can be no residual
    assert resid.iloc[: cfg.rmt_window].isna().all().all()


def _corr_with_truth(resid, truth):
    a, b = resid.to_numpy().ravel(), truth.to_numpy().ravel()
    ok = ~np.isnan(a) & ~np.isnan(b)
    return np.corrcoef(a[ok], b[ok])[0, 1]


def test_recovers_true_residuals(panel):
    cfg, sim = panel
    resid = compute_residuals(sim.returns, cfg)
    corr = _corr_with_truth(resid, sim.true_residuals)
    # RMT should strip the factors and leave residuals close to the truth
    assert corr > 0.8, f"recovered residuals only corr {corr:.2f} with truth"


def test_rmt_beats_raw_returns(panel):
    """The whole point of the RMT step: removing factors must improve recovery of
    the idiosyncratic residual versus using the raw return directly."""
    cfg, sim = panel
    resid = compute_residuals(sim.returns, cfg)
    raw_corr = _corr_with_truth(sim.returns, sim.true_residuals)
    rmt_corr = _corr_with_truth(resid, sim.true_residuals)
    assert rmt_corr > raw_corr + 0.05, (
        f"RMT ({rmt_corr:.2f}) did not beat raw returns ({raw_corr:.2f}) -- "
        f"factor removal is adding no value"
    )


def test_mp_edge_monotonic():
    # more observations per variable -> tighter (lower) noise edge
    assert marchenko_pastur_edge(50, 100) > marchenko_pastur_edge(50, 500)
