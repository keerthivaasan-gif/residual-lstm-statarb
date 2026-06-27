"""STEP 0 (synthetic path) — a factor-model price simulator with *known* ground truth.

Why synthetic data is the default
---------------------------------
`yfinance` only returns the tickers that survive to today, so any universe built
from it silently bakes in survivorship bias (the plan's Pitfall #2).  This
simulator instead generates a panel where we *control* everything:

    return_{i,t} = beta_i . factor_returns_t  +  idiosyncratic_residual_{i,t}

* The factors are a market mode plus a few sector modes (random walks).
* The idiosyncratic residual is a mean-reverting OU process with a *known* speed
  ``kappa`` -- this is the structure the s-score and the LSTM are supposed to find.
* Some names *delist* mid-sample (price -> NaN afterwards) and the panel is built
  to also allow new listings, so the point-in-time universe logic is exercised.

Because we keep the true residual panel, two things become unit-testable:
  1. RMT (Step 2) should recover residuals highly correlated with the truth.
  2. The s-score (Step 5) should recover a kappa near ``sim_kappa``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import CFG


@dataclass
class SyntheticPanel:
    """Container for one simulated market.  All frames are date x ticker."""
    prices: pd.DataFrame          # adjusted close (NaN once a name delists)
    returns: pd.DataFrame         # simple daily returns
    dollar_volume: pd.DataFrame   # price * volume (drives universe selection)
    true_residuals: pd.DataFrame  # ground-truth idiosyncratic residual returns
    betas: pd.DataFrame           # ticker x factor loadings (static, for reference)
    sectors: pd.Series            # ticker -> sector id


def simulate(cfg=CFG) -> SyntheticPanel:
    """Generate a reproducible synthetic market from ``cfg``."""
    rng = np.random.default_rng(cfg.seed)
    n, T, kf = cfg.n_names, cfg.sim_days, cfg.sim_n_factors

    dates = pd.bdate_range("2015-01-01", periods=T)
    tickers = [f"SYN{i:03d}" for i in range(n)]

    # ---- factor returns: 1 market mode + (kf-1) sector modes ----------------
    factor_vol = np.concatenate([
        [cfg.sim_market_vol],
        rng.uniform(cfg.sim_sector_vol_lo, cfg.sim_sector_vol_hi, kf - 1),
    ])
    factor_returns = rng.standard_normal((T, kf)) * factor_vol  # (T, kf)

    # ---- loadings: everyone loads on the market; one sector mode each -------
    sector_ids = rng.integers(1, kf, size=n)  # sector index in 1..kf-1
    betas = np.zeros((n, kf))
    betas[:, 0] = rng.uniform(0.8, 1.2, n)    # market beta
    for i in range(n):
        betas[i, sector_ids[i]] = rng.uniform(0.5, 1.0)

    systematic = factor_returns @ betas.T      # (T, n) systematic return component

    # ---- idiosyncratic OU residual (the tradable mean-reversion) ------------
    # The mean-reverting object is the idiosyncratic *price* residual Y (OU level);
    # the idiosyncratic *return* is its increment dY.  When Y is rich (high), the
    # next return is negative -> reversion.  This is the Avellaneda-Lee picture:
    # the cumulative residual (= Y) is OU, so the s-score fit on the cumulative
    # residual recovers kappa.  (Making the return itself OU would be momentum.)
    kappa, ovol = cfg.sim_kappa, cfg.sim_resid_vol
    level = np.zeros((T, n))                    # Y: idiosyncratic price residual
    shocks = rng.standard_normal((T, n)) * ovol
    for t in range(1, T):
        level[t] = (1.0 - kappa) * level[t - 1] + shocks[t]
    resid = np.diff(level, axis=0, prepend=0.0)  # idiosyncratic return dY

    returns = systematic + resid               # (T, n) total return

    # ---- build price & volume, then apply delistings ------------------------
    prices = 100.0 * np.cumprod(1.0 + returns, axis=0)
    base_vol = rng.uniform(1e6, 5e7, n)        # heterogeneous liquidity
    vol_noise = np.exp(rng.standard_normal((T, n)) * 0.3)
    volume = base_vol[None, :] * vol_noise
    dollar_volume = prices * volume

    prices = pd.DataFrame(prices, index=dates, columns=tickers)
    returns = pd.DataFrame(returns, index=dates, columns=tickers)
    dollar_volume = pd.DataFrame(dollar_volume, index=dates, columns=tickers)
    true_resid = pd.DataFrame(resid, index=dates, columns=tickers)

    # delist a fraction of names at random dates in the back half of the sample
    n_delist = int(cfg.sim_delist_frac * n)
    doomed = rng.choice(n, size=n_delist, replace=False)
    for j in doomed:
        d = rng.integers(T // 2, T)            # death day
        for frame in (prices, returns, dollar_volume, true_resid):
            frame.iloc[d:, j] = np.nan

    betas_df = pd.DataFrame(betas, index=tickers,
                            columns=[f"F{k}" for k in range(kf)])
    sectors = pd.Series(sector_ids, index=tickers, name="sector")

    return SyntheticPanel(prices, returns, dollar_volume, true_resid, betas_df, sectors)
