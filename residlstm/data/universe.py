"""STEP 0 — point-in-time universe construction.

We pick the top-N names by *trailing* dollar-volume and rebuild the membership
monthly.  Two properties matter for honesty:

* **Trailing only.**  Membership on a reconstitution date uses dollar-volume up to
  that date -- never future liquidity.
* **Survivorship-safe.**  A name that delists is eligible only while it is alive;
  it is held in the membership for the windows it actually traded and drops out
  naturally once its price goes NaN.  Building from "today's tickers" would be the
  trap (Pitfall #2).

``build_membership`` returns a boolean date x ticker mask: True == in-universe and
tradable on that date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import CFG


def build_membership(dollar_volume: pd.DataFrame, prices: pd.DataFrame,
                     cfg=CFG) -> pd.DataFrame:
    """Boolean date x ticker membership mask (point-in-time, monthly rebuild)."""
    dv, px = dollar_volume, prices
    top_n = min(cfg.universe_top_n, dv.shape[1])

    # trailing-window average dollar-volume (the liquidity score)
    adv = dv.rolling(cfg.rmt_window, min_periods=cfg.rmt_window // 2).mean()

    mask = pd.DataFrame(False, index=dv.index, columns=dv.columns)

    # reconstitution dates: last business day of each period
    recon_dates = dv.resample(cfg.reconstitute_freq).last().index
    recon_dates = [d for d in recon_dates if d in dv.index]

    selections = {}
    for d in recon_dates:
        scores = adv.loc[d]
        # eligible = alive (price present) AND has a liquidity score
        alive = px.loc[d].notna()
        scores = scores.where(alive)
        chosen = scores.nlargest(top_n).index
        selections[d] = chosen

    # forward-fill each month's selection until the next reconstitution
    recon_index = pd.Index(recon_dates)
    for i, d in enumerate(recon_dates):
        end = recon_dates[i + 1] if i + 1 < len(recon_dates) else dv.index[-1]
        window = (dv.index >= d) & (dv.index <= end)
        cols = selections[d]
        block = mask.loc[window, cols]
        # only keep days where the name is actually alive
        alive_block = px.loc[window, cols].notna()
        mask.loc[window, cols] = alive_block.values

    return mask


def apply_membership(panel: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Mask a date x ticker panel to in-universe cells (others -> NaN)."""
    return panel.where(mask.reindex_like(panel).fillna(False))
