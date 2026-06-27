"""STEP 6 — transaction-cost model (Pitfall #4).

LSTM signals are high-turnover, so '1.5 gross / -0.2 net' is the default outcome
you want to discover on day one.  Hence we model three cost components and sweep a
multiplier *first*:

  * per-trade cost  -- ``cost_bps`` x daily turnover (commission + half-spread)
  * slippage        -- ``slippage_bps`` x daily turnover (market impact)
  * borrow / short  -- ``borrow_bps_annual`` accrued daily on short notional;
                       the juiciest short signals are often hard-to-borrow, so
                       this is where adverse selection bites.

``net_returns`` nets a Book; ``cost_sweep`` reruns the netting across
``cfg.cost_multipliers`` so you see how fast the edge dies under pessimistic costs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import CFG
from residlstm.portfolio import Book


def net_returns(book: Book, cfg=CFG, multiplier: float = 1.0) -> pd.Series:
    """Daily net strategy return = gross - trading costs - borrow."""
    bps = 1e-4
    trade_cost = (cfg.cost_bps + cfg.slippage_bps) * bps * book.turnover * multiplier

    short_notional = book.positions.clip(upper=0).abs().sum(axis=1)
    borrow_daily = cfg.borrow_bps_annual * bps / cfg.ann_factor
    borrow_cost = short_notional * borrow_daily * multiplier

    return book.gross_ret - trade_cost - borrow_cost


def cost_sweep(book: Book, cfg=CFG) -> pd.DataFrame:
    """Net daily returns under each cost multiplier -> columns 'x1.0','x1.5',..."""
    cols = {f"x{m}": net_returns(book, cfg, m) for m in cfg.cost_multipliers}
    return pd.DataFrame(cols)
