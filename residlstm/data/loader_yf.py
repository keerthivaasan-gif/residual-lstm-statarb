"""STEP 0 (real-data path) — yfinance OHLCV loader.

SURVIVORSHIP CAVEAT (logged loudly at call time): yfinance returns only the
tickers that still trade today, so a universe built from a fixed ticker list is
survivorship-biased -- it cannot see names that delisted.  The plan flags this as
Pitfall #2.  Use this path to relate the pipeline to real prices, but treat its
Sharpe as optimistic; the synthetic path is the bias-free benchmark.

Returns the same objects as the synthetic path (minus ground-truth residuals) so
every downstream step is source-agnostic.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from config import CFG


def load_yfinance(cfg=CFG):
    """Download split/div-adjusted daily OHLCV; return (prices, returns, dollar_volume)."""
    try:
        import yfinance as yf
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "yfinance is not installed. `pip install yfinance` or use "
            "source='synthetic' (the default)."
        ) from e

    warnings.warn(
        "SURVIVORSHIP BIAS: yfinance only serves tickers alive today. This "
        "universe omits delisted names; backtest Sharpe is optimistic by "
        "~0.5-1.0 (Pitfall #2). The synthetic path is the bias-free benchmark.",
        stacklevel=2,
    )

    tickers = list(cfg.yf_tickers)
    raw = yf.download(tickers, start=cfg.yf_start, end=cfg.yf_end,
                      auto_adjust=True, progress=False)

    # yfinance returns a column MultiIndex (field, ticker)
    close = raw["Close"].copy()
    volume = raw["Volume"].copy()

    close = close.dropna(axis=1, how="all").sort_index()
    volume = volume.reindex_like(close)

    returns = close.pct_change()
    dollar_volume = close * volume

    return close, returns, dollar_volume
