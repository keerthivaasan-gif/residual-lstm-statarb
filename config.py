"""Single source of truth for the residual-LSTM stat-arb pipeline.

Every step-script imports `CFG` from here so that the universe size, sequence
length, label horizon, walk-forward geometry and cost assumptions are defined in
exactly one place.  Change a number here and the whole pipeline follows.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

# --- where artifacts live -------------------------------------------------
ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)


@dataclass
class Config:
    # ---- Step 0: universe & data ----------------------------------------
    source: str = "synthetic"          # "synthetic" | "yfinance"
    n_names: int = 60                  # tradable cross-section size (synthetic)
    universe_top_n: int = 120          # point-in-time top-N by dollar-volume
    reconstitute_freq: str = "ME"      # monthly universe rebuild (pandas alias)
    seed: int = 7

    # synthetic-simulator knobs (also the ground truth for the unit tests).
    # Factors are deliberately strong relative to the idiosyncratic shock so that
    # removing them measurably *improves* recovery of the idiosyncratic residual --
    # otherwise the RMT step would have nothing useful to strip.
    sim_days: int = 1400
    sim_n_factors: int = 4             # 1 market + 3 sector eigenmodes
    sim_market_vol: float = 0.030      # market-factor daily vol
    sim_sector_vol_lo: float = 0.015   # sector-factor daily vol (range)
    sim_sector_vol_hi: float = 0.020
    sim_kappa: float = 0.06            # idiosyncratic OU mean-reversion speed (per day)
    sim_resid_vol: float = 0.008       # idiosyncratic shock vol
    sim_delist_frac: float = 0.15      # fraction of names that die mid-sample

    # ---- Step 2: factor removal (RMT) -----------------------------------
    rmt_window: int = 252              # trailing days for covariance estimation
    rmt_top_k: int = 4                 # market+sector eigenmodes projected out
    # [#3] conditional / characteristic factor cleaning on top of statistical RMT:
    # each day, cross-sectionally regress the RMT residual on causal characteristics
    # (size, momentum, volatility) and keep the residual -> a poor-man's IPCA /
    # conditional latent-factor model (Pelger et al.).
    # EMPIRICAL NOTE: on our universes this same-day OLS neutralization OVER-STRIPS
    # signal (ablation: s-score gross Sharpe 0.21->0.05 when ON), so it is OFF by
    # default. A proper jointly-estimated IPCA would likely help; the crude version
    # hurts. Flip to True to reproduce the ablation.
    conditional_factors: bool = False
    char_mom_window: int = 60          # trailing window for the momentum characteristic
    char_vol_window: int = 20          # trailing window for the volatility characteristic
    seq_len: int = 20                  # L: residual-path lookback for the sequence model
    detrend_window: int = 63           # rolling window to z-score the cumulative residual
                                       #   (de-drifts the level the LSTM/trees see, the way
                                       #   the s-score re-estimates a local mean each window)
    use_vol_channel: bool = True       # add residual-vol channel to the tensor
    use_volume_channel: bool = True    # add volume-ratio channel to the tensor

    # ---- Step 3: target --------------------------------------------------
    label_horizon: int = 5             # k: predict the k-day-forward residual change
    target_kind: str = "zscore"        # "zscore" | "raw" | "sign"

    # ---- Step 4: cross-validation ---------------------------------------
    cv_train_days: int = 252           # ~1y train window
    cv_test_days: int = 126            # ~2q test window (more OOS days, more folds)
    cv_embargo: int = 5                # embargo days after each test block
    retrain: str = "per_window"        # "per_window" | "quarterly" (LSTM cadence)

    # ---- Step 6: portfolio & costs --------------------------------------
    weighting: str = "proportional"    # "proportional" (weight ∝ z, capped) | "decile"
    decile: float = 0.10               # long top decile / short bottom decile (decile mode)
    per_name_cap: float = 0.10         # max |weight| per name (proportional mode)
    beta_neutral: bool = True
    sector_neutral: bool = False
    exec_lag: int = 1                  # signal at close t -> trade at t+1
    cost_bps: float = 5.0              # round-trip-ish per-trade cost (bps of notional)
    borrow_bps_annual: float = 50.0    # short borrow fee (bps/yr) on short notional
    slippage_bps: float = 2.0          # turnover-scaled slippage (bps)
    cost_multipliers: tuple = (1.0, 1.5, 2.0)  # Pitfall-4 sensitivity sweep

    # ---- Step 7: evaluation ---------------------------------------------
    ann_factor: int = 252              # trading days per year for annualization
    n_trials: int = 12                 # configs tried -> deflated-Sharpe penalty

    # ---- yfinance real-data path ----------------------------------------
    # ~190 large/mid-cap names listed well before yf_start, for a wide, low-NaN
    # cross-section (more breadth -> more sequences for the deep model, #4).
    # NOTE: still survivorship-biased (only names alive today); the real fix is
    # point-in-time data (CRSP/Sharadar). Bias is logged at load time.
    yf_tickers: tuple = (
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "JNJ", "V", "PG",
        "HD", "MA", "BAC", "DIS", "ADBE", "CRM", "NFLX", "KO", "PEP", "XOM",
        "CVX", "WMT", "CSCO", "INTC", "ORCL", "QCOM", "T", "PFE", "MRK", "ABT",
        "TMO", "COST", "AVGO", "ACN", "NKE", "MCD", "TXN", "HON", "UNH", "LIN",
        "DHR", "NEE", "PM", "IBM", "AMGN", "LOW", "SBUX", "CAT", "GS", "MS",
        "BLK", "AXP", "BKNG", "GILD", "MDT", "ISRG", "ADP", "CB", "MMC", "SYK",
        "C", "WFC", "USB", "PNC", "SCHW", "DE", "LMT", "RTX", "GE", "BA",
        "MDLZ", "CL", "EL", "GM", "F", "DUK", "SO", "TGT", "FDX", "UPS",
        "EMR", "ITW", "AON", "ICE", "CME", "APD", "SHW", "ECL", "NSC", "WM",
        "ABBV", "ADI", "AEP", "AFL", "AIG", "ALL", "AMAT", "AMT", "AMP", "BAX",
        "BDX", "BIIB", "BK", "BMY", "BSX", "CARR", "CCI", "CHTR", "CMCSA", "CMG",
        "CMI", "COF", "COP", "CSX", "CTAS", "CTSH", "D", "DD", "DOW", "DVN",
        "EA", "EBAY", "EOG", "EQIX", "EW", "EXC", "FCX", "FIS", "FISV", "GD",
        "GIS", "GLW", "HAL", "HCA", "HPQ", "HUM", "JCI", "KHC", "KLAC", "KMB",
        "KMI", "KR", "LRCX", "LVS", "MAR", "MCK", "MCO", "MET", "MMM", "MO",
        "MPC", "MU", "NOC", "NOW", "NUE", "ODFL", "OXY", "PAYX", "PCAR", "PGR",
        "PLD", "PRU", "PSA", "PSX", "REGN", "ROK", "ROP", "ROST", "SLB",
        "SPG", "SPGI", "STZ", "TEL", "TFC", "TJX", "TRV", "VLO", "VRTX", "VZ",
        "WBA", "WDAY", "WELL", "WMB", "YUM", "ZTS",
    )
    yf_start: str = "2015-01-01"
    yf_end: str = "2024-12-31"

    def to_dict(self) -> dict:
        return asdict(self)


CFG = Config()
