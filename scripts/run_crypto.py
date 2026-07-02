
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
# --- crypto-appropriate configuration (set BEFORE importing the step modules) ---
config.OUTPUTS = ROOT / "outputs_crypto"
config.OUTPUTS.mkdir(exist_ok=True)
C = config.CFG
C.source = "yfinance"
C.ann_factor = 365                 # 7-day trading week
C.rmt_top_k = 2                    # BTC "market" + one sector mode; thin cross-section
C.rmt_window = 180                 # shorter history available
C.conditional_factors = False     # don't over-strip a ~25-name universe
C.universe_top_n = 25
C.cv_train_days = 252
C.cv_test_days = 126
C.cv_embargo = 5
C.yf_start = "2019-01-01"
C.yf_end = "2024-12-31"
C.borrow_bps_annual = 0.0          # crypto shorts via perps; no equity-style borrow
C.yf_tickers = (
    "BTC-USD", "ETH-USD", "XRP-USD", "LTC-USD", "BCH-USD", "EOS-USD",
    "BNB-USD", "XLM-USD", "TRX-USD", "ADA-USD", "XMR-USD", "ETC-USD",
    "DASH-USD", "ZEC-USD", "NEO-USD", "DOGE-USD", "LINK-USD", "ATOM-USD",
    "VET-USD", "BAT-USD", "ALGO-USD", "DOT-USD", "SOL-USD", "AVAX-USD",
    "MATIC-USD", "FIL-USD", "AAVE-USD", "UNI-USD", "MKR-USD", "COMP-USD",
)

# import step modules AFTER patching config so they bind the new OUTPUTS
import step0_build_universe as s0
import step2_compute_residuals as s2
import step3_make_targets as s3
import step4_make_splits as s4
import step5_run_baselines as s5b
import step5_train_lstm as s5l
import step6_backtest as s6
import step7_report as s7


def main():
    print("=" * 72)
    print("CRYPTO experiment — different dataset, same models")
    print("=" * 72)
    s0.main("yfinance")
    s2.main(); s3.main(); s4.main()
    s5b.main(); s5l.main()
    s6.main(); s7.main()
    print("\n[run_crypto] done. See project3/outputs_crypto/report.csv")


if __name__ == "__main__":
    main()
