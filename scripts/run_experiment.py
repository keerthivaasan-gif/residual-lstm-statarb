
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import step0_build_universe as s0
import step2_compute_residuals as s2
import step3_make_targets as s3
import step4_make_splits as s4
import step5_run_baselines as s5b
import step5_train_lstm as s5l
import step6_backtest as s6
import step7_report as s7


def main(source=None):
    print("=" * 72)
    print("RESIDUAL STAT-ARB — full experiment (baselines + LSTM + end-to-end head)")
    print("=" * 72)
    s0.main(source)
    s2.main()
    s3.main()
    s4.main()
    s5b.main()
    s5l.main()
    s6.main()
    s7.main()
    print("\n[run_experiment] done. See project3/outputs/report.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "yfinance"], default=None)
    main(**vars(ap.parse_args()))
