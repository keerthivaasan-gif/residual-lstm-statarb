"""Orchestrate the whole pipeline end-to-end (Step 0 -> Step 7).

    python scripts/run_all.py [--source synthetic|yfinance]

Runs every step in order on one process so a fresh checkout produces the
head-to-head report with a single command.
"""
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
    print("RESIDUAL-LSTM STAT-ARB — full pipeline")
    print("=" * 72)
    s0.main(source)
    s2.main()
    s3.main()
    s4.main()
    s5b.main()
    s5l.main()
    s6.main()
    s7.main()
    print("\n[run_all] done. See project3/outputs/ for report.csv and equity_curves.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "yfinance"], default=None)
    main(**vars(ap.parse_args()))
