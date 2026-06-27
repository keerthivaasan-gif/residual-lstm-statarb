"""STEP 7 — the head-to-head report (the deliverable).

    python scripts/step7_report.py

Prints and saves the net-Sharpe head-to-head of s-score vs trees vs LSTM on
identical residuals, with the Deflated Sharpe Ratio, cost-sweep Sharpes, the decay
curve, and the 'Sharpe > 3 => assume a bug' gate.  Saves a CSV + a text summary +
an equity-curve plot.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CFG, OUTPUTS
from residlstm.evaluate import evaluate_strategy, report_table

ORDER = ["sscore", "trees", "lstm"]   # the three-model ladder, in order


def main():
    with open(OUTPUTS / "books.pkl", "rb") as f:
        blob = pickle.load(f)
    books, returns_u = blob["books"], blob["returns_u"]

    reports = []
    names = [n for n in ORDER if n in books] + [n for n in books if n not in ORDER]
    for name in names:
        b = books[name]
        rep = evaluate_strategy(name, b["book"], b["net"], b["sweep"],
                                b["signal"], returns_u, CFG)
        reports.append(rep)

    table = report_table(reports)
    print("\n================ STEP 7: HEAD-TO-HEAD (net, out-of-sample) ============")
    print(table.to_string())
    print("\nDecay curve (gross Sharpe vs decision lag):")
    for r in reports:
        print(f"  {r.name:8s}: " + ", ".join(f"{k}={v:.2f}" for k, v in r.decay.items()))

    print("\nNotes:")
    print(f"  - Deflated Sharpe penalizes n_trials={CFG.n_trials} (multiple testing).")
    for r in reports:
        if r.bug_warning:
            print(f"  - !! {r.name}: net Sharpe {r.net_sharpe:.2f} > 3 -> ASSUME A BUG. "
                  f"Re-audit RMT causality / survivorship / label leakage.")
    print("  - A clean, honest result (even negative for the LSTM) is the deliverable.")

    # save artifacts
    table.to_csv(OUTPUTS / "report.csv")
    with open(OUTPUTS / "report.txt", "w") as f:
        f.write(table.to_string())

    # equity curves
    fig, ax = plt.subplots(figsize=(9, 5))
    for r, name in zip(reports, names):
        eq = (1 + books[name]["net"].fillna(0)).cumprod()
        ax.plot(eq.index, eq.values, label=f"{name} (SR={r.net_sharpe:.2f})")
    ax.set_title("Net cumulative return — head-to-head on identical residuals")
    ax.set_ylabel("growth of $1 (net)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUTS / "equity_curves.png", dpi=120)
    print(f"\n[step7] wrote {OUTPUTS/'report.csv'}, {OUTPUTS/'report.txt'}, "
          f"{OUTPUTS/'equity_curves.png'}")


if __name__ == "__main__":
    main()
