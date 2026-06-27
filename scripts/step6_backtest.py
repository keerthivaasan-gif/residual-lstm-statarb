"""STEP 6 — portfolio construction + cost layer for every model signal.

    python scripts/step6_backtest.py

For each signal panel in outputs/signals/, builds a neutralized, lagged book and
nets it through the cost model (with the sensitivity sweep).  Writes
outputs/books.pkl for Step 7.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CFG, OUTPUTS
from residlstm.portfolio import build_book
from residlstm.costs import net_returns, cost_sweep

SIGDIR = OUTPUTS / "signals"


def main():
    with open(OUTPUTS / "panel.pkl", "rb") as f:
        panel = pickle.load(f)
    returns_u = panel["returns_u"]

    # Fair head-to-head: every model is scored on the SAME out-of-sample test days
    # (the supervised models only predict test folds; the s-score predicts daily,
    # so we restrict it to the same OOS dates for an apples-to-apples comparison).
    with open(OUTPUTS / "folds.pkl", "rb") as f:
        folds = pickle.load(f)
    oos_pos = sorted({p for _, test in folds for p in test.tolist()})
    oos_dates = returns_u.index[oos_pos]
    print(f"[step6] scoring all models on {len(oos_dates)} common OOS test days")

    # beta for beta-neutralization: trailing beta to the cross-sectional mean (market proxy)
    market = returns_u.mean(axis=1)
    beta = returns_u.apply(lambda col: col.cov(market) / market.var())
    sector = panel.get("sectors")

    books = {}
    for sig_path in sorted(SIGDIR.glob("*.pkl")):
        name = sig_path.stem
        with open(sig_path, "rb") as f:
            signal = pickle.load(f)
        signal = signal.loc[signal.index.isin(oos_dates)]   # restrict to common OOS days
        book = build_book(signal, returns_u.loc[signal.index],
                          beta=beta, sector=sector, cfg=CFG)
        net = net_returns(book, CFG)
        sweep = cost_sweep(book, CFG)
        books[name] = {"book": book, "net": net, "sweep": sweep, "signal": signal}
        print(f"[step6] {name:8s}: gross days={int((book.gross_ret!=0).sum())}, "
              f"mean daily turnover={book.turnover.mean():.3f}")

    with open(OUTPUTS / "books.pkl", "wb") as f:
        pickle.dump({"books": books, "returns_u": returns_u}, f)
    print(f"[step6] wrote {OUTPUTS / 'books.pkl'}")


if __name__ == "__main__":
    main()
