"""STEP 0 — build the point-in-time universe and clean price panel.

    python scripts/step0_build_universe.py [--source synthetic|yfinance]

Writes outputs/panel.pkl with the masked returns/prices and (synthetic only) the
ground-truth residual panel used by the causality tests and sanity checks.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CFG, OUTPUTS
from residlstm.data import universe


def main(source: str | None = None):
    src = source or CFG.source
    print(f"[step0] building universe from source={src!r}")

    true_resid = betas = sectors = None
    if src == "synthetic":
        from residlstm.data.synthetic import simulate
        panel = simulate(CFG)
        prices, returns, dollar_volume = panel.prices, panel.returns, panel.dollar_volume
        true_resid, betas, sectors = panel.true_residuals, panel.betas, panel.sectors
    elif src == "yfinance":
        from residlstm.data.loader_yf import load_yfinance
        prices, returns, dollar_volume = load_yfinance(CFG)
    else:
        raise SystemExit(f"unknown source {src!r}")

    mask = universe.build_membership(dollar_volume, prices, CFG)
    returns_u = universe.apply_membership(returns, mask)

    n_alive = int(mask.any(axis=0).sum())
    print(f"[step0] panel: {prices.shape[0]} days x {prices.shape[1]} names; "
          f"{n_alive} ever in-universe; avg daily breadth "
          f"{mask.sum(axis=1).mean():.1f}")

    artifact = {
        "source": src,
        "prices": prices, "returns": returns, "returns_u": returns_u,
        "dollar_volume": dollar_volume, "mask": mask,
        "true_residuals": true_resid, "betas": betas, "sectors": sectors,
    }
    out = OUTPUTS / "panel.pkl"
    with open(out, "wb") as f:
        pickle.dump(artifact, f)
    print(f"[step0] wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "yfinance"], default=None)
    main(**vars(ap.parse_args()))
