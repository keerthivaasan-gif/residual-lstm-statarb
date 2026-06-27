Residual-LSTM Statistical Arbitrage

A market-neutral, cross-sectional **statistical-arbitrage** research pipeline that
tests one question honestly:

> Does an LSTM learning a *time-varying, nonlinear* mean-reversion on RMT-cleaned
> idiosyncratic residuals beat the linear Avellaneda–Lee **s-score** baseline —
> **net of costs**, out of sample?

It is organized on the Quantt *ML for Trading* 8-step framework. Each step is a
module in `residlstm/` and a runnable script in `scripts/`. The whole thing runs
out of the box on a **synthetic factor market** with known ground truth (no data
download, no survivorship bias), and a `yfinance` path is included for real prices.

## The three-baseline ladder (why there's an XGBoost-style model)

The models are a deliberate ladder that isolates *what* (if anything) the LSTM buys:

| model | nonlinear? | sequence memory? | role |
|-------|:---------:|:----------------:|------|
| **s-score** (Avellaneda–Lee) | no | — | the domain baseline: fixed-κ OU mean reversion |
| **trees** (HistGradientBoosting) | yes | **no** | nonlinear but treats lags as flat features |
| **panel-LSTM** (Conv1D→BiLSTM) | yes | **yes** | nonlinear **and** uses the residual path |

If the LSTM beats the s-score but **not** the trees, the win was generic
nonlinearity, not memory — and the LSTM isn't justified. That comparison is the
deliverable.

## Quickstart

```bash
cd /Users/keerthivaasan/Desktop/projects
source project1/.venv/bin/activate         # env with numpy/pandas/sklearn/scipy/matplotlib
pip install -r project3/requirements.txt   # (pytest for tests; tensorflow optional)

# the no-look-ahead causality gate + purge/embargo + target tests
pytest project3/tests/ -q

# Step 0 -> Step 7 end-to-end on synthetic data
python project3/scripts/run_all.py
# -> prints the head-to-head table; writes outputs/report.csv + equity_curves.png
```

Real-data run (survivorship-biased — logged as such):
```bash
python project3/scripts/run_all.py --source yfinance
```



| Step | What | Module(s) | Script |
|---|---|---|---|
| 0 | Point-in-time universe; survivorship-safe data | `data/synthetic.py`, `data/universe.py`, `data/loader_yf.py` | `step0_build_universe.py` |
| 1 | Framing: predict the forward **residual**, ranked | (design — see this README) | — |
| 2 | **Causal RMT** factor removal + sequence tensor | `features/rmt.py`, `features/sequences.py` | `step2_compute_residuals.py` |
| 3 | Target = k-day-forward residual change | `targets.py` | `step3_make_targets.py` |
| 4 | **Purged + embargoed** walk-forward CV | `cv.py` | `step4_make_splits.py` |
| 5 | s-score, trees, panel-LSTM | `models/{sscore,trees,panel_lstm,harness}.py` | `step5_run_baselines.py`, `step5_train_lstm.py` |
| 6 | Signal→position, neutralize, costs | `portfolio.py`, `costs.py` | `step6_backtest.py` |
| 7 | Net Sharpe head-to-head + Deflated Sharpe | `evaluate.py` | `step7_report.py` |
| 8 | Deploy (out of scope) | — | — |

Scripts chain through artifacts in `outputs/` (pickled panels), so you can run them
one at a time or all via `run_all.py`.



* **[GAP — RMT causality, the prime look-ahead suspect].** The eigenvectors/betas
  used to strip factors are estimated on a **trailing window ending at t−1 only**;
  a full-sample PCA would leak the future into *every* residual, which `shift(1)`
  cannot catch. `tests/test_rmt_causality.py` proves the residual at `t` is
  bit-identical whether computed on the full panel or the panel truncated at `t`.
* **[GAP — purge + embargo].** Overlapping k-day labels leak across adjacent
  train/test days. `cv.py` drops training samples whose label window reaches the
  test window (purge) and skips a few days after it (embargo). Verified by
  `tests/test_purge_embargo.py`.
* **[EXTENDS]** beta-neutral (not just dollar-neutral) portfolio; Deflated Sharpe
  over the number of configs tried; a cost **sensitivity sweep** done first.

## Reading the result

A representative synthetic run:

```
        net_sharpe  gross_sharpe  ann_turnover  deflated_SR(P>0)  net_SR_x2.0
sscore       1.82          2.72          85.7             0.71         0.91
trees        1.15          2.76         157.2             0.40        -0.48
lstm         0.48          2.51         191.0             0.14        -1.55
```

All three capture the reversion **gross**. The stable, repeatable finding is in the
**cost sweep**: the high-turnover ML signals (≈150–190 ann. turnover vs ≈85 for the
s-score) bleed their edge as costs rise and go **negative at ×2.0**, while the
linear s-score stays positive — it is the most cost-robust. The exact net ranking of
s-score vs trees varies run to run (the sklearn models carry internal randomness and
are not globally seeded), but the LSTM/trees never *durably* clear the s-score once
costs bite. That "1.5 gross / −0.2 net" outcome, stated honestly, is exactly the
kind of result the framework says is worth more than an unbelievable number. (Your
numbers will differ with config and seed.)

## Honest caveats baked into the code

* **High synthetic Sharpe is expected**, because the synthetic signal is clean and
  noise-free. The "Sharpe > 3 ⇒ assume a bug" gate in `evaluate.py` is calibrated
  for **real** data; on synthetic it is a smoke alarm for accidental leakage.
* **LSTM backend.** If TensorFlow is installed, `panel_lstm.py` trains the real
  Conv1D→BiLSTM. If not, it falls back to a scikit-learn MLP (clearly labeled
  `lstm(mlp-fallback)`) so the pipeline still runs end-to-end — but the fallback is
  **not** a sequence model, so install `tensorflow` to evaluate the real LSTM.
* **Survivorship.** The synthetic market includes delistings and is bias-free; the
  `yfinance` path cannot see delisted names and logs that caveat loudly.

## Layout

```
config.py            # one @dataclass with every knob (universe, L, k, costs, CV)
residlstm/           # the package (one module per step; see the table above)
scripts/             # step-numbered entry points + run_all.py
tests/               # causality, purge/embargo, target-horizon gates
outputs/             # artifacts, report.csv, equity_curves.png (gitignored)
```
