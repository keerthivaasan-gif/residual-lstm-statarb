Residual-LSTM Statistical Arbitrage

A market-neutral, cross-sectional **statistical-arbitrage** research pipeline that
benchmarks  LSTM learning a *time-varying, nonlinear* mean-reversion on RMT-cleaned
idiosyncratic residuals vs the linear Avellaneda–Lee **s-score** baseline.



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


## Reading the result

A representative synthetic run:

```
        net_sharpe  gross_sharpe  ann_turnover  deflated_SR(P>0)  net_SR_x2.0
sscore       1.82          2.72          85.7             0.71         0.91
trees        1.15          2.76         157.2             0.40        -0.48
lstm         0.48          2.51         191.0             0.14        -1.55
```




## Layout

```
config.py            # one @dataclass with every knob (universe, L, k, costs, CV)
residlstm/           # the package (one module per step; see the table above)
scripts/             # step-numbered entry points + run_all.py
tests/               # causality, purge/embargo, target-horizon gates
outputs/             # artifacts, report.csv, equity_curves.png (gitignored)
```
