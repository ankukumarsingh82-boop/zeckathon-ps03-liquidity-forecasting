# Zeckathon PS 03: Liquid & Overnight fund net-flow forecasting and cash-buffer sizing

This project forecasts AMFI category-level monthly net flows (₹ crore) for **Liquid** and **Overnight** funds for months T+1 to T+3. Each forecast has a point value and 80%/95% prediction intervals. It is backtested with a leakage-free rolling origin over the last 24 AMFI releases (Oct-2024 to Sep-2026, refit at every origin), and the forecasts feed a cash-buffer rule that is costed against the naive rule of holding last year's same-month outflow.

## Run it locally (nothing is deployed; the app binds to 127.0.0.1 only; no API keys)
```bash
cd zeckathon-ps03
python3.13 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt        # pinned; requirements-lock.txt is the full `pip freeze`
pip install -e . --no-deps             # makes `python -m liqfc` importable

python -m liqfc run                    # ingest -> validate -> 24-origin backtest -> metrics -> decision -> drivers
                                       #   -> live forecast -> outputs/*.csv, reports/tables_*.md, MANIFEST (~2.5 min)
python -m liqfc run --skip-backtest    # same, reusing outputs/forecasts_backtest_full.csv (~15 s)
python -m liqfc validate               # ingest + hard data checks only
python -m liqfc refresh                # download new official data, then a full run (needs internet; 3-10 min)
python -m liqfc refresh --no-rbi       #   skip the RBI press-release scan (fastest)
python -m liqfc app                    # interactive dashboard on http://127.0.0.1:8501 (--port N to change)
python -m liqfc test                   # pytest: 39 tests, including a headless launch smoke test (extra args pass through, e.g. -k app)
```
`make install | run | run-fast | validate | refresh | app | test | smoke` wraps the same commands (`make app PORT=8600`). The Makefile was not exercised on the build machine because it had no `make`; the `python -m liqfc …` commands are the verified path.
With no subcommand, `python -m liqfc` runs `run`.

Check that the app is serving:
```bash
curl -s http://127.0.0.1:8501/_stcore/health     # -> ok
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8501/   # -> 200 (Streamlit HTML shell)
```

### The interactive dashboard (`app/streamlit_app.py`)
It reads only `outputs/*.csv` and `outputs/*.json`, never the raw data or the models. If an output is missing, it tells you to run `python -m liqfc run`. Tabs:
1. **Forecast & buffer**: the last 24 months of actuals plus the 3-month forecast with 80%/95% bands per category, and the recommended buffer for the chosen τ.
2. **Backtest vs baseline**: MAPE, WAPE, pinball and skill against seasonal-naive by horizon; a fan chart of backtest forecasts vs actuals; the nested model selection.
3. **Coverage**: empirical 80%/95% coverage with Wilson intervals.
4. **Drivers**: WACR vs T-bill, net LAF liquidity, Nifty, the spread, the HAC regression and the LightGBM ablation.
5. **Decision cost**: the sidebar sliders for **τ**, **decision horizon h**, **spread** (realised FBIL 3M − WACR with an adjustable floor, or a constant) and the **shortfall penalty** recompute idle, shortfall and total cost and breaches over the 24 backtest months, using the backtest quantiles, actual flows and monthly spreads from the CSVs. They are shown against the last-year-same-month baseline, with a cost-vs-τ curve. At τ=0.75/0.95 with the default settings the numbers equal `outputs/decision_costs.csv` (a test checks this).
6. **Data & refresh**: file timestamps, the validation summary and the last refresh log. The **Refresh data + rerun pipeline** button runs `python -m liqfc refresh` and **Rerun pipeline only** runs `python -m liqfc run`, both as local subprocesses; the app reloads afterwards.

### What `refresh` fetches (official, public, keyless)
| Source | URL pattern | Rule |
|---|---|---|
| AMFI monthly report | `amfiindia.com/spages/am{mon}{yyyy}repo.xls` | months after the newest workbook on disk, up to last month; kept only if the bytes are a real OLE2/ZIP workbook, otherwise logged REJECTED in `data/raw/amfi/amfi_dl_log.txt` |
| NSE Nifty 50 | `nsearchives.nseindia.com` index close CSVs | month-ends after the newest file on disk |
| FBIL T-bill | FBIL archive endpoint | previous and current month re-fetched |
| RBI Money Market Operations | `rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx?prid=` | press-release ids after the highest one already in `data/raw/rbi_mmo/index.txt`; only "Money Market Operations as on …" pages kept |

The third-party DBIE WACR/LAF mirror used for history up to 2025 is **not** refreshed, because it is not an official endpoint; later days come from the official RBI releases above. Each refresh writes `outputs/refresh_log.json`.

### Lower-level scripts (still work)
`python scripts/run_pipeline.py` (a thin wrapper over `liqfc.pipeline`), `scripts/report_tables.py`, `scripts/build_manifest.py`. The original one-off fetchers are `scripts_fetch_nse.py`, `scripts_fetch_fbil.py` and `scripts_fetch_mmo.sh`.

## Layout
| Path | What it holds |
|---|---|
| `config.yaml` | Every setting, frozen before the backtest (models, quantiles, LGBM hyper-parameters, decision rule, nested selection windows) |
| `src/liqfc/amfi.py` | AMFI loader: Excel engine picked by magic bytes, non-workbooks (404 HTML) rejected and logged, pre-2019 sponsor-wise layout excluded, release dates taken from the "Released on" stamp (else month-end + 13 d), plus hard validation |
| `src/liqfc/macro.py` | Drivers: WACR (RBI), net LAF liquidity (RBI MMO), FBIL 3M T-bill, Nifty 50 month-end (NSE). `available_date` = month-end + 3 d |
| `src/liqfc/features.py` | Calendar features (advance-tax / quarter-end, FY-end, April, quarter start) and driver features |
| `src/liqfc/models.py` | Models: `snaive`, `snaive_ratio`, `ets_ratio`, `sarimax_cal` (classical) and `lgbm_q` (quantile LightGBM) plus its ablations |
| `src/liqfc/backtest.py` | `info_set` (as-of truncation), rolling-origin backtest, out-of-sample-error PIs (`*_oosq`) |
| `src/liqfc/metrics.py` | MAPE, median APE, sMAPE, WAPE, MAE, RMSE, MASE, pinball, scaled pinball, coverage with Wilson CI |
| `src/liqfc/decision.py` | Buffer policy, cost function, same-month-last-year baseline, block-bootstrap comparison |
| `src/liqfc/drivers.py` | HAC calendar/driver regression with SDF-2022 and MIBOR-2025 break dummies |
| `src/liqfc/dashboard.py` | Static one-page HTML, built from the CSVs in `outputs/` |
| `src/liqfc/__main__.py` | CLI: `run`, `validate`, `refresh`, `app`, `test` |
| `src/liqfc/pipeline.py` | The end-to-end pipeline (previously `scripts/run_pipeline.py`) |
| `src/liqfc/refresh.py` | Official-source incremental downloaders used by `refresh` |
| `src/liqfc/appcore.py` | Pure functions behind the app: load outputs, buffer interpolation, cost recomputation, cost-vs-τ curve, coverage |
| `app/streamlit_app.py` | Interactive local dashboard (Streamlit 1.66.1 + Altair 6.3.0) |
| `tests/test_app.py`, `tests/test_refresh.py` | App recompute-equals-pipeline, AppTest script run with slider changes, headless launch + HTTP check; offline refresh tests |
| `Makefile`, `pyproject.toml`, `requirements-lock.txt` | Entry points and packaging |
| `outputs/` | `forecasts_backtest.csv`, `metrics.csv/json`, `metrics_by_h.csv`, `nested_selection.csv`, `decision_costs.csv`, `decision_vs_baseline_bootstrap.csv`, `driver_regression.csv`, `live_forecast.csv`, `data_validation.json`, `amfi_file_log.csv`, `actuals.csv` and `drivers_monthly.csv` (app inputs), `refresh_log.json`, `run_summary.json` |
| `data/processed/` | `amfi_panel.csv` (180 rows = 90 months × 2 categories), `drivers_monthly.csv` |
| `reports/report.md`, `reports/dashboard.html` | Report and dashboard |

## Conventions
* **Origin T** is the AMFI release of month T. Models see AMFI months ≤ T and driver months whose `available_date` ≤ the release date of T.
* **Decision for month M** is made on day 1 of M. The latest AMFI release on that day is M−2, so the buffer uses the **h=2** forecast. h=1 is reported as a sensitivity.
* **Outflow** O = max(0, −net flow). **Buffer** = τ-quantile of O. The cost uses s = FBIL 3M T-bill − WACR (% p.a., floored at 0.10%):
  idle = max(0, B−O)·s/1200 and shortfall = 3·max(0, O−B)·s/1200 (₹ crore per month). As a sensitivity, s is held constant at 0.50%.
* **Baseline buffer** = max(0, −actual net flow of M−12).
