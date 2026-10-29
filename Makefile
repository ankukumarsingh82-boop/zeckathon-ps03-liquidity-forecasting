# Local-only targets. Nothing here deploys or pushes anything.
PY  ?= .venv/bin/python
PORT ?= 8501

.PHONY: install run run-fast validate refresh app test smoke docs all
install:
	python3.13 -m venv .venv
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e . --no-deps
run:        ; $(PY) -m liqfc run                 # ingest + validate + backtest + forecast + report tables (~2-3 min)
run-fast:   ; $(PY) -m liqfc run --skip-backtest # reuse outputs/forecasts_backtest_full.csv (~15 s)
validate:   ; $(PY) -m liqfc validate
refresh:    ; $(PY) -m liqfc refresh             # fetch new official data (AMFI/NSE/FBIL/RBI), then full run
app:        ; $(PY) -m liqfc app --port $(PORT)  # http://127.0.0.1:$(PORT)
test:       ; $(PY) -m pytest -q
docs:       ; $(PY) -m liqfc docs                # report.md + figures + reports/report.pdf + reports/deck.pdf
smoke:      ; $(PY) -m pytest -q tests/test_app.py -k launches_headless
all: run docs test
