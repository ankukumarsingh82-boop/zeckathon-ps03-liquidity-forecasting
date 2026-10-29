"""App tests: recomputation matches the pipeline, the script runs without exceptions, and the server serves."""
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from liqfc.appcore import buffers, cost_summary, load_outputs, recompute_costs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
needs_outputs = pytest.mark.skipif(not (OUT / "run_summary.json").exists(), reason="run `python -m liqfc run` first")


@needs_outputs
@pytest.mark.parametrize("tau,policy", [(0.75, "q75"), (0.95, "q95")])
@pytest.mark.parametrize("h", [1, 2])
@pytest.mark.parametrize("basis", ["spread", "amount"])
def test_app_recompute_matches_pipeline_decision_costs(tau, policy, h, basis):
    D = load_outputs(OUT)
    s = cost_summary(recompute_costs(D["forecasts_backtest"], D["actuals"], D["drivers_monthly"], tau, h,
                                     "realised", 0.10, 0.5, 3.0, penalty_basis=basis))
    ref = pd.read_csv(OUT / "decision_costs.csv")
    ref = ref[(ref.decision_h == h) & (ref.spread_basis == "fbil3m_minus_wacr") & (ref.penalty_basis == basis)]
    m = s.merge(ref[ref.policy == policy], on=["category", "model"], suffixes=("_app", "_pipe"))
    assert len(m) == len(s) > 0
    assert np.allclose(m.policy_cost_cr, m.total_cost_cr)
    assert np.allclose(m.idle_cr, m.idle_cost_cr) and np.allclose(m.shortfall_cr, m.shortfall_cost_cr)
    assert np.array_equal(m.breaches_app.astype(int), m.breaches_pipe.astype(int))
    assert np.allclose(m.mean_buffer_cr_app, m.mean_buffer_cr_pipe)
    b = ref[ref.policy == "smly_baseline"].set_index("category")["total_cost_cr"]
    assert np.allclose(s.baseline_cost_cr, s.category.map(b))


def test_buffers_interpolation():
    q = pd.DataFrame([{f"q_{t}": v for t, v in zip([0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975],
                                                   [-1000, -800, -600, -300, 0, 300, 600, 800, 1000])}])
    assert buffers(q, 0.95)[0] == pytest.approx(800, abs=1e-6) and buffers(q, 0.875)[0] == pytest.approx(550, abs=1e-6)  # p=0.125: -600 + (0.025/0.15)*300 = -550
    with pytest.raises(ValueError):
        buffers(q, 0.99)


@needs_outputs
def test_streamlit_script_runs_and_sliders_recompute():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 6
    first = [m.value for m in at.metric]
    at.slider(key="tau").set_value(0.95).run()
    assert not at.exception
    assert [m.value for m in at.metric] != first          # buffers changed with tau
    at.radio(key="spread_mode").set_value("constant").run()
    at.slider(key="const").set_value(1.0).run()
    assert not at.exception
    tables = lambda: [d.value.to_csv() for d in at.dataframe]
    before = tables()
    at.radio(key="pen_basis").set_value(at.radio(key="pen_basis").options[1]).run()   # reading (b)
    assert not at.exception
    assert tables() != before          # the cost tables change with the penalty reading (buffers do not)


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


@needs_outputs
def test_app_launches_headless_and_serves():
    port = _free_port()
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    proc = subprocess.Popen([sys.executable, "-m", "liqfc", "app", "--port", str(port)], cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        ok, deadline = False, time.time() + 90
        while time.time() < deadline and not ok:
            try:
                ok = urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=3).read().strip() == b"ok"
            except OSError:
                time.sleep(1)
        assert ok, "health endpoint never returned ok"
        html = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=10).read().decode()
        assert "<html" in html.lower() and "streamlit" in html.lower()
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
    # stopping `python -m liqfc app` must stop the server too (no orphaned Streamlit process)
    with pytest.raises(OSError):
        urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=3)
