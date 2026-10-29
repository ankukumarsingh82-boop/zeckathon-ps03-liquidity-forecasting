import numpy as np
import pytest

from liqfc import metrics as M


def test_pinball_sign_and_values():
    # under-prediction at high quantile is penalised by tau, over-prediction by (1-tau)
    assert M.pinball(10, 8, 0.9) == pytest.approx(0.9 * 2)
    assert M.pinball(8, 10, 0.9) == pytest.approx(0.1 * 2)
    assert M.pinball(5, 5, 0.3) == 0
    assert np.all(M.pinball(np.array([1, -1]), np.array([0, 0]), 0.5) == 0.5)


def test_median_pinball_is_half_mae():
    y, q = np.array([1.0, -3, 7]), np.array([0.0, 0, 0])
    assert M.mean_pinball(y, q[:, None], [0.5]) == pytest.approx(0.5 * M.mae(y, q))


def test_point_metrics():
    y, p = np.array([100.0, -50, 200]), np.array([110.0, -25, 100])
    assert M.mape(y, p) == pytest.approx(100 * np.mean([0.1, 0.5, 0.5]))
    assert M.median_ape(y, p) == pytest.approx(50)
    assert M.mae(y, p) == pytest.approx(np.mean([10, 25, 100]))
    assert M.smape(y, p) == pytest.approx(100 * np.mean([20 / 210, 50 / 75, 200 / 300]))
    assert M.wape(y, p) == pytest.approx(100 * 135 / 350)
    assert M.mase(y, p, np.array([10.0, 10, 10])) == pytest.approx(np.mean([1, 2.5, 10]))
    assert M.mape(np.array([0.0, 1]), np.array([1.0, 1])) == np.inf   # documented: MAPE explodes near zero


def test_coverage_and_wilson():
    assert M.coverage([1, 2, 3, 4], [0, 0, 0, 5], [5, 5, 2.5, 6]) == 0.5
    lo, hi = M.wilson_ci(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-3) and hi == pytest.approx(0.9433, abs=1e-3)
    lo, hi = M.wilson_ci(24, 24)
    assert hi == 1.0 and lo == pytest.approx(0.8619, abs=1e-3)
