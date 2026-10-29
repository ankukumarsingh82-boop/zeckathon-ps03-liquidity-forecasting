"""No-leakage tests for the rolling-origin design (auditor finding 4)."""
import numpy as np
import pandas as pd
import pytest

from liqfc.backtest import info_set, oos_calibrate, run_backtest
from liqfc.models import REGISTRY, ml_design

TAUS = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]


def test_info_set_truncates_panel_and_drivers(panel, drivers):
    T = pd.Period("2025-06", "M")
    p, d, as_of = info_set(panel, drivers, T)
    assert p["month"].max() == T and (p["release_date"] <= as_of).all()
    assert d.index.max() <= T and (d["available_date"] <= as_of).all()


def test_driver_not_available_by_release_is_dropped(panel, drivers):
    T = pd.Period("2025-06", "M")
    as_of = panel.loc[panel.month == T, "release_date"].max()
    d2 = drivers.copy()
    d2.loc[T, "available_date"] = as_of + pd.Timedelta(days=1)
    _, d, _ = info_set(panel, d2, T)
    assert T not in d.index


def test_driver_availability_precedes_amfi_release(panel, drivers):
    rel = panel.groupby("month")["release_date"].max()
    common = rel.index.intersection(drivers.index)
    assert (drivers.loc[common, "available_date"] <= rel.loc[common]).all()


@pytest.mark.parametrize("model", ["snaive", "snaive_ratio", "ets_ratio", "sarimax_cal", "lgbm_q"])
def test_forecast_invariant_to_future_data(panel, drivers, model, cfg):
    """Scrambling every observation after the origin must not change the forecast."""
    T = pd.Period("2025-03", "M")
    rng = np.random.default_rng(0)
    pz, dz = panel.copy(), drivers.copy()
    fut = pz["month"] > T
    pz.loc[fut, ["net", "aum"]] = rng.normal(0, 1e6, (fut.sum(), 2)).__abs__()
    dz.loc[dz.index > T, ["wacr", "tbill_3m", "net_liq", "nifty_close"]] = 999.0
    a = REGISTRY[model](*info_set(panel, drivers, T)[:2], T, [1, 2, 3], TAUS, cfg["lgbm"])
    b = REGISTRY[model](*info_set(pz, dz, T)[:2], T, [1, 2, 3], TAUS, cfg["lgbm"])
    assert pd.DataFrame(a).equals(pd.DataFrame(b))


def test_ml_features_equal_under_truncation(panel, drivers):
    """Features of every row with origin <= T are identical whether built from full or truncated data."""
    T = pd.Period("2024-12", "M")
    full, feats = ml_design(panel, drivers, [1, 2, 3])
    trunc, _ = ml_design(panel[panel.month <= T], drivers[drivers.index <= T], [1, 2, 3])
    key = ["category", "origin", "h"]
    m = full[full.origin <= T].merge(trunc, on=key, suffixes=("", "_t"))
    assert len(m) == len(trunc)
    for f in [f for f in feats if f not in key]:
        assert np.allclose(m[f].astype(float), m[f + "_t"].astype(float), equal_nan=True), f


def test_backtest_rows_are_out_of_sample(panel, drivers):
    tg = pd.period_range("2026-07", "2026-09", freq="M")
    fc = run_backtest(panel, drivers, ["snaive"], [1, 2, 3], TAUS, targets=tg, log=lambda *_: None)
    assert (fc["target"] > fc["origin"]).all()
    assert (fc["as_of"] < fc["target_release"]).all()
    months = pd.PeriodIndex(sorted(panel.month.unique()))
    assert (fc["n_train_months"] == fc["origin"].map(lambda o: int((months <= o).sum()))).all()
    assert fc.groupby(["category", "h"]).size().eq(3).all()


def test_oos_calibration_only_uses_released_errors():
    rows = []
    for i, o in enumerate(pd.period_range("2023-01", "2024-12", freq="M")):
        rows.append({"model": "m", "category": "c", "h": 1, "origin": o, "target": o + 1, "point": 0.0,
                     "actual": float(i), "aum_origin": 1.0, "as_of": (o + 1).to_timestamp() + pd.Timedelta(days=9),
                     "target_release": (o + 2).to_timestamp() + pd.Timedelta(days=9), **{f"q_{t}": 0.0 for t in TAUS}})
    fc = pd.DataFrame(rows)
    out = oos_calibrate(fc, TAUS, min_n=12)
    first = out.sort_values("origin").iloc[0]
    # first calibrated origin has exactly 12 released past errors: 0..11 -> max quantile <= 11
    assert first["n_calib"] == 12 and first["q_0.975"] <= 11
    assert (out["n_calib"] < out["origin"].map(lambda o: (fc.origin < o).sum() + 1)).all()
