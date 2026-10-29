"""Leakage-free rolling-origin backtest. At origin T (= release date of AMFI month T) every model is refit on
AMFI months <= T and driver months whose availability date <= release date of T."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import REGISTRY


def info_set(panel: pd.DataFrame, drivers: pd.DataFrame, T: pd.Period):
    as_of = panel.loc[panel["month"] == T, "release_date"].max()
    if pd.isna(as_of):
        raise ValueError(f"origin {T} not released")
    p = panel[(panel["month"] <= T) & (panel["release_date"] <= as_of)].copy()
    d = drivers[(drivers.index <= T) & (drivers["available_date"] <= as_of)].copy()
    assert p["month"].max() == T and (d.index.max() <= T if len(d) else True)
    return p, d, as_of


def mase_scale(panel: pd.DataFrame, T: pd.Period) -> dict:
    """In-sample MAE of the seasonal-naive forecast on the training window (per category)."""
    out = {}
    for cat, g in panel[panel["month"] <= T].groupby("category"):
        s = g.set_index("month")["net"].sort_index()
        out[cat] = float((s - s.shift(12)).abs().dropna().mean())
    return out


def test_targets(panel: pd.DataFrame, n_test: int) -> pd.PeriodIndex:
    months = pd.PeriodIndex(sorted(panel["month"].unique()))
    return months[-n_test:]


def run_backtest(panel, drivers, models, horizons, taus, n_test=None, lgbm_cfg=None, min_train=36, log=print,
                 targets=None) -> pd.DataFrame:
    if targets is None:
        targets = test_targets(panel, n_test)
    origins = pd.period_range(targets[0] - max(horizons), targets[-1] - min(horizons), freq="M")
    actual = panel.set_index(["month", "category"])["net"]
    aum = panel.set_index(["month", "category"])["aum"]
    rel = panel.groupby("month")["release_date"].max()
    rows = []
    for T in origins:
        p, d, as_of = info_set(panel, drivers, T)
        assert p["month"].nunique() >= min_train
        scale = mase_scale(p, T)
        for name in models:
            out = REGISTRY[name](p, d, T, horizons, taus, lgbm_cfg if name.startswith("lgbm") else None)
            for r in out:
                if r["target"] in targets:
                    r.update(model=name, as_of=as_of, actual=float(actual.loc[(r["target"], r["category"])]),
                             mase_scale=scale[r["category"]], n_train_months=int(p["month"].nunique()),
                             aum_origin=float(aum.loc[(T, r["category"])]), target_release=rel.loc[r["target"]])
                    rows.append(r)
        log(f"origin {T} (as_of {as_of.date()}) done")
    df = pd.DataFrame(rows)
    return df


def all_targets(panel, horizons, min_train) -> pd.PeriodIndex:
    months = pd.PeriodIndex(sorted(panel["month"].unique()))
    return months[min_train - 1 + max(horizons):]


def oos_calibrate(fc: pd.DataFrame, taus, min_n: int = 12, suffix: str = "_oosq") -> pd.DataFrame:
    """Horizon-specific out-of-sample PIs: at origin T, replace each model's quantiles by
    point + AUM_T * empirical quantile of past OOS errors (actual-point)/AUM_origin for the same model/category/h,
    using only forecasts whose target was already released at T's as_of. Rows with < min_n past errors are dropped."""
    fc = fc.copy()
    fc["err_ratio"] = (fc["actual"] - fc["point"]) / fc["aum_origin"]
    out = []
    for (m, c, h), g in fc.groupby(["model", "category", "h"]):
        g = g.sort_values("origin")
        for _, r in g.iterrows():
            past = g[g["target_release"] <= r["as_of"]]
            if len(past) < min_n:
                continue
            q = np.quantile(past["err_ratio"].values, taus)
            r2 = r.copy()
            for t, v in zip(taus, q):
                r2[f"q_{t}"] = r["point"] + r["aum_origin"] * v
            r2["model"] = m + suffix
            r2["n_calib"] = len(past)
            out.append(r2)
    res = pd.DataFrame(out).drop(columns=["err_ratio"])
    qc = [f"q_{t}" for t in taus]
    res[qc] = np.sort(res[qc].values, axis=1)
    return res
