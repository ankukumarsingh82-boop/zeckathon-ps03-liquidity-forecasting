"""Forecast metrics. All errors in INR crore unless stated. Pinball uses the standard convention
L_tau(y, q) = max(tau*(y-q), (tau-1)*(y-q)), averaged over quantile levels and observations."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def pinball(y, q, tau: float) -> np.ndarray:
    d = np.asarray(y, float) - np.asarray(q, float)
    return np.maximum(tau * d, (tau - 1.0) * d)


def mean_pinball(y, qmat: np.ndarray, taus) -> float:
    """qmat shape (n, len(taus))."""
    qmat = np.asarray(qmat, float)
    return float(np.mean([pinball(y, qmat[:, j], t).mean() for j, t in enumerate(taus)]))


def ape(y, yhat) -> np.ndarray:
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.abs(y - yhat) / np.abs(y)


def mape(y, yhat) -> float:
    """Plain MAPE (%), denominator |actual|, all months kept (as required). inf if any actual == 0."""
    return float(100 * np.mean(ape(y, yhat)))


def median_ape(y, yhat) -> float:
    return float(100 * np.median(ape(y, yhat)))


def smape(y, yhat) -> float:
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    den = np.abs(y) + np.abs(yhat)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(den == 0, 0.0, 2 * np.abs(y - yhat) / den)
    return float(100 * v.mean())


def mae(y, yhat) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(yhat, float))))


def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(yhat, float)) ** 2)))


def wape(y, yhat) -> float:
    y = np.asarray(y, float)
    return float(100 * np.sum(np.abs(y - np.asarray(yhat, float))) / np.sum(np.abs(y)))


def mase(y, yhat, scale) -> float:
    """scale: per-observation in-sample seasonal-naive MAE computed on the training window at the origin."""
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(yhat, float)) / np.asarray(scale, float)))


def coverage(y, lo, hi) -> float:
    y = np.asarray(y, float)
    return float(np.mean((y >= np.asarray(lo, float)) & (y <= np.asarray(hi, float))))


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def summarize(df: pd.DataFrame, taus, core_taus, small_thr: float) -> dict:
    """df rows = forecasts with columns actual, point, q_<tau>, mase_scale."""
    y, p = df["actual"].values, df["point"].values
    qcols = [f"q_{t}" for t in taus]
    ccols = [f"q_{t}" for t in core_taus]
    big = np.abs(y) >= small_thr
    n = len(df)
    k80 = int(((y >= df["q_0.1"]) & (y <= df["q_0.9"])).sum())
    k95 = int(((y >= df["q_0.025"]) & (y <= df["q_0.975"])).sum())
    return {
        "n": n,
        "MAPE_pct": mape(y, p),
        "MAPE_excl_small_pct": mape(y[big], p[big]) if big.any() else np.nan,
        "n_small_excluded": int((~big).sum()),
        "medianAPE_pct": median_ape(y, p),
        "sMAPE_pct": smape(y, p),
        "WAPE_pct": wape(y, p),
        "MAE_cr": mae(y, p),
        "RMSE_cr": rmse(y, p),
        "MASE": mase(y, p, df["mase_scale"].values),
        "pinball_core_cr": mean_pinball(y, df[ccols].values, core_taus),
        "pinball_all_cr": mean_pinball(y, df[qcols].values, taus),
        "scaled_pinball_core": mean_pinball(y / df["mase_scale"].values, df[ccols].values / df["mase_scale"].values[:, None], core_taus),
        "cov80": k80 / n, "cov80_lo": wilson_ci(k80, n)[0], "cov80_hi": wilson_ci(k80, n)[1],
        "cov95": k95 / n, "cov95_lo": wilson_ci(k95, n)[0], "cov95_hi": wilson_ci(k95, n)[1],
    }
