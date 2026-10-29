"""Forecast models. Each model maps an information set (data truncated at origin T) to quantile forecasts
for months T+1..T+H for every category. Point forecast = model mean/median as noted."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

from .features import DRIVER_GROUPS, calendar_frame, driver_features, sarimax_exog


def _wide(panel: pd.DataFrame, col: str) -> pd.DataFrame:
    return panel.pivot(index="month", columns="category", values=col).sort_index()


def _out(cat, T, h, point, qs: dict) -> dict:
    return {"category": cat, "origin": T, "h": h, "target": T + h, "point": float(point),
            **{f"q_{k}": float(v) for k, v in qs.items()}}


def _sort_q(rows: list[dict], taus) -> list[dict]:
    for r in rows:  # enforce monotone quantiles (no crossing)
        v = np.sort([r[f"q_{t}"] for t in taus])
        for t, x in zip(taus, v):
            r[f"q_{t}"] = float(x)
    return rows


def snaive(panel, drivers, T, horizons, taus, cfg=None) -> list[dict]:
    """Seasonal naive on levels: yhat(T+h) = y(T+h-12). PIs: empirical quantiles of in-sample seasonal differences."""
    y = _wide(panel, "net")
    rows = []
    for cat in y.columns:
        s = y[cat].dropna()
        d = (s - s.shift(12)).dropna().values
        for h in horizons:
            pt = s.loc[T + h - 12]
            rows.append(_out(cat, T, h, pt, {t: pt + np.quantile(d, t) for t in taus}))
    return _sort_q(rows, taus)


def snaive_ratio(panel, drivers, T, horizons, taus, cfg=None) -> list[dict]:
    """Seasonal naive on flow/lagged-AUM ratio, rescaled by AUM at origin (flows scale with category size)."""
    y, a = _wide(panel, "net"), _wide(panel, "aum")
    rows = []
    for cat in y.columns:
        r = (y[cat] / a[cat].shift(1)).dropna()
        d = (r - r.shift(12)).dropna().values
        aT = a[cat].loc[T]
        for h in horizons:
            pr = r.loc[T + h - 12]
            rows.append(_out(cat, T, h, pr * aT, {t: (pr + np.quantile(d, t)) * aT for t in taus}))
    return _sort_q(rows, taus)


def ets_ratio(panel, drivers, T, horizons, taus, cfg=None) -> list[dict]:
    """ETS(A,N,A) on flow/lagged-AUM ratio; analytic Gaussian PIs; rescaled by AUM at origin."""
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    y, a = _wide(panel, "net"), _wide(panel, "aum")
    rows = []
    H = max(horizons)
    for cat in y.columns:
        r = (y[cat] / a[cat].shift(1)).dropna()
        r.index = r.index.to_timestamp(how="end").normalize()
        r = r.asfreq("ME")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ETSModel(r, error="add", trend=None, seasonal="add", seasonal_periods=12).fit(disp=False, maxiter=2000)
            pred = res.get_prediction(start=len(r), end=len(r) + H - 1)
            mean = np.asarray(pred.predicted_mean)
            var = np.asarray(pred.var_pred_mean) if hasattr(pred, "var_pred_mean") else None
        if var is None or not np.all(np.isfinite(var)):
            sig = np.std(res.resid) * np.sqrt(np.arange(1, H + 1))
        else:
            sig = np.sqrt(var)
        aT = a[cat].loc[T]
        for h in horizons:
            m, s = mean[h - 1], sig[h - 1]
            rows.append(_out(cat, T, h, m * aT, {t: (m + norm.ppf(t) * s) * aT for t in taus}))
    return _sort_q(rows, taus)


def sarimax_cal(panel, drivers, T, horizons, taus, cfg=None) -> list[dict]:
    """Regression with AR(1) errors on flow/lagged-AUM ratio; exog = known calendar dummies
    (Jun/Sep/Dec advance-tax quarter-ends, March FY-end, April, other quarter starts)."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    y, a = _wide(panel, "net"), _wide(panel, "aum")
    rows = []
    H = max(horizons)
    for cat in y.columns:
        r = (y[cat] / a[cat].shift(1)).dropna()
        X = sarimax_exog(r.index)
        fut = pd.period_range(T + 1, T + H, freq="M")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = SARIMAX(r.values, exog=X.values, order=(1, 0, 0), trend="c").fit(disp=False, maxiter=500)
            fc = res.get_forecast(steps=H, exog=sarimax_exog(fut).values)
            mean, se = np.asarray(fc.predicted_mean), np.sqrt(np.asarray(fc.var_pred_mean))
        aT = a[cat].loc[T]
        for h in horizons:
            m, s = mean[h - 1], se[h - 1]
            rows.append(_out(cat, T, h, m * aT, {t: (m + norm.ppf(t) * s) * aT for t in taus}))
    return _sort_q(rows, taus)


# ---------------------------------------------------------------- ML challenger
def ml_design(panel: pd.DataFrame, drivers: pd.DataFrame, horizons, driver_groups=("rates", "liquidity", "equity")):
    """Direct multi-horizon design matrix. Row = (category, origin t, h); target = y(t+h)/AUM(t).
    Features use only months <= t (lags of the target ratio, drivers at t) plus calendar of the target month."""
    y, a = _wide(panel, "net"), _wide(panel, "aum")
    dfeat = driver_features(drivers)
    dcols = [c for g in driver_groups for c in DRIVER_GROUPS[g]]
    recs = []
    months = y.index
    for ci, cat in enumerate(y.columns):
        r = y[cat] / a[cat].shift(1)
        for t in months:
            for h in horizons:
                tgt = t + h
                if (t - 13) not in months:      # need r(t+h-12) and 12m AUM growth
                    continue
                rec = {"category": cat, "is_liquid": int(cat.startswith("Liquid")), "origin": t, "h": h, "target_month": tgt,
                       "r_t": r.get(t), "r_t1": r.get(t - 1), "r_t2": r.get(t - 2),
                       "r_smly": r.get(tgt - 12), "r_smly2": r.get(tgt - 24, np.nan),
                       "aum_g12": np.log(a[cat].loc[t] / a[cat].loc[t - 12]), "aum_t": a[cat].loc[t],
                       "y_target": y[cat].get(tgt, np.nan)}
                rec.update(calendar_frame(pd.PeriodIndex([tgt])).iloc[0].to_dict())
                for c in dcols:
                    rec[c] = dfeat[c].get(t, np.nan)
                recs.append(rec)
    D = pd.DataFrame(recs)
    D["z_target"] = D["y_target"] / D["aum_t"]
    feats = ["is_liquid", "h", "r_t", "r_t1", "r_t2", "r_smly", "r_smly2", "aum_g12",
             "adv_tax", "fy_end", "fy_start", "q_start", "month_sin", "month_cos"] + dcols
    return D, feats


def _lgbm_q(panel, drivers, T, horizons, taus, cfg, groups):
    import lightgbm as lgb
    D, feats = ml_design(panel, drivers, horizons, groups)
    train = D[D["target_month"] <= T].dropna(subset=["z_target"])     # targets realised (and released) by T
    test = D[D["origin"] == T]
    p = dict(cfg or {})
    params = dict(objective="quantile", n_estimators=p.get("n_estimators", 300), learning_rate=p.get("learning_rate", 0.03),
                  num_leaves=p.get("num_leaves", 4), min_child_samples=p.get("min_data_in_leaf", 8),
                  colsample_bytree=p.get("feature_fraction", 0.8), subsample=p.get("bagging_fraction", 0.8),
                  subsample_freq=p.get("bagging_freq", 1), reg_lambda=p.get("lambda_l2", 1.0),
                  random_state=p.get("seed", 42), verbose=-1,
                  n_jobs=p.get("num_threads", 1))  # single thread: tiny data; avoids OpenMP spin-stall on a busy CPU
    preds = {}
    for t in taus:
        m = lgb.LGBMRegressor(alpha=t, **params).fit(train[feats], train["z_target"])
        preds[t] = m.predict(test[feats])
    rows = []
    for i, (_, r) in enumerate(test.iterrows()):
        qs = {t: preds[t][i] * r["aum_t"] for t in taus}
        rows.append(_out(r["category"], T, int(r["h"]), qs[0.5], qs))
    return _sort_q(rows, taus)


def lgbm_q(panel, drivers, T, horizons, taus, cfg=None):
    """Quantile LightGBM, pooled over categories and horizons; calendar + target lags + rates + liquidity + equity."""
    return _lgbm_q(panel, drivers, T, horizons, taus, cfg, ("rates", "liquidity", "equity"))


def lgbm_q_calonly(panel, drivers, T, horizons, taus, cfg=None):
    """Ablation: same ML model without market drivers (calendar + target lags only)."""
    return _lgbm_q(panel, drivers, T, horizons, taus, cfg, ())


def lgbm_q_no_rates(panel, drivers, T, horizons, taus, cfg=None):
    return _lgbm_q(panel, drivers, T, horizons, taus, cfg, ("liquidity", "equity"))


def lgbm_q_no_liq(panel, drivers, T, horizons, taus, cfg=None):
    return _lgbm_q(panel, drivers, T, horizons, taus, cfg, ("rates", "equity"))


def lgbm_q_no_equity(panel, drivers, T, horizons, taus, cfg=None):
    return _lgbm_q(panel, drivers, T, horizons, taus, cfg, ("rates", "liquidity"))


REGISTRY = {f.__name__: f for f in [snaive, snaive_ratio, ets_ratio, sarimax_cal, lgbm_q,
                                     lgbm_q_calonly, lgbm_q_no_rates, lgbm_q_no_liq, lgbm_q_no_equity]}
