"""Cash-buffer decision layer.

Outflow O_M = max(0, -net_M) (category-level NET redemption, INR crore; AMFI publishes gross redemptions but the
buffer is sized against the net drain a category-wide portfolio must fund).
Policy buffer B_M = max(0, -Q_{1-tau}(net_M)) = tau-quantile of outflow from the forecast made at origin M-h.
Baseline  B_M = max(0, -net_{M-12})  ("hold last year's same-month outflow", actual t-12).
Monthly cost (INR crore) with s_M = annualised spread in % (T-bill 3M FBIL minus WACR, floored):
    idle      = max(0, B - O) * s/100/12
    shortfall = penalty * max(0, O - B) * s/100/12
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def buffer_from_quantiles(qrow: pd.Series | dict, tau: float, taus) -> float:
    """tau-quantile of outflow = -(1-tau)-quantile of net flow, floored at 0. Linear interpolation across stored levels."""
    lv = np.array(sorted(taus), float)
    qs = np.array([qrow[f"q_{t}"] for t in sorted(taus)], float)
    p = 1.0 - tau
    if p < lv[0] or p > lv[-1]:
        raise ValueError(f"quantile {p} outside stored levels")
    return float(max(0.0, -np.interp(p, lv, qs)))


def cost(buffer, outflow, spread_pct, penalty=3.0) -> dict:
    b, o, s = np.asarray(buffer, float), np.asarray(outflow, float), np.asarray(spread_pct, float)
    idle = np.maximum(0.0, b - o) * s / 100 / 12
    short_amt = np.maximum(0.0, o - b)
    shortfall = penalty * short_amt * s / 100 / 12
    return {"idle_cost": idle, "shortfall_cost": shortfall, "total_cost": idle + shortfall,
            "shortfall_amt": short_amt, "breach": (short_amt > 0).astype(int)}


def spread_series(drivers: pd.DataFrame, floor_pct: float) -> pd.Series:
    return (drivers["tbill_3m"] - drivers["wacr"]).clip(lower=floor_pct)


def evaluate_policies(fc: pd.DataFrame, panel: pd.DataFrame, drivers: pd.DataFrame, horizon: int, taus,
                      policy_taus, penalty: float, floor_pct: float, const_spread: float | None = None) -> pd.DataFrame:
    """fc: backtest forecasts. Returns one row per (model, policy, category, target month)."""
    act = panel.set_index(["month", "category"])["net"]
    spr = spread_series(drivers, floor_pct)
    f = fc[fc["h"] == horizon]
    rows = []
    for _, r in f.iterrows():
        M, cat = r["target"], r["category"]
        O = max(0.0, -r["actual"])
        s = const_spread if const_spread is not None else float(spr.loc[M])
        base_B = max(0.0, -float(act.loc[(M - 12, cat)]))
        for tau in policy_taus:
            B = buffer_from_quantiles(r, tau, taus)
            c = cost(B, O, s, penalty)
            rows.append({"model": r["model"], "policy": f"q{int(round(tau*100))}", "category": cat, "target": M,
                         "buffer": B, "outflow": O, "spread_pct": s, **{k: float(v) for k, v in c.items()}})
        c = cost(base_B, O, s, penalty)
        rows.append({"model": r["model"], "policy": "smly_baseline", "category": cat, "target": M, "buffer": base_B,
                     "outflow": O, "spread_pct": s, **{k: float(v) for k, v in c.items()}})
    out = pd.DataFrame(rows)
    # baseline rows do not depend on the model; keep a single copy
    base = out[out["policy"] == "smly_baseline"].drop_duplicates(["category", "target"]).assign(model="baseline")
    return pd.concat([out[out["policy"] != "smly_baseline"], base], ignore_index=True)


def paired_vs_baseline(detail: pd.DataFrame, n_boot: int = 2000, block: int = 3, seed: int = 0) -> pd.DataFrame:
    """Monthly cost difference (policy - baseline) with a moving-block bootstrap CI of the total difference."""
    rng = np.random.default_rng(seed)
    keys = ["decision_h", "spread_basis", "category"]
    out = []
    for k, g in detail.groupby(keys):
        base = g[g["policy"] == "smly_baseline"].set_index("target")["total_cost"]
        for (m, pol), gp in g[g["policy"] != "smly_baseline"].groupby(["model", "policy"]):
            d = (gp.set_index("target")["total_cost"] - base).dropna().sort_index().values
            n = len(d)
            starts = rng.integers(0, n - block + 1, size=(n_boot, int(np.ceil(n / block))))
            idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
            boot = d[idx].sum(axis=1)
            out.append({**dict(zip(keys, k)), "model": m, "policy": pol, "n_months": n, "total_diff_cr": d.sum(),
                        "diff_ci90_lo": np.quantile(boot, 0.05), "diff_ci90_hi": np.quantile(boot, 0.95),
                        "months_cheaper": int((d < 0).sum()), "p_boot_diff_ge_0": float((boot >= 0).mean())})
    return pd.DataFrame(out)
