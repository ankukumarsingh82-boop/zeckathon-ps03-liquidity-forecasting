"""Pure data functions behind the interactive app. Inputs are ONLY the CSV/JSON files in outputs/."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

TAUS = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]
QCOLS = [f"q_{t}" for t in TAUS]
REQUIRED = ["actuals.csv", "drivers_monthly.csv", "forecasts_backtest.csv", "live_forecast.csv", "metrics.csv",
            "metrics_by_h.csv", "nested_selection.csv", "decision_costs.csv", "driver_regression.csv", "run_summary.json"]


def load_outputs(out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    missing = [f for f in REQUIRED if not (out_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"missing outputs: {missing} - run `python -m liqfc run` first")
    d = {f[:-4]: pd.read_csv(out_dir / f) for f in REQUIRED if f.endswith(".csv")}
    d["run_summary"] = json.load(open(out_dir / "run_summary.json"))
    for k in ("decision_vs_baseline_bootstrap", "data_validation"):
        p = out_dir / (k + (".csv" if k.startswith("decision") else ".json"))
        if p.exists():
            d[k] = pd.read_csv(p) if p.suffix == ".csv" else json.load(open(p))
    d["mtimes"] = {f: pd.Timestamp((out_dir / f).stat().st_mtime, unit="s", tz="UTC") for f in REQUIRED}
    return d


def buffers(q: pd.DataFrame, tau: float) -> np.ndarray:
    """tau-quantile of outflow = max(0, -Q_{1-tau}(net)), linear interpolation over stored quantile levels."""
    p = 1.0 - tau
    if not (TAUS[0] <= p <= TAUS[-1]):
        raise ValueError(f"tau must be in [{1-TAUS[-1]}, {1-TAUS[0]}]")
    vals = q[QCOLS].to_numpy(float)
    return np.maximum(0.0, -np.array([np.interp(p, TAUS, v) for v in vals]))


def monthly_spread(drivers: pd.DataFrame, mode: str, floor_pct: float, const_pct: float) -> pd.Series:
    s = drivers.set_index("month")
    if mode == "constant":
        return pd.Series(const_pct, index=s.index)
    return (s["tbill_3m"] - s["wacr"]).clip(lower=floor_pct)


def recompute_costs(fc: pd.DataFrame, actuals: pd.DataFrame, drivers: pd.DataFrame, tau: float, h: int,
                    spread_mode: str = "realised", floor_pct: float = 0.10, const_pct: float = 0.50,
                    penalty: float = 3.0, penalty_basis: str = "spread") -> pd.DataFrame:
    """Per (model, category, target) cost of the tau-buffer policy and of the same-month-last-year baseline,
    from backtest forecasts (outputs/forecasts_backtest.csv) and actual flows (outputs/actuals.csv)."""
    if penalty_basis not in ("spread", "amount"):
        raise ValueError("penalty_basis must be 'spread' or 'amount'")
    f = fc[fc["h"] == h].copy()
    act = actuals.set_index(["month", "category"])["net"]
    spr = monthly_spread(drivers, spread_mode, floor_pct, const_pct)
    f["buffer"] = buffers(f, tau)
    f["outflow"] = np.maximum(0.0, -f["actual"].astype(float))
    prev = (pd.PeriodIndex(f["target"], freq="M") - 12).astype(str)
    f["baseline_buffer"] = np.maximum(0.0, -np.array([act.loc[(m, c)] for m, c in zip(prev, f["category"])], float))
    f["spread_pct"] = spr.reindex(f["target"].astype(str)).to_numpy(float)
    k = f["spread_pct"] / 100 / 12
    for pre, b in [("", f["buffer"]), ("baseline_", f["baseline_buffer"])]:
        f[pre + "idle_cost"] = np.maximum(0.0, b - f["outflow"]) * k
        # reading (a) "spread": 3 x shortfall x spread/12 ; reading (b) "amount": 3 x shortfall in rupees
        f[pre + "shortfall_cost"] = penalty * np.maximum(0.0, f["outflow"] - b) * (k if penalty_basis == "spread" else 1.0)
        f[pre + "total_cost"] = f[pre + "idle_cost"] + f[pre + "shortfall_cost"]
        f[pre + "breach"] = (f["outflow"] > b).astype(int)
    return f[["model", "category", "target", "h", "buffer", "baseline_buffer", "outflow", "actual", "spread_pct",
              "idle_cost", "shortfall_cost", "total_cost", "breach", "baseline_idle_cost", "baseline_shortfall_cost",
              "baseline_total_cost", "baseline_breach"]]


def cost_summary(detail: pd.DataFrame) -> pd.DataFrame:
    g = detail.groupby(["category", "model"]).agg(
        policy_cost_cr=("total_cost", "sum"), idle_cr=("idle_cost", "sum"), shortfall_cr=("shortfall_cost", "sum"),
        breaches=("breach", "sum"), mean_buffer_cr=("buffer", "mean"), baseline_cost_cr=("baseline_total_cost", "sum"),
        baseline_breaches=("baseline_breach", "sum"), months=("target", "size")).reset_index()
    g["vs_baseline_pct"] = 100 * (g["policy_cost_cr"] / g["baseline_cost_cr"] - 1)
    return g.sort_values(["category", "policy_cost_cr"])


def cost_curve(fc, actuals, drivers, h, models, taus=None, **kw) -> pd.DataFrame:
    taus = taus if taus is not None else np.round(np.arange(0.50, 0.976, 0.025), 3)
    rows = []
    for t in taus:
        s = cost_summary(recompute_costs(fc[fc.model.isin(models)], actuals, drivers, float(t), h, **kw))
        rows.append(s.assign(tau=float(t)))
    return pd.concat(rows, ignore_index=True)


def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - hw), min(1.0, c + hw)


def coverage_table(fc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (m, c, h), g in fc.groupby(["model", "category", "h"]):
        y = g["actual"]; n = len(g)
        for lvl, lo, hi in [(80, "q_0.1", "q_0.9"), (95, "q_0.025", "q_0.975")]:
            k = int(((y >= g[lo]) & (y <= g[hi])).sum())
            a, b = wilson(k, n)
            rows.append({"model": m, "category": c, "h": h, "nominal": lvl, "coverage": k / n, "lo": a, "hi": b, "n": n})
    return pd.DataFrame(rows)
