"""End-to-end pipeline (invoked via `python -m liqfc run`): data -> backtest -> metrics -> decision -> drivers -> live forecast. Run from repo root."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
from .amfi import load_amfi, validate_panel  # noqa: E402
from .backtest import all_targets, info_set, oos_calibrate, run_backtest  # noqa: E402
from .decision import buffer_from_quantiles, evaluate_policies, paired_vs_baseline, spread_series  # noqa: E402
from .drivers import calendar_driver_regression  # noqa: E402
from .macro import build_monthly_drivers  # noqa: E402
from .metrics import summarize  # noqa: E402
from .models import REGISTRY  # noqa: E402

warnings.filterwarnings("ignore")
OUT, PROC = ROOT / "outputs", ROOT / "data/processed"


def per(s):
    return pd.Period(s, "M")


def metrics_table(fc, cfg, by=("model", "category", "h")):
    rows = []
    for k, g in fc.groupby(list(by)):
        rows.append({**dict(zip(by, k if isinstance(k, tuple) else (k,))),
                     **summarize(g, cfg["quantiles"], cfg["core_quantiles"], cfg["mape_small_threshold_crore"])})
    return pd.DataFrame(rows)


def main(skip_backtest=False, log=print):
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    OUT.mkdir(exist_ok=True); PROC.mkdir(parents=True, exist_ok=True)
    taus, H = cfg["quantiles"], cfg["backtest"]["horizons"]
    # ---------------- data
    panel, meta = load_amfi(ROOT / cfg["data"]["amfi_raw_dir"], cfg["data"]["categories"], cfg["data"]["panel_start"],
                            cfg["data"]["release_lag_default_days"], cfg["data"]["identity_tol_crore"])
    vrep = validate_panel(panel, cfg["data"]["categories"], cfg["data"]["panel_start"], cfg["data"]["identity_tol_crore"])
    drivers = build_monthly_drivers()
    panel.assign(month=panel["month"].astype(str)).to_csv(PROC / "amfi_panel.csv", index=False)
    drivers.rename_axis("month").reset_index().assign(month=lambda d: d["month"].astype(str)).to_csv(PROC / "drivers_monthly.csv", index=False)
    meta.to_csv(OUT / "amfi_file_log.csv", index=False)
    json.dump({"validation": vrep, "n_files": len(meta), "n_rejected": int(meta["layout"].astype(str).str.startswith("REJECTED").sum()),
               "release_date_imputed_months": sorted(panel.loc[panel["release_date_imputed"], "month"].astype(str).unique().tolist())}, open(OUT / "data_validation.json", "w"), indent=2, default=str)
    # copies for the interactive app, which reads only outputs/
    panel.assign(month=panel["month"].astype(str)).to_csv(OUT / "actuals.csv", index=False)
    drivers.rename_axis("month").reset_index().assign(month=lambda d: d["month"].astype(str)).to_csv(OUT / "drivers_monthly.csv", index=False)
    print("panel", panel.shape, panel["month"].min(), panel["month"].max())

    # ---------------- backtest (all feasible origins; headline = last 24 targets)
    fpath = OUT / "forecasts_backtest_full.csv"
    models = cfg["models"] + cfg["ablation_models"]
    if skip_backtest and fpath.exists():
        fc = pd.read_csv(fpath, parse_dates=["as_of", "target_release"])
        for c in ["origin", "target"]:
            fc[c] = fc[c].map(per)
    else:
        tg = all_targets(panel, H, cfg["backtest"]["min_train_months"])
        fc = run_backtest(panel, drivers, models, H, taus, lgbm_cfg=cfg["lgbm"],
                          min_train=cfg["backtest"]["min_train_months"], targets=tg)
        fc.to_csv(fpath, index=False)
    cal = oos_calibrate(fc[fc["model"].isin(cfg["models"])], taus, cfg["calibration"]["oos_min_n"])
    fca = pd.concat([fc, cal], ignore_index=True)
    months = pd.PeriodIndex(sorted(panel["month"].unique()))
    test = months[-cfg["backtest"]["n_test_months"]:]
    ft = fca[fca["target"].isin(test)].copy()
    assert ft.groupby(["model", "category", "h"]).size().eq(len(test)).all(), "every model must cover all 24 targets"
    ft.to_csv(OUT / "forecasts_backtest.csv", index=False)

    m_h = metrics_table(ft, cfg)
    m_all = metrics_table(ft, cfg, by=("model", "category"))
    m_h.to_csv(OUT / "metrics_by_h.csv", index=False)
    m_all.to_csv(OUT / "metrics.csv", index=False)

    # nested selection: choose on select window, evaluate on eval window
    if cfg["selection"].get("select_months", "auto") == "auto":   # first half / second half of the test window
        half = len(test) // 2
        s0, s1, e0, e1 = test[0], test[half - 1], test[half], test[-1]
    else:
        s0, s1 = map(per, cfg["selection"]["select_months"]); e0, e1 = map(per, cfg["selection"]["eval_months"])
    cand = cfg["models"] + [m + "_oosq" for m in cfg["models"]]
    sel_rows = []
    for cat in cfg["data"]["categories"]:
        sel = metrics_table(ft[(ft["category"] == cat) & ft["target"].between(s0, s1) & ft["model"].isin(cand)], cfg, ("model",))
        best = sel.sort_values(cfg["selection"]["criterion"]).iloc[0]["model"]
        ev = metrics_table(ft[(ft["category"] == cat) & ft["target"].between(e0, e1) & ft["model"].isin(cand)], cfg, ("model",))
        ev["category"], ev["selected_on_first_half"] = cat, ev["model"] == best
        ev = ev.merge(sel[["model", cfg["selection"]["criterion"]]].rename(columns={cfg["selection"]["criterion"]: "select_window_pinball"}), on="model")
        sel_rows.append(ev)
    nested = pd.concat(sel_rows, ignore_index=True)
    nested.to_csv(OUT / "nested_selection.csv", index=False)
    chosen = nested[nested["selected_on_first_half"]].set_index("category")["model"].to_dict()

    # ---------------- decision layer
    dc = cfg["decision"]
    dec_frames = []
    for hz in sorted({dc["decision_horizon"], 1}):
        for sp_name, const in [("fbil3m_minus_wacr", None), ("const_%.2f" % dc["sensitivity_constant_spread_pct"], dc["sensitivity_constant_spread_pct"])]:
            d = evaluate_policies(ft, panel, drivers, hz, taus, dc["policy_quantiles"], dc["penalty_multiplier"],
                                  dc["spread_floor_pct"], const)
            dec_frames.append(d.assign(decision_h=hz, spread_basis=sp_name))
    dec = pd.concat(dec_frames, ignore_index=True)
    dec.to_csv(OUT / "decision_costs_detail.csv", index=False)
    dsum = (dec.groupby(["decision_h", "spread_basis", "category", "model", "policy"])
            .agg(total_cost_cr=("total_cost", "sum"), idle_cost_cr=("idle_cost", "sum"), shortfall_cost_cr=("shortfall_cost", "sum"),
                 breaches=("breach", "sum"), n=("breach", "size"), mean_buffer_cr=("buffer", "mean"),
                 shortfall_amt_cr=("shortfall_amt", "sum")).reset_index())
    base = dsum[dsum["policy"] == "smly_baseline"][["decision_h", "spread_basis", "category", "total_cost_cr"]].rename(columns={"total_cost_cr": "baseline_cost_cr"})
    dsum = dsum.merge(base, on=["decision_h", "spread_basis", "category"])
    dsum["cost_vs_baseline_pct"] = 100 * (dsum["total_cost_cr"] / dsum["baseline_cost_cr"] - 1)
    dsum.to_csv(OUT / "decision_costs.csv", index=False)
    paired_vs_baseline(dec).to_csv(OUT / "decision_vs_baseline_bootstrap.csv", index=False)

    # ---------------- driver analysis (descriptive, in-sample, HAC) + robustness for rate-regime breaks
    reg = calendar_driver_regression(panel, drivers)
    reg.to_csv(OUT / "driver_regression.csv", index=False)

    # ---------------- live forecast from latest release
    T = panel["month"].max() if str(cfg["live"]["origin"]) == "latest" else per(cfg["live"]["origin"])
    p, d, as_of = info_set(panel, drivers, T)
    live = []
    for name in cfg["models"]:
        for r in REGISTRY[name](p, d, T, H, taus, cfg["lgbm"] if name.startswith("lgbm") else None):
            live.append({**r, "model": name, "as_of": as_of, "aum_origin": float(p.set_index(["month", "category"]).loc[(T, r["category"]), "aum"])})
    live = pd.DataFrame(live)
    # OOS-quantile versions use all past OOS errors released by as_of
    hist = fc[fc["model"].isin(cfg["models"])].copy()
    hist["err_ratio"] = (hist["actual"] - hist["point"]) / hist["aum_origin"]
    lc = []
    for _, r in live.iterrows():
        past = hist[(hist["model"] == r["model"]) & (hist["category"] == r["category"]) & (hist["h"] == r["h"]) & (hist["target_release"] <= as_of)]
        q = np.sort(np.quantile(past["err_ratio"], taus))
        r2 = r.copy(); r2["model"] = r["model"] + "_oosq"; r2["n_calib"] = len(past)
        for t, v in zip(taus, q):
            r2[f"q_{t}"] = r["point"] + r["aum_origin"] * v
        lc.append(r2)
    live = pd.concat([live, pd.DataFrame(lc)], ignore_index=True)
    for tau in dc["policy_quantiles"]:
        live[f"buffer_q{int(round(tau*100))}"] = live.apply(lambda r: buffer_from_quantiles(r, tau, taus), axis=1)
    act = panel.set_index(["month", "category"])["net"]
    live["baseline_buffer_smly"] = [max(0.0, -act.loc[(t - 12, c)]) for t, c in zip(live["target"], live["category"])]
    live["selected_model"] = live.apply(lambda r: chosen.get(r["category"]) == r["model"], axis=1)
    live.to_csv(OUT / "live_forecast.csv", index=False)

    summary = {"panel": {"start": str(panel["month"].min()), "end": str(panel["month"].max()), "rows": len(panel)},
               "test_targets": [str(test[0]), str(test[-1])], "chosen_model_by_nested_selection": chosen,
               "selection_window": [str(s0), str(s1)], "eval_window": [str(e0), str(e1)],
               "live_origin": str(T), "run_finished_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "live_as_of": str(as_of.date())}
    json.dump(summary, open(OUT / "run_summary.json", "w"), indent=2)
    m_all.to_json(OUT / "metrics.json", orient="records", indent=1)
    print(json.dumps(summary, indent=2))
    from .dashboard import build
    print("dashboard:", build(ROOT, cfg))


