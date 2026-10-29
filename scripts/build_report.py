"""Render reports/report.md from reports/report_template.md + outputs/*.csv (audit F8: prose numbers come from data).

Tokens in the template:
  {{include:tables_x.md}}                         -> contents of reports/tables_x.md
  {{fact:key}}                                    -> liqfc.claims.report_facts(...)[key]  (plus a few keys built here)
  {{cost:<category>|<model>|<policy>|<basis>|<field>}}  -> one cell of outputs/decision_costs.csv (h=2, realised spread)
       field: cost (₹cr, 1 dp) | pct (vs baseline, signed %) | breaches
An unknown token raises, so the report cannot silently ship a stale or missing number.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import yaml  # noqa: E402
from liqfc.claims import (ci_sentence, flip_sentence, q75_vs_q95, report_facts, submitted_forecasts,  # noqa: E402
                          submitted_line, vs_baseline_label)

O, R = ROOT / "outputs", ROOT / "reports"
CORE = ["snaive", "snaive_ratio", "ets_ratio", "sarimax_cal", "lgbm_q"]
ALL = CORE + [m + "_oosq" for m in CORE]
PM = ["sarimax_cal", "ets_ratio", "snaive", "lgbm_q", "lgbm_q_oosq", "sarimax_cal_oosq"]


def facts() -> dict:
    met, nested = pd.read_csv(O / "metrics.csv"), pd.read_csv(O / "nested_selection.csv")
    dsum, boot = pd.read_csv(O / "decision_costs.csv"), pd.read_csv(O / "decision_vs_baseline_bootstrap.csv")
    summ = json.loads((O / "run_summary.json").read_text())
    f = report_facts(met, nested, dsum, boot, ALL)
    f["n_models"] = str(len(ALL))
    f["flip_sentence"] = flip_sentence(dsum, PM)
    fl = q75_vs_q95(dsum, PM)
    f["flip_n"] = f"{int(fl.flips.sum())} of {len(fl)}"
    for cat in ("Liquid Fund", "Overnight Fund"):
        k = cat.split()[0].lower()
        m = summ["chosen_model_by_nested_selection"][cat]
        f[f"{k}_q95_label"] = vs_baseline_label(boot, cat, m, "q95")
        f[f"{k}_q75_label"] = vs_baseline_label(boot, cat, m, "q75")
        f[f"{k}_sarimax_q75_ci"] = ci_sentence(boot, cat, "sarimax_cal", "q75")
        ev = nested[nested.category == cat].sort_values("pinball_core_cr").reset_index(drop=True)
        for mo in ev.model:
            f[f"{k}_evalrank_{mo}"] = f"{int(ev.index[ev.model == mo][0]) + 1} of {len(ev)}"
    # live: the one 'submitted' figure per category (config.yaml `submission`) and the buffers
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))["submission"]
    live = pd.read_csv(O / "live_forecast.csv")
    sub = submitted_forecasts(live, summ["chosen_model_by_nested_selection"], cfg["rule"], cfg["horizon"])
    for cat, r in sub.items():
        k = cat.split()[0].lower()
        f[f"{k}_submitted_line"] = submitted_line(r)
        f[f"{k}_submitted_model"] = r["model"]
    for cat in ("Liquid Fund", "Overnight Fund"):
        k = cat.split()[0].lower()
        for mo in live[live.category == cat].model.unique():
            for h in (1, 2, 3):
                r = live[(live.category == cat) & (live.model == mo) & (live.h == h)].iloc[0]
                for col in ("point", "buffer_q75", "buffer_q95", "baseline_buffer_smly", "q_0.1", "q_0.9"):
                    f[f"{k}_{mo}_h{h}_{col}"] = f"{r[col]:+,.0f}" if col == "point" else f"{r[col]:,.0f}"
    lc = f["liquid_sarimax_q75_ci"]
    f["liquid_a_ci_note"] = ("The point estimates favour the forecast buffer, but under reading (a) the Liquid gain is not statistically clear: its "
                             if "includes 0" in lc else "Under reading (a) the Liquid gain is statistically clear: its ") + lc
    man = pd.read_csv(ROOT / "data/raw/MANIFEST.csv")
    f["manifest_files"] = str(len(man))
    return f


def render(template: str, f: dict, dsum: pd.DataFrame) -> str:
    d = dsum[(dsum.decision_h == 2) & (dsum.spread_basis == "fbil3m_minus_wacr")]

    def cost(arg: str) -> str:
        cat, model, pol, basis, field = arg.split("|")
        r = d[(d.category == cat) & (d.model == model) & (d.policy == pol) & (d.penalty_basis == basis)]
        if len(r) != 1:
            raise KeyError(f"cost token matched {len(r)} rows: {arg}")
        r = r.iloc[0]
        return {"cost": f"{r.total_cost_cr:,.1f}", "pct": f"{r.cost_vs_baseline_pct:+.0f}%", "breaches": str(int(r.breaches))}[field]

    def sub(m: re.Match) -> str:
        kind, arg = m.group(1), m.group(2)
        if kind == "include":
            return (R / arg).read_text().strip()
        if kind == "fact":
            return f[arg]
        if kind == "cost":
            return cost(arg)
        raise KeyError(m.group(0))

    out = re.sub(r"\{\{(include|fact|cost):([^}]+)\}\}", sub, template)
    if "{{" in out:
        raise ValueError("unrendered token: " + out[out.index("{{"):][:80])
    return out


def main() -> Path:
    t = (R / "report_template.md").read_text()
    out = render(t, facts(), pd.read_csv(O / "decision_costs.csv"))
    p = R / "report.md"
    p.write_text("<!-- GENERATED by scripts/build_report.py from reports/report_template.md + outputs/*.csv. Edit the template, not this file. -->\n" + out)
    return p


if __name__ == "__main__":
    print(main())
