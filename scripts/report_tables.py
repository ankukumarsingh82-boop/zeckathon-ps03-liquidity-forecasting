"""Render markdown tables for reports/report.md from outputs/*.csv (so report numbers trace to CSVs)."""
from pathlib import Path

import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from liqfc.claims import BASIS_LABEL, flip_sentence, penalty_table, q75_vs_q95  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]; O = ROOT / "outputs"
CORE = ["snaive", "snaive_ratio", "ets_ratio", "sarimax_cal", "lgbm_q"]
m = pd.read_csv(O / "metrics.csv"); d = pd.read_csv(O / "decision_costs.csv"); b = pd.read_csv(O / "decision_vs_baseline_bootstrap.csv")
d_all, b_all = d, b
d, b = d[d.penalty_basis == "spread"], b[b.penalty_basis == "spread"]   # primary reading (a) for the main tables
mods = CORE + [x + "_oosq" for x in CORE]
dd = d[(d.decision_h == 2) & (d.spread_basis == "fbil3m_minus_wacr")]
lines = ["| Category | Model | MAPE % | MAPE excl. abs(y)<2k cr % | Median APE % | sMAPE % | MAE ₹cr | MASE | Pinball (core) ₹cr | Cov80 [Wilson] | Cov95 [Wilson] | Cost q75 (a) ₹cr (vs base) | Cost q95 (a) ₹cr (vs base) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in m[m.model.isin(mods)].sort_values(["category", "pinball_core_cr"]).itertuples():
    c = {p: dd[(dd.category == r.category) & (dd.model == r.model) & (dd.policy == p)].iloc[0] for p in ["q75", "q95"]}
    lines.append(f"| {r.category} | {r.model} | {r.MAPE_pct:.0f} | {r.MAPE_excl_small_pct:.0f} | {r.medianAPE_pct:.0f} | {r.sMAPE_pct:.0f} | {r.MAE_cr:,.0f} | {r.MASE:.2f} | {r.pinball_core_cr:,.0f} | "
                 f"{r.cov80:.0%} [{r.cov80_lo:.0%},{r.cov80_hi:.0%}] | {r.cov95:.0%} [{r.cov95_lo:.0%},{r.cov95_hi:.0%}] | "
                 f"{c['q75'].total_cost_cr:.1f} ({c['q75'].cost_vs_baseline_pct:+.0f}%) | {c['q95'].total_cost_cr:.1f} ({c['q95'].cost_vs_baseline_pct:+.0f}%) |")
base = dd[dd.policy == "smly_baseline"]
lines.append("")
lines.append("Baseline (hold last year's same-month net outflow) 24-month cost, h=2 decision: " + "; ".join(
    f"{r.category}: ₹{r.total_cost_cr:.1f} cr ({int(r.breaches)} shortfall months, mean buffer ₹{r.mean_buffer_cr:,.0f} cr)" for r in base.itertuples()))
(ROOT / "reports/tables_metrics.md").write_text("\n".join(lines) + "\n")

lb = pd.read_csv(O / "metrics_by_h.csv")
t = ["| Category | Model | h | MAE ₹cr | Pinball ₹cr | Cov80 | Cov95 |", "|---|---|---|---|---|---|---|"]
for r in lb[lb.model.isin(["snaive", "sarimax_cal", "ets_ratio", "lgbm_q", "lgbm_q_oosq"])].sort_values(["category", "model", "h"]).itertuples():
    t.append(f"| {r.category} | {r.model} | {r.h} | {r.MAE_cr:,.0f} | {r.pinball_core_cr:,.0f} | {r.cov80:.0%} | {r.cov95:.0%} |")
(ROOT / "reports/tables_by_h.md").write_text("\n".join(t) + "\n")

ab = m[m.model.str.startswith("lgbm_q") & ~m.model.str.endswith("_oosq")].sort_values(["category", "model"])
t = ["| Category | LGBM variant | MAE ₹cr | Pinball ₹cr | Cov80 |", "|---|---|---|---|---|"]
t += [f"| {r.category} | {r.model} | {r.MAE_cr:,.0f} | {r.pinball_core_cr:,.0f} | {r.cov80:.0%} |" for r in ab.itertuples()]
(ROOT / "reports/tables_ablation.md").write_text("\n".join(t) + "\n")

ns = pd.read_csv(O / "nested_selection.csv").sort_values(["category", "pinball_core_cr"])
t = ["| Category | Model | Pinball select window (Oct24-Sep25) | Pinball eval window (Oct25-Sep26) | MAE eval | Cov80 eval | Cov95 eval | Selected |", "|---|---|---|---|---|---|---|---|"]
t += [f"| {r.category} | {r.model} | {r.select_window_pinball:,.0f} | {r.pinball_core_cr:,.0f} | {r.MAE_cr:,.0f} | {r.cov80:.0%} | {r.cov95:.0%} | {'**yes**' if r.selected_on_first_half else ''} |" for r in ns.itertuples()]
(ROOT / "reports/tables_nested.md").write_text("\n".join(t) + "\n")

bb = b[(b.decision_h == 2) & (b.spread_basis == "fbil3m_minus_wacr") & b.model.isin(["snaive", "sarimax_cal", "ets_ratio", "lgbm_q", "lgbm_q_oosq", "snaive_ratio"])]
t = ["| Category | Model | Policy | Total cost diff vs baseline ₹cr | 90% block-bootstrap CI | Months cheaper (of 24) |", "|---|---|---|---|---|---|"]
t += [f"| {r.category} | {r.model} | {r.policy} | {r.total_diff_cr:+.1f} | [{r.diff_ci90_lo:+.1f}, {r.diff_ci90_hi:+.1f}] | {r.months_cheaper} |" for r in bb.sort_values(["category", "policy", "total_diff_cr"]).itertuples()]
(ROOT / "reports/tables_decision.md").write_text("\n".join(t) + "\n")

sens = d[d.model.isin(["sarimax_cal", "snaive", "lgbm_q_oosq"]) | (d.policy == "smly_baseline")]
t = ["| decision h | spread basis | Category | Model | Policy | Total cost ₹cr | vs baseline |", "|---|---|---|---|---|---|---|"]
t += [f"| {r.decision_h} | {r.spread_basis} | {r.category} | {r.model} | {r.policy} | {r.total_cost_cr:.1f} | {r.cost_vs_baseline_pct:+.0f}% |" for r in sens.sort_values(["category", "decision_h", "spread_basis", "model", "policy"]).itertuples()]
(ROOT / "reports/tables_decision_sensitivity.md").write_text("\n".join(t) + "\n")

# penalty readings: both tables + the q75-vs-q95 flip check (audit F2)
PM = ["sarimax_cal", "ets_ratio", "snaive", "lgbm_q", "lgbm_q_oosq", "sarimax_cal_oosq"]
t = []
for basis in ("spread", "amount"):
    pt = penalty_table(d_all, basis, PM)
    t += [f"**Penalty {BASIS_LABEL[basis]}**, h=2, realised FBIL 3M − WACR spread (floor 0.10%), 24 months Oct-2024…Sep-2026", "",
          "| Category | Model | Policy | Total cost ₹cr | Idle ₹cr | Shortfall ₹cr | Breach months | Mean buffer ₹cr | vs baseline |", "|---|---|---|---|---|---|---|---|---|"]
    t += [f"| {r.category} | {r.model} | {r.policy} | {r.total_cost_cr:,.1f} | {r.idle_cost_cr:,.1f} | {r.shortfall_cost_cr:,.1f} | {int(r.breaches)} | {r.mean_buffer_cr:,.0f} | {r.cost_vs_baseline_pct:+.0f}% |" for r in pt.itertuples()]
    t.append("")
fl = q75_vs_q95(d_all, PM)
t += ["**Does the q75 vs q95 conclusion flip?** (cheaper policy per reading, 24-month total ₹cr)", "",
      "| Category | Model | (a) q75 | (a) q95 | (a) cheaper | (b) q75 | (b) q95 | (b) cheaper | Flips |", "|---|---|---|---|---|---|---|---|---|"]
t += [f"| {r.category} | {r.model} | {r.q75_spread:,.1f} | {r.q95_spread:,.1f} | {r.cheaper_spread} | {r.q75_amount:,.0f} | {r.q95_amount:,.0f} | {r.cheaper_amount} | {'**yes**' if r.flips else 'no'} |" for r in fl.itertuples()]
t += ["", flip_sentence(d_all, PM)]
(ROOT / "reports/tables_penalty.md").write_text("\n".join(t) + "\n")

# compact version for report.md / PDF: both readings side by side, selected + reference models
CM = ["sarimax_cal", "lgbm_q_oosq", "snaive", "ets_ratio"]
pa, pb = penalty_table(d_all, "spread", CM), penalty_table(d_all, "amount", CM)
mm = pa.merge(pb, on=["category", "model", "policy"], suffixes=("_a", "_b"))
t = ["| Category | Model | Policy | Breach months | Mean buffer ₹cr | (a) cost ₹cr | (a) vs baseline | (b) cost ₹cr | (b) vs baseline |", "|---|---|---|---|---|---|---|---|---|"]
t += [f"| {r.category} | {r.model} | {r.policy} | {int(r.breaches_a)} | {r.mean_buffer_cr_a:,.0f} | {r.total_cost_cr_a:,.1f} | {r.cost_vs_baseline_pct_a:+.0f}% | {r.total_cost_cr_b:,.0f} | {r.cost_vs_baseline_pct_b:+.0f}% |" for r in mm.itertuples()]
(ROOT / "reports/tables_penalty_compact.md").write_text("\n".join(t) + "\n")
t = ["| Category | Model | (a) q75 | (a) q95 | (a) cheaper | (b) q75 | (b) q95 | (b) cheaper | Flips |", "|---|---|---|---|---|---|---|---|---|"]
t += [f"| {r.category} | {r.model} | {r.q75_spread:,.1f} | {r.q95_spread:,.1f} | {r.cheaper_spread} | {r.q75_amount:,.0f} | {r.q95_amount:,.0f} | {r.cheaper_amount} | {'**yes**' if r.flips else 'no'} |" for r in fl.itertuples()]
(ROOT / "reports/tables_penalty_flip.md").write_text("\n".join(t) + "\n")

lv = pd.read_csv(O / "live_forecast.csv")
t = ["| Category | Model | Month | h | Point ₹cr | 80% PI | 95% PI | Buffer q75 | Buffer q95 | Baseline buffer |", "|---|---|---|---|---|---|---|---|---|---|"]
t += [f"| {r['category']} | {r['model']}{' (selected)' if r['selected_model'] else ''} | {r['target']} | {r['h']} | {r['point']:,.0f} | [{r['q_0.1']:,.0f}, {r['q_0.9']:,.0f}] | [{r['q_0.025']:,.0f}, {r['q_0.975']:,.0f}] | {r['buffer_q75']:,.0f} | {r['buffer_q95']:,.0f} | {r['baseline_buffer_smly']:,.0f} |"
      for _, r in lv[lv.model.isin(["snaive", "sarimax_cal", "ets_ratio", "lgbm_q", "lgbm_q_oosq", "sarimax_cal_oosq"])].sort_values(["category", "model", "h"]).iterrows()]
(ROOT / "reports/tables_live.md").write_text("\n".join(t) + "\n")
sel = lv[lv.selected_model.astype(bool)].sort_values(["category", "h"])
t = ["| Category | Model | Month | h | Point ₹cr | 80% PI | 95% PI | Buffer q75 | Buffer q95 | Baseline buffer |", "|---|---|---|---|---|---|---|---|---|---|"]
t += [f"| {r['category']} | {r['model']} | {r['target']} | {r['h']} | {r['point']:,.0f} | [{r['q_0.1']:,.0f}, {r['q_0.9']:,.0f}] | [{r['q_0.025']:,.0f}, {r['q_0.975']:,.0f}] | {r['buffer_q75']:,.0f} | {r['buffer_q95']:,.0f} | {r['baseline_buffer_smly']:,.0f} |" for _, r in sel.iterrows()]
(ROOT / "reports/tables_live_selected.md").write_text("\n".join(t) + "\n")
print("ok")
