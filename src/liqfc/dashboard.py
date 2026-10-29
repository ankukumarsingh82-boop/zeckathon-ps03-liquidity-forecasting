"""Static one-page fund-manager dashboard. Every number is read from the CSV/JSON files in outputs/ and data/processed/
(file named next to each block) so the page can be audited against machine-readable outputs."""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .claims import BASIS_LABEL, PRIMARY_BASIS, penalty_table, vs_baseline_label  # noqa: E402

CORE = ["sarimax_cal", "ets_ratio", "snaive", "snaive_ratio", "lgbm_q"]


def _fmt(x):
    return f"{x:,.0f}"


def _img(fig) -> str:
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=110, bbox_inches="tight"); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def build(root: Path, cfg: dict) -> Path:
    out = root / "outputs"
    summ = json.load(open(out / "run_summary.json"))
    live = pd.read_csv(out / "live_forecast.csv")
    panel = pd.read_csv(root / "data/processed/amfi_panel.csv")
    drv = pd.read_csv(root / "data/processed/drivers_monthly.csv")
    met = pd.read_csv(out / "metrics.csv")
    nested = pd.read_csv(out / "nested_selection.csv")
    dsum = pd.read_csv(out / "decision_costs.csv")
    boot = pd.read_csv(out / "decision_vs_baseline_bootstrap.csv")
    fcb = pd.read_csv(out / "forecasts_backtest.csv")
    chosen = summ["chosen_model_by_nested_selection"]
    dh = cfg["decision"]["decision_horizon"]
    T = pd.Period(summ["live_origin"], "M")
    dec_month = str(T + dh)

    blocks = []
    for cat in cfg["data"]["categories"]:
        mdl = chosen[cat]
        L = live[(live.category == cat) & (live.model == mdl)].sort_values("h")
        rows = "".join(
            f"<tr><td>{r.target}</td><td>h={r.h}</td><td><b>{_fmt(r.point)}</b></td><td>{_fmt(r['q_0.1'])} … {_fmt(r['q_0.9'])}</td>"
            f"<td>{_fmt(r['q_0.025'])} … {_fmt(r['q_0.975'])}</td><td>{_fmt(r.buffer_q75)}</td><td>{_fmt(r.buffer_q95)}</td>"
            f"<td>{_fmt(r.baseline_buffer_smly)}</td></tr>" for _, r in L.iterrows())
        rec = L[L.target == dec_month].iloc[0]
        alt = live[(live.category == cat) & (live.target == dec_month) & live.model.isin(CORE)][["model", "point", "buffer_q75", "buffer_q95"]]
        alt_txt = ", ".join(f"{a.model}: {_fmt(a.buffer_q75)}" for a in alt.itertuples())
        bp = boot[boot.penalty_basis == PRIMARY_BASIS] if "penalty_basis" in boot else boot
        b = bp[(bp.category == cat) & (bp.model == mdl) & (bp.policy == "q75") & (bp.decision_h == dh) & (bp.spread_basis == "fbil3m_minus_wacr")].iloc[0]
        dp = dsum[dsum.penalty_basis == PRIMARY_BASIS] if "penalty_basis" in dsum else dsum
        ds = dp[(dp.category == cat) & (dp.decision_h == dh) & (dp.spread_basis == "fbil3m_minus_wacr")]
        q95_label = vs_baseline_label(boot, cat, mdl, "q95", dh)
        pen_rows = ""
        for basis in ("spread", "amount"):
            pt = penalty_table(dsum, basis, [mdl], dh)
            pt = pt[pt.category == cat].set_index("policy")
            pen_rows += (f"<tr><td>{BASIS_LABEL[basis]}</td>" + "".join(f"<td>{pt.loc[p, 'total_cost_cr']:,.1f} ({int(pt.loc[p, 'breaches'])})</td>" for p in ("smly_baseline", "q75", "q95"))
                         + f"<td>{'q75' if pt.loc['q75', 'total_cost_cr'] < pt.loc['q95', 'total_cost_cr'] else 'q95'}</td></tr>")
        pol = ds[(ds.model == mdl) & (ds.policy == "q75")].iloc[0]; base = ds[ds.policy == "smly_baseline"].iloc[0]
        # chart: history + backtest h=2 band + live fan
        P = panel[panel.category == cat].assign(m=lambda d: pd.PeriodIndex(d.month, freq="M").to_timestamp())
        P = P[P.m >= "2022-01-01"]
        F = fcb[(fcb.category == cat) & (fcb.model == mdl) & (fcb.h == dh)].assign(m=lambda d: pd.PeriodIndex(d.target, freq="M").to_timestamp())
        Lm = L.assign(m=lambda d: pd.PeriodIndex(d.target, freq="M").to_timestamp())
        fig, ax = plt.subplots(figsize=(7.2, 2.6))
        ax.bar(P.m, P.net / 1e3, width=20, color=["#2a9d8f" if v >= 0 else "#e76f51" for v in P.net], label="actual net flow")
        ax.fill_between(F.m, F["q_0.025"] / 1e3, F["q_0.975"] / 1e3, color="#264653", alpha=.15, label=f"backtest 95% PI (h={dh})")
        ax.fill_between(F.m, F["q_0.1"] / 1e3, F["q_0.9"] / 1e3, color="#264653", alpha=.25, label="backtest 80% PI")
        ax.errorbar(Lm.m, Lm.point / 1e3, yerr=[(Lm.point - Lm["q_0.025"]) / 1e3, (Lm["q_0.975"] - Lm.point) / 1e3], fmt="o", color="#e9c46a", ecolor="#b08900", label="live forecast (95% PI)")
        ax.axhline(0, color="k", lw=.5); ax.set_ylabel("₹ '000 crore"); ax.legend(fontsize=6, ncol=2, loc="lower left"); ax.set_title(f"{cat}: monthly net flow", fontsize=9)
        blocks.append(f"""
<section><h2>{cat}</h2>
<div class=kpi><div><span>Recommended buffer for {dec_month}</span><b>₹{_fmt(rec.buffer_q75)} cr</b><small>75th pct of outflow (newsvendor 3:1), model <code>{mdl}</code></small></div>
<div><span>Prudent (spec 95th pct)</span><b>₹{_fmt(rec.buffer_q95)} cr</b><small>{q95_label}</small></div>
<div><span>Naive baseline (last year same month)</span><b>₹{_fmt(rec.baseline_buffer_smly)} cr</b><small>outflow of {pd.Period(dec_month,'M')-12}</small></div>
<div><span>Backtest cost, q75 vs baseline (24 m)</span><b>₹{pol.total_cost_cr:,.1f} vs {base.total_cost_cr:,.1f} cr</b><small>diff 90% CI [{b.diff_ci90_lo:,.1f}, {b.diff_ci90_hi:,.1f}]; breaches {int(pol.breaches)} vs {int(base.breaches)}</small></div></div>
<table><tr><th>Month</th><th>h</th><th>Point (₹ cr)</th><th>80% PI</th><th>95% PI</th><th>Buffer q75</th><th>Buffer q95</th><th>Baseline</th></tr>{rows}</table>
<table class=pen><tr><th>24-month cost ₹cr (breach months), h={dh}, model <code>{mdl}</code></th><th>Baseline</th><th>q75</th><th>q95</th><th>Cheaper</th></tr>{pen_rows}</table>
<p class=note>Other models' q75 buffer for {dec_month}: {alt_txt}. Source: outputs/live_forecast.csv, decision_costs.csv, decision_vs_baseline_bootstrap.csv.</p>
<img src="{_img(fig)}"></section>""")

    # metrics table
    mt = met[met.model.isin(CORE + [m + "_oosq" for m in CORE])].sort_values(["category", "pinball_core_cr"])
    mrows = "".join(f"<tr{'' if r.model in (chosen[r.category], 'snaive') else ' class=np'}><td>{r.category.split()[0]}</td><td>{r.model}</td><td>{r.MAPE_pct:,.0f}%</td><td>{r.medianAPE_pct:,.0f}%</td><td>{_fmt(r.MAE_cr)}</td>"
                    f"<td>{r.MASE:.2f}</td><td>{_fmt(r.pinball_core_cr)}</td><td>{r.cov80:.0%} [{r.cov80_lo:.0%},{r.cov80_hi:.0%}]</td><td>{r.cov95:.0%} [{r.cov95_lo:.0%},{r.cov95_hi:.0%}]</td></tr>"
                    for r in mt.itertuples())
    d = drv.dropna(subset=["wacr"]).iloc[-1]
    dsep = drv[drv.month == str(T)].iloc[0]
    reg = pd.read_csv(out / "driver_regression.csv")
    rc = reg[(reg.category == "Liquid Fund") & (reg.spec == "calendar")].set_index("variable")["coef_pct_of_aum"]
    html = f"""<!doctype html><html><head><meta charset=utf-8><title>Liquid & Overnight flow dashboard</title><style>
body{{font:13px system-ui,sans-serif;max-width:1100px;margin:16px auto;color:#222}}h1{{font-size:20px;margin:0}}h2{{font-size:15px;margin:14px 0 6px}}
.kpi{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}}.kpi div{{background:#f4f6f7;padding:8px;border-radius:6px}}.kpi span{{display:block;font-size:11px;color:#555}}
.kpi b{{font-size:17px}}.kpi small{{display:block;font-size:10px;color:#666}}table{{border-collapse:collapse;width:100%;margin:6px 0;font-size:12px}}
td,th{{border-bottom:1px solid #ddd;padding:3px 6px;text-align:right}}td:first-child,th:first-child{{text-align:left}}.note{{font-size:11px;color:#555}}
section{{border-top:2px solid #264653;margin-top:10px}}img{{max-width:100%}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.printonly{{display:none}}@media print{{body{{font-size:8.5px;margin:0;max-width:none;line-height:1.25}}h1{{font-size:14px;margin:0 0 2px}}img{{max-height:100px}}h2{{margin:3px 0;font-size:11px}}h3{{font-size:10px;margin:2px 0}}section{{margin-top:3px}}p{{margin:2px 0}}table{{font-size:7.5px;margin:2px 0}}td,th{{padding:0 3px}}tr.np,.screen{{display:none}}.printonly{{display:block}}@page{{size:A4;margin:8mm}}}}</style></head><body>
<h1>Day-1 cash-buffer dashboard — decision for {dec_month}</h1>
<p class=note>Latest AMFI data: {T} (as-of {summ['live_as_of']}; release date imputed as month-end+13 d where the file has no stamp). Forecasts are category-level AMFI net flows (₹ crore; negative = net outflow).
Next AMFI release (Oct-2026 data, ~10 Nov 2026) is the h=1 row. Buffers are category-level; scale by the fund's share of category AUM.</p>
{''.join(blocks)}
<section><h2>Market drivers (latest)</h2><p>{T}: WACR {dsep.wacr:.2f}%, FBIL 3M T-bill {dsep.tbill_3m:.2f}%, avg net LAF today's-ops {dsep.net_liq/1e5:,.2f} lakh cr (negative = RBI absorbing surplus; {int(dsep.net_liq_days)} days),
Nifty 50 month-end {dsep.nifty_close:,.0f}. Latest partial month {d.month}: WACR {d.wacr:.2f}% ({int(d.wacr_days)} days). Advance-tax months (Jun/Sep/Dec/Mar) and March FY-end carry the large outflows
(HAC regression: Liquid {rc['qe_jun_sep_dec']:+.1f}% of prior AUM in Jun/Sep/Dec, {rc['march_fy_end']:+.1f}% in March, {rc['april']:+.1f}% in April; outputs/driver_regression.csv). Source: data/processed/drivers_monthly.csv.</p></section>
<section><h2>Backtest (24 targets Oct-2024…Sep-2026, h=1–3 pooled, refit every origin)</h2>
<table><tr><th>Cat</th><th>Model</th><th>MAPE</th><th>Median APE</th><th>MAE ₹cr</th><th>MASE</th><th>Pinball ₹cr</th><th>Cov 80% [Wilson 95%]</th><th>Cov 95%</th></tr>{mrows}</table>
<p class="note printonly">Printed copy: selected model and seasonal naive per category; the full table is on screen and in reports/tables_metrics.md.</p><p class=note>Source: outputs/metrics.csv. MAPE is unstable when monthly net flow is near zero; read median APE / MASE / pinball. <code>_oosq</code> = PI from past out-of-sample errors. Model chosen per category by pinball on Oct-24…Sep-25 only (outputs/nested_selection.csv).</p></section>
<section><h2>Caveats</h2><p class=note>AMFI figures are first releases. Liquidity data for Sep-2026 come from a third-party DBIE mirror to 17-Sep plus 6 official RBI releases; 5 days missing. Idle-cash spread (T-bill − WACR) was at its 0.10% floor in most months, so costs are small in ₹ and rank-sensitive to the floor. Net (not gross) category outflow is used.</p></section>
</body></html>"""
    p = root / "reports/dashboard.html"
    p.write_text(html, encoding="utf-8")
    return p
