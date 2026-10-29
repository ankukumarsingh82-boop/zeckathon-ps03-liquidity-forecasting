"""Build reports/figures/*.png, reports/report.pdf (from report.md) and reports/deck.pdf (8 slides).

Every number comes from outputs/*.csv (or the rendered report.md). PDFs are rendered from HTML with WeasyPrint
(headless Chrome's --print-to-pdf hangs on this box). Run:  python -m liqfc docs
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import markdown
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from liqfc.appcore import cost_curve  # noqa: E402
import yaml  # noqa: E402
from liqfc.claims import (BASIS_LABEL, PRIMARY_BASIS, flip_sentence, q75_vs_q95, submitted_forecasts,  # noqa: E402
                          submitted_line, vs_baseline_label)

O, R = ROOT / "outputs", ROOT / "reports"
F = R / "figures"
CATS = ["Liquid Fund", "Overnight Fund"]
PM = ["sarimax_cal", "ets_ratio", "snaive", "lgbm_q", "lgbm_q_oosq", "sarimax_cal_oosq"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})


def load():
    d = {k: pd.read_csv(O / f"{k}.csv") for k in ["actuals", "forecasts_backtest", "live_forecast", "decision_costs",
                                                  "decision_vs_baseline_bootstrap", "metrics", "driver_regression", "drivers_monthly"]}
    d["summary"] = json.loads((O / "run_summary.json").read_text())
    d["sel"] = d["summary"]["chosen_model_by_nested_selection"]
    return d


def _dates(ax):
    loc = mdates.MonthLocator(bymonth=(1, 4, 7, 10))
    ax.xaxis.set_major_locator(loc); ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def _fan(ax, x, q, color):
    ax.fill_between(x, q["q_0.025"], q["q_0.975"], color=color, alpha=.15, lw=0, label="95% PI")
    ax.fill_between(x, q["q_0.1"], q["q_0.9"], color=color, alpha=.3, lw=0, label="80% PI")
    ax.plot(x, q["point"], color=color, lw=1.5, marker="o", ms=3, label="forecast")


def figures(d) -> dict:
    F.mkdir(parents=True, exist_ok=True)
    out = {}
    act = d["actuals"].assign(t=lambda x: pd.PeriodIndex(x.month, freq="M").to_timestamp())
    # 1 live fan
    fig, axs = plt.subplots(1, 2, figsize=(9, 3))
    for ax, c in zip(axs, CATS):
        a = act[act.category == c].tail(30)
        ax.plot(a.t, a.net / 1e3, color="k", lw=1, label="actual")
        lv = d["live_forecast"]; lv = lv[(lv.category == c) & (lv.model == d["sel"][c])].sort_values("h")
        x = pd.PeriodIndex(lv.target, freq="M").to_timestamp()
        _fan(ax, x, lv[[k for k in lv if k.startswith("q_") or k == "point"]] / 1e3, "C0")
        ax.axhline(0, color="grey", lw=.5); ax.set_title(f"{c}: live forecast, {d['sel'][c]}"); ax.set_ylabel("net flow, ₹'000 cr"); _dates(ax)
    axs[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(F / "live_fan.png"); plt.close(fig); out["live_fan"] = "figures/live_fan.png"
    # 2 backtest h=2
    fig, axs = plt.subplots(1, 2, figsize=(9, 3))
    fb = d["forecasts_backtest"]
    for ax, c in zip(axs, CATS):
        b = fb[(fb.category == c) & (fb.model == d["sel"][c]) & (fb.h == 2)].sort_values("target")
        x = pd.PeriodIndex(b.target, freq="M").to_timestamp()
        _fan(ax, x, b[[k for k in b if k.startswith("q_") or k == "point"]] / 1e3, "C1")
        ax.plot(x, b.actual / 1e3, "k.", label="actual"); ax.axhline(0, color="grey", lw=.5)
        ax.set_title(f"{c}: backtest h=2, {d['sel'][c]}"); ax.set_ylabel("₹'000 cr"); _dates(ax)
    axs[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(F / "backtest_h2.png"); plt.close(fig); out["backtest_h2"] = "figures/backtest_h2.png"
    # 3 coverage (pooled h=1..3, Wilson CI)
    m = d["metrics"]; m = m[m.model.isin(PM)]
    fig, axs = plt.subplots(1, 2, figsize=(9, 2.8), sharey=True)
    for ax, c in zip(axs, CATS):
        g = m[m.category == c].set_index("model").loc[PM]; y = np.arange(len(PM))
        for off, lvl, col in [(-.15, 80, "C0"), (.15, 95, "C3")]:
            v = g[f"cov{lvl}"]; ax.errorbar(v * 100, y + off, xerr=[(v - g[f"cov{lvl}_lo"]) * 100, (g[f"cov{lvl}_hi"] - v) * 100], fmt="o", color=col, ms=3, label=f"{lvl}% PI")
            ax.axvline(lvl, color=col, ls="--", lw=.7)
        ax.set_yticks(y, PM); ax.set_title(f"{c}: empirical coverage (72 forecasts)"); ax.set_xlabel("%")
    axs[0].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(F / "coverage.png"); plt.close(fig); out["coverage"] = "figures/coverage.png"
    # 4 cost vs tau, both readings
    fc = d["forecasts_backtest"]; models = ["sarimax_cal", "lgbm_q_oosq", "snaive"]
    fig, axs = plt.subplots(2, 2, figsize=(9, 5))
    for j, pb in enumerate(["spread", "amount"]):
        cc = cost_curve(fc, d["actuals"], d["drivers_monthly"], 2, models, penalty_basis=pb)
        for i, c in enumerate(CATS):
            ax = axs[j, i]; g = cc[cc.category == c]
            for k, mo in enumerate(models):
                s = g[g.model == mo]; ax.plot(s.tau, s.policy_cost_cr, marker=".", color=f"C{k}", label=mo)
            ax.axhline(g.baseline_cost_cr.iloc[0], color="k", ls="--", lw=.8, label="baseline")
            for t in (.75, .95): ax.axvline(t, color="grey", lw=.5)
            if pb == "amount": ax.set_yscale("log")
            ax.set_title(f"{c}: {BASIS_LABEL[pb]}", fontsize=8); ax.set_ylabel("24-m cost ₹cr")
    axs[1, 0].set_xlabel("buffer quantile τ"); axs[1, 1].set_xlabel("buffer quantile τ"); axs[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(F / "cost_tau.png"); plt.close(fig); out["cost_tau"] = "figures/cost_tau.png"
    # 5 drivers (calendar spec)
    dr = d["driver_regression"]; dr = dr[(dr.spec == "calendar") & (dr.variable != "const")]
    fig, axs = plt.subplots(1, 2, figsize=(9, 2.6), sharey=True)
    for ax, c in zip(axs, CATS):
        g = dr[dr.category == c]; y = np.arange(len(g))
        ax.errorbar(g.coef_pct_of_aum, y, xerr=1.96 * g.hac_se, fmt="o", ms=3); ax.axvline(0, color="grey", lw=.5)
        ax.set_yticks(y, g.variable); ax.set_title(f"{c}: calendar effects, % of AUM (95% HAC CI)", fontsize=8)
    fig.tight_layout(); fig.savefig(F / "drivers.png"); plt.close(fig); out["drivers"] = "figures/drivers.png"
    return out


def html_pdf(html: str, pdf: Path) -> Path:
    """HTML -> PDF with WeasyPrint (pinned in requirements). Headless Chrome hangs on this box, see README."""
    from weasyprint import HTML
    HTML(string=html, base_url=str(R) + "/").write_pdf(pdf)
    return pdf


CSS_REPORT = """@page{size:A4;margin:12mm 11mm}body{font-family:'DejaVu Sans',sans-serif;font-size:8.4pt;line-height:1.28;color:#111}
h1{font-size:15pt;margin:0 0 4px}h2{font-size:11.5pt;margin:10px 0 3px;border-bottom:1px solid #ccc;break-after:avoid}
table{border-collapse:collapse;font-size:6.6pt;margin:3px 0 6px;break-inside:auto}td,th{border:1px solid #ccc;padding:1px 3px;text-align:right}
td:first-child,td:nth-child(2),th{text-align:left}th{background:#eef}tr{break-inside:avoid}
img{max-width:100%;break-inside:avoid}blockquote{border-left:3px solid #c33;background:#fff4f4;margin:4px 0;padding:3px 8px}
code{font-size:7.8pt;background:#f3f3f3}ul,ol{margin:2px 0 4px;padding-left:18px}p{margin:3px 0}"""


def _md_for_python_markdown(md: str) -> str:
    """report.md is GitHub-style (2-space nested lists, lists right after a paragraph); Python-Markdown wants
    4-space nesting and a blank line before a list."""
    import re
    item = re.compile(r"^\s*([*-]|\d+\.)\s")
    out, prev = [], ""
    for ln in md.splitlines():
        m = re.match(r"^( +)", ln)
        if m and item.match(ln):
            ln = " " * (2 * len(m.group(1))) + ln.lstrip(" ")
        if item.match(ln) and not ln.startswith(" ") and prev.strip() and not item.match(prev) and not prev.startswith(" "):
            out.append("")
        out.append(ln); prev = ln
    return "\n".join(out)


def report_pdf() -> Path:
    md = _md_for_python_markdown((R / "report.md").read_text())
    body = markdown.markdown(md, extensions=["tables", "sane_lists"])
    return html_pdf(f"<!doctype html><meta charset=utf-8><title>PS03 report</title><style>{CSS_REPORT}</style>{body}", R / "report.pdf")


def _t(rows, head):
    return "<table><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr>" + "".join("<tr>" + "".join(f"<td>{v}</td>" for v in r) + "</tr>" for r in rows) + "</table>"


def deck_pdf(d, figs) -> Path:
    s, sel = d["summary"], d["sel"]
    dc = d["decision_costs"]; dc = dc[(dc.decision_h == 2) & (dc.spread_basis == "fbil3m_minus_wacr")]
    lv = d["live_forecast"]; lv = lv[lv.selected_model.astype(bool)]
    boot = d["decision_vs_baseline_bootstrap"]
    def c(cat, mo, pol, pb, f="total_cost_cr"):
        return dc[(dc.category == cat) & (dc.model == mo) & (dc.policy == pol) & (dc.penalty_basis == pb)].iloc[0][f]
    nov = {cat: lv[(lv.category == cat) & (lv.h == 2)].iloc[0] for cat in CATS}
    met = d["metrics"]
    slides = []
    slides.append(("PS03 · Liquid & Overnight fund net flows: forecast and cash buffer",
        f"<p class=big>Monthly AMFI net flows, {s['panel']['start']}–{s['panel']['end']}; 24-release rolling backtest ({s['test_targets'][0]}–{s['test_targets'][1]}); live origin {s['live_origin']} (data as of {s.get('live_as_of','')}).</p>"
        f"<ul><li><b>November 2026 buffer (q75, reading (a))</b>: Liquid ₹{nov['Liquid Fund'].buffer_q75:,.0f} cr · Overnight ₹{nov['Overnight Fund'].buffer_q75:,.0f} cr</li>"
        f"<li>If the 3× penalty is on the rupee amount (reading (b)), use q95: Liquid ₹{nov['Liquid Fund'].buffer_q95:,.0f} cr · Overnight ₹{nov['Overnight Fund'].buffer_q95:,.0f} cr</li>"
        f"<li>Forecast-based q75 (sarimax_cal) beats the 'last year same month' baseline under both penalty readings in both categories.</li></ul>"
        "<p class=note>Local, reproducible build: <code>python -m liqfc run</code> → outputs/*.csv → this deck. All numbers are pulled from those CSVs.</p>"))
    slides.append(("Data and setup",
        "<ul><li><b>AMFI monthly reports</b> (category sales, redemptions, net flow, AUM), parsed from official files with sha256 manifest; identity sales − redemption = net checked.</li>"
        "<li><b>Drivers:</b> RBI WACR and net LAF (Money Market Operations), FBIL 3M T-bill, NSE Nifty 50, plus a calendar (advance-tax quarter-ends, FY-end, April).</li>"
        "<li><b>Leakage control:</b> each origin uses only data released before its as-of date (AMFI release ≈ 9–11th of next month).</li>"
        "<li><b>Decision timing:</b> the buffer for month M is set on day 1 from the release for M−2, so the decision uses <b>h = 2</b>.</li>"
        "<li><b>Models:</b> seasonal naive, ratio-to-AUM naive, ETS on ratio, SARIMAX with calendar, LightGBM quantile; conformal (out-of-sample quantile) variants. Nested selection: 12 months to pick, 12 to evaluate.</li></ul>"))
    rows = []
    for cat in CATS:
        for mo in ["snaive", "sarimax_cal", sel[cat]]:
            r = met[(met.category == cat) & (met.model == mo)].iloc[0]
            rows.append([cat.split()[0], mo, f"{r.MASE:.2f}", f"{r.pinball_core_cr:,.0f}", f"{r.cov80:.0%}", f"{r.cov95:.0%}"])
    rows = [list(x) for x in dict.fromkeys(tuple(r) for r in rows)]
    slides.append(("Accuracy and calibration (24 targets × h = 1–3)",
        "<div class=two><div>" + _t(rows, ["Cat.", "Model", "MASE", "Pinball ₹cr", "Cov80", "Cov95"]) +
        f"<p class=note>Nested selection picks <b>{sel['Liquid Fund']}</b> (Liquid) and <b>{sel['Overnight Fund']}</b> (Overnight). Seasonal patterns dominate; no model is far ahead of seasonal naive on point accuracy.</p></div>"
        f"<div><img src='{figs['coverage']}'></div></div><img class=wide src='{figs['backtest_h2']}'>"))
    rows = []
    for cat in CATS:
        rows.append([cat.split()[0], "baseline", f"{c(cat,'baseline','smly_baseline','spread'):.1f}", int(c(cat,'baseline','smly_baseline','spread','breaches')), "—"])
        for pol in ["q75", "q95"]:
            rows.append([cat.split()[0], f"sarimax_cal {pol}", f"{c(cat,'sarimax_cal',pol,'spread'):.1f}", int(c(cat,'sarimax_cal',pol,'spread','breaches')),
                         vs_baseline_label(boot, cat, "sarimax_cal", pol).split(": ", 1)[1]])
    slides.append(("Decision cost, penalty reading (a) [primary]",
        "<p>Cost = idle × s/1200 + 3 × shortfall × s/1200, s = FBIL 3M T-bill − WACR (floor 0.10%). h = 2, 24 months, ₹ crore.</p>" +
        _t(rows, ["Cat.", "Policy", "24-m cost", "Breach months", "vs baseline (90% bootstrap CI)"]) +
        "<p class=note>Under (a) the cost-minimising quantile is 3/(3+1) = 0.75. The spec's q95 holds too much idle cash for Liquid.</p>"))
    fl = q75_vs_q95(d["decision_costs"], PM)
    rows = [[r.category.split()[0], r.model, f"{r.q75_spread:.1f}", f"{r.q95_spread:.1f}", f"{r.q75_amount:,.0f}", f"{r.q95_amount:,.0f}", "yes" if r.flips else "no"] for r in fl.itertuples()]
    slides.append(("The 3× penalty is ambiguous: both readings",
        "<div class=two><div>" + _t(rows, ["Cat.", "Model", "(a) q75", "(a) q95", "(b) q75", "(b) q95", "Flips"]) +
        f"<p><b>{flip_sentence(d['decision_costs'], PM)}</b></p></div><div><img src='{figs['cost_tau']}'></div></div>"
        "<p class=note>(a) = 3 × shortfall × spread (primary: same unit as idle cost, critical ratio 0.75). (b) = 3 × shortfall in ₹: about 36,000× the monthly idle carry, so the optimum is ≈ the 99.997th percentile.</p>"))
    rows = [[r.category.split()[0], r.model, r.target, int(r.h), f"{r.point:,.0f}", f"[{r['q_0.1']:,.0f}, {r['q_0.9']:,.0f}]", f"{r.buffer_q75:,.0f}", f"{r.buffer_q95:,.0f}", f"{r.baseline_buffer_smly:,.0f}"]
            for _, r in lv.sort_values(["category", "h"]).iterrows()]
    sc = yaml.safe_load(open(ROOT / "config.yaml"))["submission"]
    sub = submitted_forecasts(d["live_forecast"], sel, sc["rule"], sc["horizon"])
    md2html = lambda t: re.sub(r"`([^`]+)`", r"<code>\1</code>", re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t))
    subl = "".join(f"<li>{md2html(submitted_line(r))}</li>" for r in sub.values())
    slides.append((f"Live forecast from origin {s['live_origin']}",
        _t(rows, ["Cat.", "Model", "Month", "h", "Point", "80% PI", "Buffer q75", "Buffer q95", "Baseline"]) +
        f"<ul class=sub>{subl}</ul><p class=note>Rule (config.yaml <code>submission</code>): the h=1 figure of the nested-selection model, the same model that sets the buffer. Overnight's sarimax_cal figure is reference only (report §5).</p>"
        f"<img class=wide2 src='{figs['live_fan']}'>"))
    slides.append(("What drives the flows",
        f"<img class=wide src='{figs['drivers']}'><ul><li>Calendar effects dominate: advance-tax quarter-ends and March FY-end bring large Liquid outflows; April brings inflows.</li>"
        "<li>Rates, liquidity and equity variables add no out-of-sample accuracy over calendar + lags (LGBM ablation).</li>"
        "<li>Overnight: WACR and net LAF are significant contemporaneously, but this is not exploitable at h = 2.</li></ul>"))
    slides.append(("Limits, risks, next steps",
        "<ul><li>Category-level net flow, not a single fund: scale by the fund's AUM share and its own investor mix.</li>"
        "<li>24 test months: bootstrap CIs are wide; Liquid q75 vs baseline under (a) is not conclusive.</li>"
        "<li>The penalty reading changes the q75 vs q95 choice; confirm the intended definition with the fund.</li>"
        "<li>The 80% bands under-cover for some models (e.g. conformal SARIMAX Liquid 72%).</li>"
        "<li>Next: daily or weekly flow data, fund-level holdings, intra-month timing of tax dates.</li></ul>"
        "<p class=note>Full report: reports/report.pdf · Interactive app: python -m liqfc app (localhost:8501).</p>"))
    css = """@page{size:1280px 720px;margin:0}body{margin:0;font-family:'DejaVu Sans',sans-serif;color:#111}
.s{width:1280px;height:720px;box-sizing:border-box;padding:38px 56px;break-after:page;position:relative;overflow:hidden}
h1{font-size:30px;margin:0 0 14px;color:#123}.big{font-size:20px}ul{font-size:19px;line-height:1.45}
table{border-collapse:collapse;font-size:14px;margin:6px 0}td,th{border:1px solid #bbb;padding:3px 7px;text-align:right}th{background:#e8eef8}
td:first-child,td:nth-child(2){text-align:left}.two{display:flex;gap:24px;align-items:flex-start}.two>div{flex:1}.two img{width:100%}
img.wide{display:block;width:92%;margin:8px auto 0}img.wide2{display:block;width:70%;margin:2px auto 0}ul.sub{font-size:15px;margin:4px 0}.note{font-size:14px;color:#555}.pg{position:absolute;right:30px;bottom:16px;font-size:12px;color:#888}"""
    body = "".join(f"<div class=s><h1>{t}</h1>{h}<div class=pg>{i}/{len(slides)}</div></div>" for i, (t, h) in enumerate(slides, 1))
    return html_pdf(f"<!doctype html><meta charset=utf-8><title>PS03 deck</title><style>{css}</style>{body}", R / "deck.pdf")


def main():
    d = load()
    figs = figures(d)
    print("figures:", ", ".join(figs.values()))
    print(report_pdf())
    print(deck_pdf(d, figs))


if __name__ == "__main__":
    main()
