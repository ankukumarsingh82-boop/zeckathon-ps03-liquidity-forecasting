"""Interactive local dashboard. Reads ONLY files in outputs/ (written by `python -m liqfc run`).
Launch: python -m liqfc app   (http://127.0.0.1:8501)"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from liqfc.appcore import buffers, coverage_table, cost_curve, cost_summary, load_outputs, recompute_costs  # noqa: E402

OUT = ROOT / "outputs"
CORE = ["snaive", "snaive_ratio", "ets_ratio", "sarimax_cal", "lgbm_q"]
st.set_page_config(page_title="Liquid & Overnight flows - cash buffer", layout="wide")


@st.cache_data(show_spinner=False)
def data(stamp: float):
    return load_outputs(OUT)


def _stamp():
    return max((OUT / f).stat().st_mtime for f in ["run_summary.json", "forecasts_backtest.csv", "live_forecast.csv"] if (OUT / f).exists()) if OUT.exists() else 0.0


try:
    D = data(_stamp())
except FileNotFoundError as e:
    st.error(str(e)); st.stop()

S = D["run_summary"]
cats = sorted(D["actuals"]["category"].unique())
fc = D["forecasts_backtest"]
all_models = sorted(fc["model"].unique(), key=lambda m: (m.replace("_oosq", "") not in CORE, m))
chosen = S["chosen_model_by_nested_selection"]

# ---------------- sidebar controls
st.sidebar.header("Decision settings")
tau = st.sidebar.slider("Buffer quantile τ (of outflow)", 0.50, 0.975, 0.75, 0.025, key="tau",
                        help="0.75 = newsvendor optimum for a 3:1 shortfall:idle penalty; 0.95 = spec example")
h_dec = st.sidebar.selectbox("Decision horizon h (day-1 of M uses release M-2 → h=2)", [1, 2, 3], index=1, key="h")
spread_mode = st.sidebar.radio("Idle-cash spread", ["realised", "constant"], key="spread_mode",
                               format_func=lambda x: "Realised FBIL 3M T-bill − WACR (floored)" if x == "realised" else "Constant")
floor = st.sidebar.slider("Spread floor (% p.a.)", 0.0, 0.5, 0.10, 0.01, key="floor", disabled=spread_mode == "constant")
const = st.sidebar.slider("Constant spread (% p.a.)", 0.05, 2.0, 0.50, 0.05, key="const", disabled=spread_mode == "realised")
pen = st.sidebar.slider("Shortfall penalty multiplier", 1.0, 6.0, 3.0, 0.5, key="pen")
st.sidebar.caption(f"Data through **{S['live_origin']}** (as-of {S['live_as_of']}). Pipeline run {S.get('run_finished_utc', 'n/a')} UTC.")
kw = dict(spread_mode=spread_mode, floor_pct=floor, const_pct=const, penalty=pen)

st.title("Liquid & Overnight fund flows: forecast and cash buffer")
tabs = st.tabs(["Forecast & buffer", "Backtest vs baseline", "Coverage", "Drivers", "Decision cost", "Data & refresh"])

# ---------------- 1. forecast
with tabs[0]:
    live = D["live_forecast"]
    T = pd.Period(S["live_origin"], "M"); dec_m = str(T + h_dec)
    st.markdown(f"Origin **{T}** (released as-of {S['live_as_of']}). Buffer decision month **{dec_m}** uses the h={h_dec} forecast. "
                f"Next AMFI release (data for {T+1}) is the h=1 row. Category-level ₹ crore; negative = net outflow.")
    cols = st.columns(len(cats))
    for col, cat in zip(cols, cats):
        with col:
            opts = sorted(live[live.category == cat]["model"].unique())
            m = st.selectbox(f"{cat} model", opts, index=opts.index(chosen[cat]) if chosen.get(cat) in opts else 0, key=f"m_{cat}")
            L = live[(live.category == cat) & (live.model == m)].sort_values("h").copy()
            L["buffer_tau"] = buffers(L, tau)
            row = L[L.target == dec_m].iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric(f"Buffer τ={tau:.3f} for {dec_m}", f"₹{row.buffer_tau:,.0f} cr")
            c2.metric("Baseline (same month last yr)", f"₹{row.baseline_buffer_smly:,.0f} cr")
            c3.metric(f"Point forecast {dec_m}", f"₹{row.point:,.0f} cr")
            A = D["actuals"][D["actuals"].category == cat].tail(36)
            A = A.assign(date=pd.PeriodIndex(A.month, freq="M").to_timestamp(), sign=np.where(A.net >= 0, "inflow", "outflow"))
            Lc = L.assign(date=pd.PeriodIndex(L.target, freq="M").to_timestamp())
            bars = alt.Chart(A).mark_bar(opacity=.7).encode(x=alt.X("date:T", title=None), y=alt.Y("net:Q", title="₹ crore"),
                                                            color=alt.Color("sign:N", scale=alt.Scale(domain=["inflow", "outflow"], range=["#2a9d8f", "#e76f51"]), legend=None),
                                                            tooltip=["month", alt.Tooltip("net:Q", format=",.0f")])
            b95 = alt.Chart(Lc).mark_rule(strokeWidth=4, opacity=.35, color="#264653").encode(x="date:T", y="q_0.025:Q", y2="q_0.975:Q")
            b80 = alt.Chart(Lc).mark_rule(strokeWidth=10, opacity=.6, color="#264653").encode(x="date:T", y="q_0.1:Q", y2="q_0.9:Q")
            pt = alt.Chart(Lc).mark_point(filled=True, size=80, color="#e9c46a").encode(
                x="date:T", y="point:Q", tooltip=["target", alt.Tooltip("point:Q", format=",.0f"), alt.Tooltip("q_0.1:Q", format=",.0f"),
                                                  alt.Tooltip("q_0.9:Q", format=",.0f"), alt.Tooltip("q_0.025:Q", format=",.0f"), alt.Tooltip("q_0.975:Q", format=",.0f")])
            st.altair_chart((bars + b95 + b80 + pt).properties(height=260, title=f"{cat}: actuals and 3-month forecast (80% thick / 95% thin band)"), use_container_width=True)
            show = L[["target", "h", "point", "q_0.1", "q_0.9", "q_0.025", "q_0.975", "buffer_tau", "buffer_q75", "buffer_q95", "baseline_buffer_smly"]]
            st.dataframe(show.rename(columns={"q_0.1": "80% lo", "q_0.9": "80% hi", "q_0.025": "95% lo", "q_0.975": "95% hi",
                                              "buffer_tau": f"buffer τ={tau:.3f}", "baseline_buffer_smly": "baseline buffer"}).style.format(precision=0, thousands=","),
                         hide_index=True, use_container_width=True)
    st.caption("Source: outputs/live_forecast.csv, outputs/actuals.csv. Buffers are category-level; scale by your fund's share of category AUM.")

# ---------------- 2. backtest metrics
with tabs[1]:
    M = D["metrics"]
    base = M[M.model == "snaive"].set_index("category")
    M = M.assign(pinball_skill_vs_snaive=lambda d: 1 - d.pinball_core_cr / d.category.map(base.pinball_core_cr),
                 MAE_skill_vs_snaive=lambda d: 1 - d.MAE_cr / d.category.map(base.MAE_cr))
    sel = st.multiselect("Models", all_models, default=[m for m in all_models if m.replace("_oosq", "") in CORE], key="bt_models")
    cols = ["category", "model", "MAPE_pct", "MAPE_excl_small_pct", "medianAPE_pct", "sMAPE_pct", "MAE_cr", "MASE", "pinball_core_cr",
            "pinball_skill_vs_snaive", "MAE_skill_vs_snaive", "cov80", "cov95"]
    st.dataframe(M[M.model.isin(sel)][cols].sort_values(["category", "pinball_core_cr"]).style.format(
        {c: "{:,.0f}" for c in ["MAPE_pct", "MAPE_excl_small_pct", "medianAPE_pct", "sMAPE_pct", "MAE_cr", "pinball_core_cr"]} |
        {"MASE": "{:.2f}", "pinball_skill_vs_snaive": "{:+.1%}", "MAE_skill_vs_snaive": "{:+.1%}", "cov80": "{:.0%}", "cov95": "{:.0%}"}),
        hide_index=True, use_container_width=True)
    st.caption(f"Backtest: 24 targets {S['test_targets'][0]}…{S['test_targets'][1]}, h=1–3, refit at every origin. Baseline = seasonal naive (snaive). "
               "MAPE is unstable when net flow is near 0; read median APE, MASE and pinball. Source: outputs/metrics.csv")
    c1, c2, c3 = st.columns(3)
    cat_b = c1.selectbox("Category", cats, key="bt_cat"); m_b = c2.selectbox("Model", all_models, index=all_models.index("sarimax_cal") if "sarimax_cal" in all_models else 0, key="bt_m")
    h_b = c3.selectbox("Horizon", [1, 2, 3], key="bt_h")
    F = fc[(fc.category == cat_b) & (fc.model == m_b) & (fc.h == h_b)].assign(date=lambda d: pd.PeriodIndex(d.target, freq="M").to_timestamp())
    band95 = alt.Chart(F).mark_area(opacity=.18, color="#264653").encode(x=alt.X("date:T", title=None), y=alt.Y("q_0.025:Q", title="₹ crore"), y2="q_0.975:Q")
    band80 = alt.Chart(F).mark_area(opacity=.3, color="#264653").encode(x="date:T", y="q_0.1:Q", y2="q_0.9:Q")
    lp = alt.Chart(F).mark_line(color="#264653").encode(x="date:T", y="point:Q")
    la = alt.Chart(F).mark_line(point=True, color="#e76f51").encode(x="date:T", y="actual:Q", tooltip=["target", alt.Tooltip("actual:Q", format=",.0f"), alt.Tooltip("point:Q", format=",.0f")])
    st.altair_chart((band95 + band80 + lp + la).properties(height=300, title=f"{cat_b} – {m_b}, h={h_b}: forecast (line, 80/95 bands) vs actual (red)"), use_container_width=True)
    st.subheader("Nested selection (choose on first 12 months, evaluate on last 12)")
    st.dataframe(D["nested_selection"][["category", "model", "select_window_pinball", "pinball_core_cr", "MAE_cr", "cov80", "cov95", "selected_on_first_half"]]
                 .sort_values(["category", "pinball_core_cr"]).style.format(precision=2, thousands=","), hide_index=True, use_container_width=True)

# ---------------- 3. coverage
with tabs[2]:
    cv = coverage_table(fc)
    msel = st.multiselect("Models", all_models, default=["snaive", "sarimax_cal", "ets_ratio", "lgbm_q", "lgbm_q_oosq"], key="cov_models")
    cv = cv[cv.model.isin(msel)]
    for lvl in (80, 95):
        x = cv[cv.nominal == lvl]
        pts = alt.Chart().mark_point(filled=True).encode(x=alt.X("model:N", title=None), y=alt.Y("coverage:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                                                           color="h:N", xOffset="h:N", tooltip=["model", "h", alt.Tooltip("coverage:Q", format=".0%"), alt.Tooltip("lo:Q", format=".0%"), alt.Tooltip("hi:Q", format=".0%")])
        err = alt.Chart().mark_rule().encode(x="model:N", y="lo:Q", y2="hi:Q", color="h:N", xOffset="h:N")
        nom = alt.Chart().mark_rule(strokeDash=[4, 4], color="black").encode(y=alt.datum(lvl / 100))
        st.altair_chart(alt.layer(err, pts, nom, data=x).properties(height=230, width=180).facet(column="category:N", title=f"{lvl}% PI empirical coverage (Wilson 95% CI, n=24 per h)"))
    st.caption("Computed from outputs/forecasts_backtest.csv. Forecasts overlap across months, so Wilson intervals are optimistic.")

# ---------------- 4. drivers
with tabs[3]:
    dr = D["drivers_monthly"].assign(date=lambda d: pd.PeriodIndex(d.month, freq="M").to_timestamp(),
                                      net_liq_lakhcr=lambda d: d.net_liq / 1e5, spread=lambda d: d.tbill_3m - d.wacr)
    c1, c2 = st.columns(2)
    rates = dr.melt("date", ["wacr", "tbill_3m"], "series", "pct")
    c1.altair_chart(alt.Chart(rates).mark_line().encode(x=alt.X("date:T", title=None), y=alt.Y("pct:Q", title="% p.a."), color="series:N").properties(height=220, title="WACR vs FBIL 3M T-bill"), use_container_width=True)
    c2.altair_chart(alt.Chart(dr.dropna(subset=["net_liq"])).mark_area(opacity=.6).encode(x=alt.X("date:T", title=None), y=alt.Y("net_liq_lakhcr:Q", title="₹ lakh crore (+ = RBI injecting)")).properties(height=220, title="Banking liquidity: avg net LAF (today's ops)"), use_container_width=True)
    c1.altair_chart(alt.Chart(dr).mark_line(color="#6a4c93").encode(x=alt.X("date:T", title=None), y=alt.Y("nifty_close:Q", title="Nifty 50", scale=alt.Scale(zero=False))).properties(height=220, title="Nifty 50 month-end"), use_container_width=True)
    c2.altair_chart(alt.Chart(dr).mark_bar().encode(x=alt.X("date:T", title=None), y=alt.Y("spread:Q", title="pp")).properties(height=220, title="T-bill − WACR spread (idle-cash opportunity cost before floor)"), use_container_width=True)
    reg = D["driver_regression"]
    spec = st.selectbox("Regression spec (in-sample, HAC SE; flow as % of prior AUM)", reg.spec.unique().tolist(), key="reg_spec")
    st.dataframe(reg[reg.spec == spec][["category", "variable", "coef_pct_of_aum", "hac_se", "p_value", "n", "r2"]].style.format(precision=3), hide_index=True, use_container_width=True)
    abl = D["metrics"][D["metrics"].model.str.match(r"^lgbm_q(_calonly|_no_\w+)?$")][["category", "model", "MAE_cr", "pinball_core_cr", "cov80"]]
    st.markdown("**Out-of-sample driver ablation (LightGBM feature groups removed)** – lower pinball = better")
    st.dataframe(abl.sort_values(["category", "pinball_core_cr"]).style.format(precision=2, thousands=","), hide_index=True, use_container_width=True)
    st.caption("Sources: outputs/drivers_monthly.csv, driver_regression.csv, metrics.csv. Advance-tax dates 15 Jun/Sep/Dec/Mar; FY-end March.")

# ---------------- 5. decision cost (recomputed live)
with tabs[4]:
    dm = st.multiselect("Models", all_models, default=[m for m in ["sarimax_cal", "ets_ratio", "snaive", "lgbm_q", "lgbm_q_oosq", "sarimax_cal_oosq"] if m in all_models], key="dec_models")
    det = recompute_costs(fc[fc.model.isin(dm)], D["actuals"], D["drivers_monthly"], tau, h_dec, **kw)
    summ = cost_summary(det)
    st.markdown(f"Policy: buffer = **τ={tau:.3f}** quantile of forecast outflow (h={h_dec}); cost = idle × s/12 + {pen:g} × shortfall × s/12, "
                f"s = {'realised FBIL 3M − WACR, floored at %.2f%%' % floor if spread_mode == 'realised' else 'constant %.2f%%' % const}. "
                "Baseline: hold last year's same-month net outflow.")
    st.dataframe(summ.style.format({c: "{:,.1f}" for c in ["policy_cost_cr", "idle_cr", "shortfall_cr", "baseline_cost_cr"]} |
                                   {"mean_buffer_cr": "{:,.0f}", "vs_baseline_pct": "{:+.1f}%"}), hide_index=True, use_container_width=True)
    curve = cost_curve(fc, D["actuals"], D["drivers_monthly"], h_dec, dm, **kw)
    base_line = curve.groupby(["category", "tau"])["baseline_cost_cr"].first().reset_index()
    cdf = pd.concat([curve[["category", "model", "tau", "policy_cost_cr"]].assign(series="policy"),
                     base_line.rename(columns={"baseline_cost_cr": "policy_cost_cr"}).assign(model="baseline (last year same month)", series="baseline")],
                    ignore_index=True)
    xs = alt.X("tau:Q", title="τ", scale=alt.Scale(domain=[0.5, 0.975]))
    lines = alt.Chart().mark_line().encode(x=xs, y=alt.Y("policy_cost_cr:Q", title="24-month cost ₹ crore"), color="model:N",
                                           strokeDash=alt.StrokeDash("series:N", legend=None)).transform_filter(alt.datum.series == "policy")
    bl = alt.Chart().mark_line(strokeDash=[5, 3], color="black").encode(x=xs, y="policy_cost_cr:Q").transform_filter(alt.datum.series == "baseline")
    now = alt.Chart().mark_rule(color="red").encode(x=alt.datum(tau))
    st.altair_chart(alt.layer(lines, bl, now, data=cdf).properties(height=260, width=420)
                    .facet(column="category:N", title="Total cost vs τ (dashed = baseline, red = current τ)"))
    cat_d = st.selectbox("Monthly detail: category", cats, key="dec_cat"); m_d = st.selectbox("Monthly detail: model", dm or all_models, key="dec_m")
    st.dataframe(det[(det.category == cat_d) & (det.model == m_d)].drop(columns=["model", "category"]).style.format(precision=1, thousands=","), hide_index=True, use_container_width=True)
    st.caption("Recomputed from outputs/forecasts_backtest.csv + actuals.csv + drivers_monthly.csv. At τ=0.75/0.95, h=2, floor 0.10, penalty 3 "
               "it reproduces outputs/decision_costs.csv (unit-tested).")

# ---------------- 6. data & refresh
with tabs[5]:
    st.json(S)
    if "data_validation" in D:
        st.markdown("**Validation**"); st.json(D["data_validation"], expanded=False)
    st.dataframe(pd.DataFrame({"file": list(D["mtimes"]), "modified_IST": [t.tz_convert("Asia/Kolkata").strftime("%Y-%m-%d %H:%M") for t in D["mtimes"].values()]}), hide_index=True)
    if (OUT / "refresh_log.json").exists():
        st.markdown("**Last refresh** (outputs/refresh_log.json)"); st.json(json.loads((OUT / "refresh_log.json").read_text()), expanded=False)
    st.markdown("**Refresh** downloads new months from official public sources (AMFI, NSE, FBIL, RBI press releases; no API keys) and reruns the pipeline (~2–4 min).")
    c1, c2 = st.columns(2)
    for col, label, args in [(c1, "Refresh data + rerun pipeline", ["refresh"]), (c2, "Rerun pipeline only (no download)", ["run"])]:
        if col.button(label, key="btn_" + args[0]):
            with st.status(f"python -m liqfc {' '.join(args)} …", expanded=True) as stt:
                p = subprocess.run([sys.executable, "-m", "liqfc", *args], cwd=ROOT, capture_output=True, text=True,
                                   env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
                st.code((p.stdout + p.stderr)[-4000:])
                stt.update(label=f"exit code {p.returncode}", state="complete" if p.returncode == 0 else "error")
            st.cache_data.clear()
