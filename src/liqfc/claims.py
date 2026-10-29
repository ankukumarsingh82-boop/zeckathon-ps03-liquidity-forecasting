"""Data-driven text and tables shared by the static dashboard, the Streamlit app and the report.

Every sentence here is computed from outputs/*.csv, so the prose cannot drift from the numbers
(audit finding F4/F8: a hard-coded label describing the Liquid result was shown on the Overnight card).
"""
from __future__ import annotations

import pandas as pd

PRIMARY_BASIS = "spread"
BASIS_LABEL = {"spread": "(a) 3x spread on the shortfall [primary]", "amount": "(b) 3x the shortfall amount (₹)"}
BASIS_SHORT = {"spread": "reading (a): 3 × shortfall × spread/12", "amount": "reading (b): 3 × shortfall ₹"}
REAL = "fbil3m_minus_wacr"


def _sel(df: pd.DataFrame, h: int = 2, basis: str = PRIMARY_BASIS, spread: str = REAL) -> pd.DataFrame:
    if "penalty_basis" in df.columns:
        df = df[df.penalty_basis == basis]
    return df[(df.decision_h == h) & (df.spread_basis == spread)]


def vs_baseline_label(boot: pd.DataFrame, category: str, model: str, policy: str, h: int = 2, basis: str = PRIMARY_BASIS) -> str:
    """e.g. 'backtest: 22.9 ₹cr cheaper than baseline over 24 m (90% CI excludes 0)' for THIS category and model."""
    r = _sel(boot, h, basis)
    r = r[(r.category == category) & (r.model == model) & (r.policy == policy)]
    if r.empty:
        return "backtest: no comparison available"
    r = r.iloc[0]
    d, lo, hi = r.total_diff_cr, r.diff_ci90_lo, r.diff_ci90_hi
    word = "cheaper" if d < 0 else "costlier"
    sig = "90% CI excludes 0" if (lo > 0 or hi < 0) else "90% CI includes 0"
    return f"backtest {policy}, penalty reading {'(a)' if basis == 'spread' else '(b)'}: ₹{abs(d):,.1f} cr {word} than baseline over {int(r.n_months)} m ({sig})"


def penalty_table(dsum: pd.DataFrame, basis: str, models: list[str], h: int = 2) -> pd.DataFrame:
    d = _sel(dsum, h, basis)
    keep = d[d.model.isin(models) & d.policy.isin(["q75", "q95"]) | (d.policy == "smly_baseline")]
    cols = ["category", "model", "policy", "total_cost_cr", "idle_cost_cr", "shortfall_cost_cr", "breaches", "mean_buffer_cr", "cost_vs_baseline_pct"]
    order = {m: i for i, m in enumerate(["baseline"] + models)}
    return keep[cols].assign(_o=keep.model.map(order)).sort_values(["category", "_o", "policy"]).drop(columns="_o").reset_index(drop=True)


def q75_vs_q95(dsum: pd.DataFrame, models: list[str], h: int = 2) -> pd.DataFrame:
    """Per category x model x reading: which of q75 / q95 is cheaper, and whether the winner flips between readings."""
    rows = []
    for basis in ("spread", "amount"):
        d = _sel(dsum, h, basis)
        for (cat, m), g in d[d.model.isin(models)].groupby(["category", "model"]):
            c = g.set_index("policy").total_cost_cr
            rows.append({"category": cat, "model": m, "basis": basis, "q75": c["q75"], "q95": c["q95"],
                         "cheaper": "q75" if c["q75"] < c["q95"] else "q95"})
    t = pd.DataFrame(rows).pivot_table(index=["category", "model"], columns="basis", values=["q75", "q95", "cheaper"], aggfunc="first")
    t.columns = [f"{a}_{b}" for a, b in t.columns]
    t = t.reset_index()
    t["flips"] = t["cheaper_spread"] != t["cheaper_amount"]
    return t


def flip_sentence(dsum: pd.DataFrame, models: list[str], h: int = 2) -> str:
    t = q75_vs_q95(dsum, models, h)
    n, nf = len(t), int(t.flips.sum())
    a = t.cheaper_spread.value_counts().to_dict(); b = t.cheaper_amount.value_counts().to_dict()
    def counts(col):
        return f"q75 is cheaper than q95 in {a.get('q75', 0) if col == 'a' else b.get('q75', 0)} of {n} category × model cases and q95 in {a.get('q95', 0) if col == 'a' else b.get('q95', 0)}"
    if nf == 0:
        return f"**No, the conclusion does not flip.** Under reading (a) {counts('a')}; under reading (b) {counts('b')}."
    return f"**Yes, the conclusion flips.** Under reading (a) {counts('a')}; under reading (b) {counts('b')}. The cheaper policy changes in {nf} of {n} cases."


def report_facts(met: pd.DataFrame, nested: pd.DataFrame, dsum: pd.DataFrame, boot: pd.DataFrame, models: list[str]) -> dict:
    """Numbers for the sentences in reports/report.md that the audit found wrong (F8)."""
    f = {}
    ev = nested.sort_values(["category", "pinball_core_cr"]).copy()
    ev["rank"] = ev.groupby("category").pinball_core_cr.rank(method="min").astype(int)
    for cat, g in ev.groupby("category"):
        sel = g[g.selected_on_first_half].iloc[0]
        key = cat.split()[0].lower()
        f[f"{key}_sel_model"] = sel.model
        f[f"{key}_sel_rank"] = f"{sel['rank']} of {len(g)}"
        f[f"{key}_sel_pinball"] = f"{sel.pinball_core_cr:,.0f}"
        best = g.iloc[0]
        f[f"{key}_eval_best"] = f"`{best.model}` ({best.pinball_core_cr:,.0f})"
    d = _sel(dsum, 2, PRIMARY_BASIS)
    d = d[d.model.isin(models)]
    for cat, g in d.groupby("category"):
        key = cat.split()[0].lower()
        for pol in ("q75", "q95"):
            p = g[g.policy == pol].cost_vs_baseline_pct
            f[f"{key}_{pol}_range"] = f"{p.min():+.0f}% to {p.max():+.0f}%"
            f[f"{key}_{pol}_n_cheaper"] = f"{int((p < 0).sum())} of {len(p)}"
    b = _sel(boot, 2, PRIMARY_BASIS)
    b = b[b.model.isin(models)]
    for cat, g in b.groupby("category"):
        key = cat.split()[0].lower()
        for pol in ("q75", "q95"):
            gp = g[g.policy == pol]
            f[f"{key}_{pol}_n_ci_excl0_costlier"] = f"{int((gp.diff_ci90_lo > 0).sum())} of {len(gp)}"
            f[f"{key}_{pol}_n_ci_excl0_cheaper"] = f"{int((gp.diff_ci90_hi < 0).sum())} of {len(gp)}"
    return f
