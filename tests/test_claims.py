"""Penalty readings, the q75-vs-q95 flip, and data-generated text (audit F4/F8)."""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from liqfc.claims import flip_sentence, q75_vs_q95, report_facts, vs_baseline_label

ROOT = Path(__file__).resolve().parents[1]
OUT, REP = ROOT / "outputs", ROOT / "reports"
needs_outputs = pytest.mark.skipif(not (OUT / "run_summary.json").exists(), reason="run `python -m liqfc run` first")

# phrases the audit found wrong; they must never come back
BAD = ["costlier than baseline for Liquid", "ranks last", "in every model", "34–45%", "34-45%", "70–206%"]


def _dsum(rows):
    return pd.DataFrame([{"category": c, "model": m, "policy": p, "penalty_basis": b, "total_cost_cr": v,
                          "decision_h": 2, "spread_basis": "fbil3m_minus_wacr"} for c, m, p, b, v in rows])


def test_q75_vs_q95_and_flip_sentence_synthetic():
    d = _dsum([("L", "m1", "q75", "spread", 1), ("L", "m1", "q95", "spread", 2), ("L", "m1", "q75", "amount", 9), ("L", "m1", "q95", "amount", 3),
               ("L", "m2", "q75", "spread", 5), ("L", "m2", "q95", "spread", 4), ("L", "m2", "q75", "amount", 9), ("L", "m2", "q95", "amount", 3)])
    t = q75_vs_q95(d, ["m1", "m2"]).set_index("model")
    assert t.loc["m1", "cheaper_spread"] == "q75" and t.loc["m1", "cheaper_amount"] == "q95" and t.loc["m1", "flips"]
    assert t.loc["m2", "cheaper_spread"] == "q95" and not t.loc["m2", "flips"]
    s = flip_sentence(d, ["m1", "m2"])
    assert s.startswith("**Yes, the conclusion flips.**") and "changes in 1 of 2" in s
    same = d.assign(total_cost_cr=np.where(d.policy == "q75", 1, 2))
    assert flip_sentence(same, ["m1", "m2"]).startswith("**No, the conclusion does not flip.**")


def test_vs_baseline_label_word_follows_sign():
    boot = pd.DataFrame([{"category": "Overnight Fund", "model": "m", "policy": "q95", "decision_h": 2, "spread_basis": "fbil3m_minus_wacr",
                          "penalty_basis": "spread", "total_diff_cr": -22.9, "diff_ci90_lo": -40, "diff_ci90_hi": -5, "n_months": 24}])
    lab = vs_baseline_label(boot, "Overnight Fund", "m", "q95")
    assert "cheaper" in lab and "costlier" not in lab and "Liquid" not in lab and "excludes 0" in lab
    boot2 = boot.assign(total_diff_cr=47.2, diff_ci90_lo=-1, diff_ci90_hi=90)
    assert "costlier" in vs_baseline_label(boot2, "Overnight Fund", "m", "q95") and "includes 0" in vs_baseline_label(boot2, "Overnight Fund", "m", "q95")


@pytest.mark.parametrize("path", ["reports/dashboard.html", "src/liqfc/dashboard.py", "reports/report.md",
                                  "reports/report_template.md", "app/streamlit_app.py"])
def test_known_wrong_phrases_are_gone(path):
    p = ROOT / path
    if not p.exists():
        pytest.skip(f"{path} not built")
    txt = p.read_text()
    for bad in BAD:
        assert bad not in txt, f"{path} still contains {bad!r}"


@needs_outputs
def test_dashboard_card_labels_match_bootstrap_sign():
    import json
    sel = json.loads((OUT / "run_summary.json").read_text())["chosen_model_by_nested_selection"]
    boot = pd.read_csv(OUT / "decision_vs_baseline_bootstrap.csv")
    html = (REP / "dashboard.html").read_text()
    for cat, model in sel.items():
        r = boot[(boot.category == cat) & (boot.model == model) & (boot.policy == "q95") & (boot.decision_h == 2)
                 & (boot.spread_basis == "fbil3m_minus_wacr") & (boot.penalty_basis == "spread")].iloc[0]
        word = "cheaper" if r.total_diff_cr < 0 else "costlier"
        expect = f"₹{abs(r.total_diff_cr):,.1f} cr {word} than baseline"
        assert expect in html, (cat, expect)


@needs_outputs
def test_report_is_rendered_from_template_and_data():
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_report
    fresh = build_report.render((REP / "report_template.md").read_text(), build_report.facts(), pd.read_csv(OUT / "decision_costs.csv"))
    rep = (REP / "report.md").read_text()
    assert rep.split("\n", 1)[1] == fresh, "reports/report.md is stale: run `python -m liqfc docs`"
    assert "{{" not in rep
    f = build_report.facts()
    assert f"ranks {f['overnight_sel_rank']} on the evaluation window" in rep
    assert f["flip_sentence"] in rep
    with pytest.raises(KeyError):
        build_report.render("{{fact:no_such_key}}", f, pd.read_csv(OUT / "decision_costs.csv"))


@needs_outputs
def test_decision_costs_have_both_readings_and_report_claims_hold():
    d = pd.read_csv(OUT / "decision_costs.csv")
    assert set(d.penalty_basis) == {"spread", "amount"}
    d = d[(d.decision_h == 2) & (d.spread_basis == "fbil3m_minus_wacr")]
    a, b = (d[d.penalty_basis == x].set_index(["category", "model", "policy"]).sort_index() for x in ("spread", "amount"))
    assert a.index.equals(b.index)
    # idle cost and breaches do not depend on the reading; shortfall in (b) = 3 x shortfall amount
    assert np.allclose(a.idle_cost_cr, b.idle_cost_cr) and (a.breaches == b.breaches).all()
    assert np.allclose(a.mean_buffer_cr, b.mean_buffer_cr)
    det = pd.read_csv(OUT / "decision_costs_detail.csv")
    det = det[(det.decision_h == 2) & (det.spread_basis == "fbil3m_minus_wacr") & (det.penalty_basis == "amount")]
    assert np.allclose(det.shortfall_cost, 3 * det.shortfall_amt)
    # sentence in the report: sarimax_cal q75 beats the baseline under both readings in both categories
    for cat in ("Liquid Fund", "Overnight Fund"):
        for x in (a, b):
            assert x.loc[(cat, "sarimax_cal", "q75"), "cost_vs_baseline_pct"] < 0


@pytest.mark.skipif(not (REP / "deck.pdf").exists(), reason="run `python -m liqfc docs` first")
def test_pdfs_exist_with_expected_length():
    def pages(p):
        try:
            out = subprocess.run(["pdfinfo", str(p)], capture_output=True, text=True, check=True).stdout
        except FileNotFoundError:
            pytest.skip("pdfinfo not installed")
        return int(re.search(r"Pages:\s+(\d+)", out).group(1))
    assert pages(REP / "deck.pdf") == 8
    assert 1 <= pages(REP / "report.pdf") <= 8
