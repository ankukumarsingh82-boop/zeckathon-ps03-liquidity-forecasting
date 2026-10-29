"""v0.4 polish: one 'submitted' figure per category, stable MANIFEST, generated counts, page-1 CI note,
PDF text free of known-wrong phrases, one-page dashboard print."""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from liqfc.claims import ci_sentence, submitted_forecasts

ROOT = Path(__file__).resolve().parents[1]
OUT, REP = ROOT / "outputs", ROOT / "reports"
needs_outputs = pytest.mark.skipif(not (OUT / "run_summary.json").exists(), reason="run `python -m liqfc run` first")
needs_pdfs = pytest.mark.skipif(not (REP / "deck.pdf").exists() or not shutil.which("pdftotext"), reason="needs reports/*.pdf and pdftotext")
BAD = ["costlier than baseline for Liquid", "ranks last", "in every model", "34–45%", "34-45%", "70–206%", "311 files"]


def pdf_text(p, first=None, last=None):
    args = ["pdftotext", "-layout"] + (["-f", str(first), "-l", str(last)] if first else []) + [str(p), "-"]
    return re.sub(r"\s+", " ", subprocess.run(args, capture_output=True, text=True, check=True).stdout)


@needs_pdfs
@pytest.mark.parametrize("pdf", ["report.pdf", "deck.pdf"])
def test_known_wrong_phrases_absent_from_pdf_text(pdf):
    txt = pdf_text(REP / pdf)
    assert len(txt) > 2000                     # extraction worked
    for bad in BAD:
        assert bad not in txt, f"{pdf} contains {bad!r}"


def _submitted():
    summ = json.loads((OUT / "run_summary.json").read_text())
    sc = yaml.safe_load(open(ROOT / "config.yaml"))["submission"]
    return submitted_forecasts(pd.read_csv(OUT / "live_forecast.csv"), summ["chosen_model_by_nested_selection"], sc["rule"], sc["horizon"])


@needs_outputs
def test_submission_rule_picks_nested_selection_model_h1():
    sub = _submitted()
    summ = json.loads((OUT / "run_summary.json").read_text())
    assert set(sub) == {"Liquid Fund", "Overnight Fund"}
    for cat, r in sub.items():
        assert r["model"] == summ["chosen_model_by_nested_selection"][cat] and int(r["h"]) == 1


@needs_outputs
@pytest.mark.parametrize("source", ["report.md", "report.pdf", "deck.pdf"])
def test_exactly_one_submitted_figure_per_category(source):
    p = REP / source
    if p.suffix == ".pdf":
        if not shutil.which("pdftotext") or not p.exists():
            pytest.skip("needs pdftotext and the PDF")
        txt = pdf_text(p)
    else:
        txt = re.sub(r"\s+", " ", p.read_text())
    assert len(re.findall(r"(?i)\bsubmitted\b", txt)) == 2, "only the two category figures may be labelled submitted"
    for cat, r in _submitted().items():
        hits = re.findall(rf"Submitted (\w+ \d{{4}}) forecast, {cat}:\W*([+-][\d,]+) cr", txt)
        assert len(hits) == 1, (source, cat, hits)
        assert hits[0][0] == pd.Period(r["target"], freq="M").strftime("%B %Y")
        assert hits[0][1] == f"{r['point']:+,.0f}"


@needs_outputs
def test_page1_summary_states_liquid_ci_under_reading_a():
    boot = pd.read_csv(OUT / "decision_vs_baseline_bootstrap.csv")
    s = ci_sentence(boot, "Liquid Fund", "sarimax_cal", "q75")
    head = (REP / "report.md").read_text().split("## 1. Data")[0]
    assert s in head
    r = boot[(boot.category == "Liquid Fund") & (boot.model == "sarimax_cal") & (boot.policy == "q75") & (boot.decision_h == 2)
             & (boot.spread_basis == "fbil3m_minus_wacr") & (boot.penalty_basis == "spread")].iloc[0]
    if r.diff_ci90_lo <= 0 <= r.diff_ci90_hi:
        assert "includes 0" in s and "not statistically clear" in head
    if shutil.which("pdftotext") and (REP / "report.pdf").exists():
        assert "includes 0" in pdf_text(REP / "report.pdf", 1, 1) or "excludes 0" in pdf_text(REP / "report.pdf", 1, 1)


def test_manifest_count_is_generated():
    n = len(pd.read_csv(ROOT / "data/raw/MANIFEST.csv"))
    rep = (REP / "report.md").read_text()
    assert f"notes for {n} files" in rep
    assert "311" not in (REP / "report_template.md").read_text() and "{{fact:manifest_files}}" in (REP / "report_template.md").read_text()


def test_manifest_keeps_retrieval_times_when_mtimes_change(tmp_path):
    """Fresh clone: every file's mtime is the checkout time; MANIFEST.csv must come out byte-identical."""
    (tmp_path / "scripts").mkdir()
    shutil.copy(ROOT / "scripts/build_manifest.py", tmp_path / "scripts")
    shutil.copytree(ROOT / "data/raw", tmp_path / "data/raw")      # copytree without copy2 metadata -> new mtimes
    for p in (tmp_path / "data/raw").rglob("*"):
        p.touch()
    before = (tmp_path / "data/raw/MANIFEST.csv").read_bytes()
    subprocess.run([sys.executable, str(tmp_path / "scripts/build_manifest.py")], check=True, capture_output=True)
    assert (tmp_path / "data/raw/MANIFEST.csv").read_bytes() == before
    # a changed file gets a fresh time and hash
    f = sorted((tmp_path / "data/raw/nse").glob("ind_close_all_*.csv"))[0]
    f.write_bytes(f.read_bytes() + b"\n")
    subprocess.run([sys.executable, str(tmp_path / "scripts/build_manifest.py")], check=True, capture_output=True)
    m0 = pd.read_csv(pd.io.common.BytesIO(before)).set_index("file")
    m1 = pd.read_csv(tmp_path / "data/raw/MANIFEST.csv").set_index("file")
    k = str(f.relative_to(tmp_path))
    assert m1.loc[k, "sha256"] != m0.loc[k, "sha256"]
    assert (m1.drop(index=k).retrieved_utc == m0.drop(index=k).retrieved_utc).all()


def test_dashboard_prints_on_one_page():
    weasyprint = pytest.importorskip("weasyprint")
    doc = weasyprint.HTML(filename=str(REP / "dashboard.html")).render()
    assert len(doc.pages) == 1
