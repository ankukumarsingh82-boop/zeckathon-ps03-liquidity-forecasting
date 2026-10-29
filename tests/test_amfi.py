"""Regression tests for auditor findings (1) panel start, (2) HTML rejection, (3) magic bytes + identity, (4) release dates."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from liqfc.amfi import NotAWorkbook, detect_engine, load_amfi, parse_workbook

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/amfi"
AUDIT = Path("/workspace/zeckathon-ps03-audit/amfi_liquid_overnight_extract.csv")
CATS = ["Liquid Fund", "Overnight Fund"]


def test_html_404_rejected_and_logged(tmp_path):
    f = tmp_path / "amoct2026repo.xls"
    f.write_text("<!DOCTYPE html><html><body>404 Page not found</body></html>")
    with pytest.raises(NotAWorkbook):
        detect_engine(f)
    # a directory with one good workbook and one HTML page: HTML is logged as rejected, panel unaffected
    import shutil
    good = sorted(RAW.glob("amsep2026repo.xls"))[0]
    shutil.copy(good, tmp_path / good.name)
    p, meta = load_amfi(tmp_path, CATS, "2026-09")
    assert meta.loc[meta.file == "amoct2026repo.xls", "layout"].str.startswith("REJECTED").all()
    assert set(p["month"].astype(str)) == {"2026-09"}


def test_engine_from_magic_bytes_not_extension():
    # Apr 2019-Jun 2025 files are xlsx (ZIP) content saved as .xls; older/newer ones are OLE2
    assert detect_engine(RAW / "amapr2019repo.xls") == "openpyxl"
    assert detect_engine(RAW / "amjun2025repo.xls") == "openpyxl"
    assert detect_engine(RAW / "amsep2026repo.xls") == "xlrd"
    assert (RAW / "amapr2019repo.xls").read_bytes()[:4] == b"PK\x03\x04"


def test_pre_2019_files_yield_no_category_rows():
    rows, meta = parse_workbook(RAW / "amjun2018repo.xls", CATS)
    assert rows == [] and "pre-2019" in meta["layout"]


def test_panel_starts_2019_04_with_90_months_each(panel):
    assert str(panel["month"].min()) == "2019-04" and str(panel["month"].max()) == "2026-09"
    assert panel.groupby("category")["month"].nunique().to_dict() == {c: 90 for c in CATS}
    assert not panel.duplicated(["month", "category"]).any()


def test_flow_identity_within_tolerance(panel):
    assert panel["identity_resid"].abs().max() <= 0.1


def test_known_values_sep_2026(panel):
    s = panel[panel.month == pd.Period("2026-09", "M")].set_index("category")
    assert s.loc["Liquid Fund", "net"] == pytest.approx(-95743.20, abs=0.01)
    assert s.loc["Liquid Fund", "aum"] == pytest.approx(625847.01, abs=0.01)
    assert s.loc["Overnight Fund", "net"] == pytest.approx(-7281.20, abs=0.01)


@pytest.mark.skipif(not AUDIT.exists(), reason="auditor extract not present")
def test_matches_independent_auditor_extract(panel):
    a = pd.read_csv(AUDIT)
    a = a[a["label"].isin(CATS)].assign(month=lambda d: d["month"].map(lambda s: pd.Period(s, "M")))
    m = panel.merge(a, left_on=["month", "category"], right_on=["month", "label"], suffixes=("", "_aud"))
    assert len(m) == 180
    for c in ["net", "aum", "sales", "redemption"]:
        assert np.allclose(m[c], m[c + "_aud"], atol=0.01), c


def test_release_dates_after_month_end(panel):
    lag = (panel["release_date"] - panel["month"].dt.to_timestamp(how="end").dt.normalize()).dt.days
    assert lag.min() >= 6 and lag.max() <= 13
    imputed = panel[panel["release_date_imputed"]]
    assert ((imputed["release_date"] - imputed["month"].dt.to_timestamp(how="end").dt.normalize()).dt.days == 13).all()
