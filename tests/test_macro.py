"""Driver-data tests: official RBI press-release parser and the third-party DBIE mirror cross-check."""
from pathlib import Path

import pandas as pd
import pytest

from liqfc.macro import load_mmo_mirror, parse_mmo_press_release

ROOT = Path(__file__).resolve().parents[1]


def _read(p):
    return Path(p).read_text(encoding="utf-8", errors="ignore")


def test_press_release_parser_handles_nbsp():
    r = parse_mmo_press_release(_read(ROOT / "data/raw/rbi_mmo/prid_63629.html"))
    assert r["date"] == pd.Timestamp("2026-09-18")
    assert r["net_liq_today"] == pytest.approx(-331731.0)
    assert r["call_wacr"] == pytest.approx(4.94)


def test_mirror_matches_official_release_sep15():
    off = parse_mmo_press_release(_read(ROOT / "data/raw/rbi_mmo_crosscheck/prid_63598.html"))
    mir = load_mmo_mirror().set_index("date").loc[pd.Timestamp("2026-09-15")]
    assert mir["net_liq_today"] == pytest.approx(off["net_liq_today"]) == pytest.approx(-671097.0)
    assert mir["call_wacr"] == pytest.approx(off["call_wacr"])


def test_monthly_drivers_shape(drivers):
    assert str(drivers.index.min()) <= "2019-04" and str(drivers.index.max()) >= "2026-09"
    assert drivers.loc[pd.Period("2026-09", "M"), "wacr"] == pytest.approx(4.97, abs=0.05)
    assert drivers.loc["2020-07":"2026-09", "net_liq"].notna().all()
