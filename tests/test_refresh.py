"""Offline tests of the refresh logic (fake HTTP session; no network)."""
import datetime as dt
import shutil
from pathlib import Path

from liqfc.refresh import fetch_amfi, fetch_rbi_mmo

ROOT = Path(__file__).resolve().parents[1]


class Resp:
    def __init__(self, code, body):
        self.status_code, self.content = code, body
        self.text = body.decode("utf-8", "ignore")


class FakeSession:
    def __init__(self, table):
        self.table, self.calls = table, []

    def get(self, url, headers=None, timeout=None):
        self.calls.append(url)
        for k, v in self.table.items():
            if k in url:
                return v
        return Resp(404, b"<html>404</html>")


def test_amfi_refresh_saves_only_real_workbooks(tmp_path):
    (tmp_path / "amfi").mkdir()
    shutil.copy(ROOT / "data/raw/amfi/amaug2026repo.xls", tmp_path / "amfi")
    good = (ROOT / "data/raw/amfi/amsep2026repo.xls").read_bytes()
    s = FakeSession({"amsep2026": Resp(200, good), "amoct2026": Resp(200, b"<!DOCTYPE html><html>not found</html>")})
    res = fetch_amfi(tmp_path, "https://x/am{mon}{yyyy}repo.xls", dt.date(2026, 11, 20), session=s)
    assert [r["file"] for r in res] == ["amsep2026repo.xls", "amoct2026repo.xls"]
    assert (tmp_path / "amfi/amsep2026repo.xls").exists() and not (tmp_path / "amfi/amoct2026repo.xls").exists()
    assert "REJECTED" in (tmp_path / "amfi_dl_log.txt").read_text()


def test_rbi_scan_keeps_only_mmo_releases(tmp_path):
    d = tmp_path / "rbi_mmo"; d.mkdir()
    (d / "index.txt").write_text("63754|Money Market Operations as on October 07, 2026|t\n")
    s = FakeSession({"prid=63756": Resp(200, b"<p>Money Market Operations as on October 08, 2026</p>"),
                     "prid=63755": Resp(200, b"<p>Some other press release</p>")})
    res = fetch_rbi_mmo(tmp_path, "https://x?prid={prid}", scan=3, session=s, workers=1)
    assert (d / "prid_63756.html").exists() and not (d / "prid_63755.html").exists()
    assert "63756|Money Market Operations as on October 08, 2026" in (d / "index.txt").read_text()
    assert res[-1]["status"].startswith("scanned, 1")
