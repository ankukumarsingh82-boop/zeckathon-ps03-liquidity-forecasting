"""AMFI monthly report loader + validation.

Source: https://portal.amfiindia.com/spages/am{mon}{yyyy}repo.xls (AMFI monthly category report).
Units: INR crore. Rows located by label, columns by header text; engine chosen from magic bytes
(AMFI serves both OLE2 .xls and xlsx/ZIP content behind the same .xls extension).
"""
from __future__ import annotations

import calendar
import datetime as dt
import hashlib
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
OLE2_MAGIC = b"\xd0\xcf\x11\xe0"
ZIP_MAGIC = b"PK\x03\x04"


class NotAWorkbook(ValueError):
    pass


def detect_engine(path: Path) -> str:
    head = Path(path).read_bytes()[:8]
    if head.startswith(OLE2_MAGIC):
        return "xlrd"
    if head.startswith(ZIP_MAGIC):
        return "openpyxl"
    raise NotAWorkbook(f"{path}: not an Excel workbook (magic={head!r})")


def month_from_filename(path: Path) -> pd.Period:
    m = re.match(r"am([a-z]{3})(\d{4})repo\.xls$", Path(path).name)
    if not m:
        raise ValueError(f"unexpected filename {path}")
    return pd.Period(f"{m.group(2)}-{MONTHS.index(m.group(1)) + 1:02d}", "M")


def _num(v) -> float:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if isinstance(v, (int, float, np.number)):
        return float(v)
    s = str(v).replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def _find_cols(header_text: list[str]) -> dict[str, int]:
    out = {}
    for j, h in enumerate(header_text):
        h = h.lower()
        if "net inflow" in h and "net" not in out:
            out["net"] = j
        elif ("mobili" in h) and "sales" not in out:
            out["sales"] = j
        elif "redemption" in h and "redemption" not in out:
            out["redemption"] = j
        elif "average" in h and "aaum" not in out and "segregated" not in h:
            out["aaum"] = j
        elif ("assets under management as on" in h or h.startswith("aum as on")) and "aum" not in out and "segregated" not in h:
            out["aum"] = j
        elif "no. of schemes" in h and "n_schemes" not in out:
            out["n_schemes"] = j
        elif "folios" in h and "folios" not in out:
            out["folios"] = j
    return out


def parse_workbook(path: Path, categories: list[str]) -> tuple[list[dict], dict]:
    """Return (rows, meta). Raises NotAWorkbook for HTML/404 payloads."""
    engine = detect_engine(path)
    month = month_from_filename(path)
    x = pd.read_excel(path, header=None, engine=engine, sheet_name=0)
    text = x.astype(str).where(x.notna(), "")
    meta = {"file": Path(path).name, "month": str(month), "engine": engine,
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "release_date": None, "units_text": None, "layout": None}
    flat = " ".join(" ".join(r) for r in text.values.tolist())
    m = re.search(r"Released on\s*:?\s*(\d{1,2}[-./][A-Za-z0-9]{2,3}[-./]\d{4})", flat)
    if m:
        s = m.group(1)
        for fmt in ("%d-%b-%Y", "%d.%m.%Y", "%d-%m-%Y", "%d/%m/%Y"):
            try:
                meta["release_date"] = str(dt.datetime.strptime(s, fmt).date())
                break
            except ValueError:
                pass
    u = re.search(r"(` in Crore|₹ in Crore|Rs\.? in Crore|INR in crore)", flat, re.I)
    meta["units_text"] = u.group(1) if u else None
    # header row = first row containing "Net Inflow"
    hdr = [i for i in range(len(text)) if any("net inflow" in c.lower() for c in text.iloc[i])]
    if not hdr:
        meta["layout"] = "pre-2019 sponsor-wise (no Liquid/Overnight category rows)"
        return [], meta
    meta["layout"] = "scheme-category"
    h = hdr[0]
    cols = _find_cols(list(text.iloc[h]))
    needed = {"sales", "redemption", "net", "aum"}
    if not needed <= cols.keys():
        raise ValueError(f"{path}: missing columns {needed - cols.keys()} in header {list(text.iloc[h])}")
    if not any("crore" in c.lower() for c in text.iloc[h]) and not meta["units_text"]:
        # some releases (e.g. Apr 2020) carry no unit text; units are then verified by AUM continuity in validate_panel
        meta["units_text"] = "MISSING (verified by AUM continuity)"
        log.warning("%s: no unit text; relying on AUM-continuity unit check", path)
    rows = []
    for cat in categories:
        hits = [i for i in range(h + 1, len(text)) if any(c.strip().lower() == cat.lower() for c in text.iloc[i])]
        if len(hits) != 1:
            raise ValueError(f"{path}: expected exactly one row for {cat}, found {hits}")
        r = x.iloc[hits[0]]
        rec = {"month": month, "category": cat}
        for k, j in cols.items():
            rec[k] = _num(r.iloc[j])
        rows.append(rec)
    return rows, meta


def load_amfi(raw_dir: str | Path, categories: list[str], panel_start: str, release_lag_default_days: int = 13,
              identity_tol: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all workbooks -> (panel, file_meta). panel indexed by (month, category)."""
    raw_dir = Path(raw_dir)
    rows, metas = [], []
    for p in sorted(raw_dir.glob("am*repo.xls")):
        try:
            r, meta = parse_workbook(p, categories)
        except NotAWorkbook as e:
            log.warning("rejected non-workbook: %s", e)
            metas.append({"file": p.name, "month": str(month_from_filename(p)), "engine": None,
                          "layout": "REJECTED: not an Excel workbook", "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
            continue
        metas.append(meta)
        rows.extend(r)
    meta = pd.DataFrame(metas).sort_values("month").reset_index(drop=True)
    df = pd.DataFrame(rows)
    df = df[df["month"] >= pd.Period(panel_start, "M")].copy()
    rel = {k: v for k, v in meta.set_index("month")["release_date"].to_dict().items() if isinstance(v, str) and v}

    def release(m: pd.Period) -> pd.Timestamp:
        s = rel.get(str(m))
        if s:
            return pd.Timestamp(s)
        return m.to_timestamp(how="end").normalize() + pd.Timedelta(days=release_lag_default_days)

    df["release_date"] = df["month"].map(release)
    df["release_date_imputed"] = df["month"].map(lambda m: not bool(rel.get(str(m))))
    df["identity_resid"] = df["sales"] - df["redemption"] - df["net"]
    df = df.sort_values(["category", "month"]).reset_index(drop=True)
    validate_panel(df, categories, panel_start, identity_tol)
    return df, meta


def validate_panel(df: pd.DataFrame, categories: list[str], panel_start: str, identity_tol: float = 0.1) -> dict:
    """Hard checks; raises AssertionError on failure. Returns a summary dict."""
    issues = {}
    dups = df.duplicated(["month", "category"]).sum()
    assert dups == 0, f"{dups} duplicate month/category rows"
    for cat in categories:
        s = df[df.category == cat].set_index("month").sort_index()
        full = pd.period_range(panel_start, s.index.max(), freq="M")
        missing = full.difference(s.index)
        assert len(missing) == 0, f"{cat}: missing months {list(missing)}"
        assert s["net"].notna().all() and s["aum"].notna().all(), f"{cat}: NaN values"
        assert (s["sales"] >= 0).all() and (s["redemption"] >= 0).all(), f"{cat}: negative gross flows (sign error)"
        assert (s["aum"] > 0).all(), f"{cat}: non-positive AUM"
        bad = s[np.abs(s["identity_resid"]) > identity_tol]
        assert bad.empty, f"{cat}: sales-redemption != net for {list(bad.index)}"
        # unit sanity: liquid AUM must be in lakh-crore range when expressed in crore
        assert s["aum"].median() > 1000, f"{cat}: AUM median too small - unit error?"
        # unit consistency month to month: a crore/lakh mix-up would show as a 100x jump in AUM
        jump = (s["aum"] / s["aum"].shift(1)).dropna()
        assert ((jump > 0.33) & (jump < 3.0)).all(), f"{cat}: AUM jumps suggest unit change {list(jump[(jump <= 0.33) | (jump >= 3)].index)}"
        # outlier flags (not removed): robust z of net/AUM_lag
        ratio = s["net"] / s["aum"].shift(1)
        z = (ratio - ratio.median()) / (1.4826 * (ratio - ratio.median()).abs().median())
        issues[cat] = {"n_months": len(s), "outlier_months_|z|>4": [str(m) for m in z[np.abs(z) > 4].index]}
    assert not set(df.month.astype(str)) - set(str(m) for m in pd.period_range(panel_start, df.month.max(), freq="M"))
    return issues
