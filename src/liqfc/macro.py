"""Driver data: money-market rates, banking-system liquidity, T-bill yields, Nifty 50.

Sources (see data/raw/MANIFEST.csv):
* RBI DBIE "Daily Money Market Operations" (report 659) and "Daily Weighted Average Call/Notice Money Rates"
  (Handbook report 198), retrieved via the dbie.rbihub.in read-only data API (a scrape of RBI DBIE, NOT an RBI
  property; values spot-checked against official RBI press releases, see tests/test_macro.py).
* RBI official "Money Market Operations as on <date>" press releases (rbi.org.in) for days after the mirror's last date.
* FBIL T-Bill benchmark (3-month tenor), fbil.org.in public archive.
* NSE official daily index close files (nsearchives.nseindia.com), month-end Nifty 50 close.
All amounts INR crore; all rates % p.a.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"


def _f(x) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).replace(",", "").strip()
    if s in {"", "-", "–", "nan"}:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def load_mmo_mirror(raw: Path = RAW) -> pd.DataFrame:
    """Daily RBI MMO (DBIE report 659): call WACR, LAF today's-operation amounts -> net liquidity (inj +)."""
    a = pd.read_csv(raw / "rbi_dbie" / "r659_daily_money_market_operations.csv", dtype=str)
    a = a[a["c1"].fillna("").str.match(r"^[A-Z][a-z]{2} \d{2}, \d{4}$")].copy()
    out = pd.DataFrame({"date": pd.to_datetime(a["c1"], format="%b %d, %Y")})
    out["call_wacr"] = a["c3"].map(_f).values
    amt = {k: a[c].map(_f).fillna(0.0).values for k, c in
           dict(repo_fix="c39", rrepo_fix="c41", msf="c43", sdf="c45", repo_var="c47", rrepo_var="c49").items()}
    out["net_liq_today"] = amt["repo_fix"] + amt["repo_var"] + amt["msf"] - amt["rrepo_fix"] - amt["rrepo_var"] - amt["sdf"]
    out["msf_rate"] = a["c44"].map(_f).values
    out["sdf_rate"] = a["c46"].map(_f).values
    out["source"] = "dbie_mirror_r659"
    return out.sort_values("date").reset_index(drop=True)


def parse_mmo_press_release(html: str) -> dict:
    """Parse an official RBI 'Money Market Operations as on <date>' press release."""
    from bs4 import BeautifulSoup
    t = BeautifulSoup(html, "lxml").get_text(" ", strip=True).replace("\xa0", " ")
    d = re.search(r"Money Market Operations as on ([A-Za-z]+ \d{1,2}, \d{4})", t).group(1)
    call = re.search(r"I\. Call Money\s+([\d,\.]+)\s+([\d\.]+)", t)
    net = re.search(r"5\. Net liquidity injected from today.s operations \[injection \(\+\)/absorption \(-\)\]\*?\s+(-?[\d,\.]+)", t)
    msf = re.search(r"3\. MSF\s*#?\s+\w{3}, \d{2}/\d{2}/\d{4}\s+\d+\s+\w{3}, \d{2}/\d{2}/\d{4}\s+[\d,\.]+\s+([\d\.]+)", t)
    sdf = re.search(r"4\. SDF\s*\S*\s+\w{3}, \d{2}/\d{2}/\d{4}\s+\d+\s+\w{3}, \d{2}/\d{2}/\d{4}\s+[\d,\.]+\s+([\d\.]+)", t)
    return {"date": pd.Timestamp(pd.to_datetime(d, format="%B %d, %Y")),
            "call_wacr": _f(call.group(2)) if call else np.nan,
            "net_liq_today": _f(net.group(1)) if net else np.nan,
            "msf_rate": _f(msf.group(1)) if msf else np.nan,
            "sdf_rate": _f(sdf.group(1)) if sdf else np.nan,
            "source": "rbi_press_release"}


def load_mmo_official(raw: Path = RAW) -> pd.DataFrame:
    rows = [parse_mmo_press_release(p.read_text(errors="ignore")) for p in sorted((raw / "rbi_mmo").glob("prid_*.html"))]
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True) if rows else pd.DataFrame()


def load_daily_call_r198(raw: Path = RAW) -> pd.DataFrame:
    b = pd.read_csv(raw / "rbi_dbie" / "r198_daily_weighted_average_call_notice_money_rates.csv", dtype=str)
    b = b[b["c1"].fillna("").str.match(r"^\d{2}-[A-Za-z]{3}-\d{2}$")].copy()
    return pd.DataFrame({"date": pd.to_datetime(b["c1"], format="%d-%b-%y"), "call_wacr_r198": b["c4"].map(_f).values}) \
        .sort_values("date").drop_duplicates("date").reset_index(drop=True)


def load_daily_mmo(raw: Path = RAW) -> pd.DataFrame:
    mir = load_mmo_mirror(raw)
    off = load_mmo_official(raw)
    if not off.empty:
        off = off[off["date"] > mir["date"].max()]
    d = pd.concat([mir, off], ignore_index=True).sort_values("date")
    assert not d["date"].duplicated().any()
    # trading days only for rates (weekend rows carry only SDF/MSF)
    return d.reset_index(drop=True)


def load_fbil_tbill(raw: Path = RAW, tenor: str = "3 Months") -> pd.DataFrame:
    rows = []
    for p in sorted((raw / "fbil").glob("tbill_*.json")):
        for r in json.loads(p.read_text()):
            if r.get("tenorName") == tenor:
                rows.append({"date": pd.Timestamp(r["processRunDate"][:10]), "tbill_3m": float(r["rate"])})
    return pd.DataFrame(rows).drop_duplicates("date").sort_values("date").reset_index(drop=True)


def load_nifty_month_end(raw: Path = RAW) -> pd.DataFrame:
    rows = []
    for p in sorted((raw / "nse").glob("ind_close_all_*.csv")):
        x = pd.read_csv(p)
        r = x[x["Index Name"].str.strip().str.lower() == "nifty 50"].iloc[0]
        rows.append({"date": pd.to_datetime(r["Index Date"], format="%d-%m-%Y"), "nifty_close": float(r["Closing Index Value"])})
    d = pd.DataFrame(rows).sort_values("date")
    d["month"] = d["date"].dt.to_period("M")
    assert not d["month"].duplicated().any()
    return d.reset_index(drop=True)


def build_monthly_drivers(raw: Path = RAW) -> pd.DataFrame:
    """Monthly driver table indexed by month with an `available_date` (when month-M values are knowable)."""
    call = load_daily_call_r198(raw)
    mmo = load_daily_mmo(raw)
    tb = load_fbil_tbill(raw)
    nf = load_nifty_month_end(raw)
    # WACR: r198 (full history) then official press releases for later days
    c = call.rename(columns={"call_wacr_r198": "call_wacr"})
    extra = mmo[(mmo["date"] > c["date"].max()) & mmo["call_wacr"].notna()][["date", "call_wacr"]]
    c = pd.concat([c, extra]).dropna()
    m = pd.DataFrame(index=pd.period_range("2018-01", "2026-12", freq="M"))
    m["wacr"] = c.groupby(c["date"].dt.to_period("M"))["call_wacr"].mean()
    m["wacr_days"] = c.groupby(c["date"].dt.to_period("M"))["call_wacr"].count()
    liq = mmo.dropna(subset=["net_liq_today"])
    m["net_liq"] = liq.groupby(liq["date"].dt.to_period("M"))["net_liq_today"].mean()       # INR crore/day, inj(+)
    m["net_liq_days"] = liq.groupby(liq["date"].dt.to_period("M"))["net_liq_today"].count()
    corr = mmo.dropna(subset=["msf_rate"])
    m["msf_rate_eom"] = corr.groupby(corr["date"].dt.to_period("M"))["msf_rate"].last()
    m["tbill_3m"] = tb.groupby(tb["date"].dt.to_period("M"))["tbill_3m"].mean()
    m["nifty_close"] = nf.set_index("month")["nifty_close"]
    m["available_date"] = (m.index.to_timestamp(how="end").normalize() + pd.Timedelta(days=3))
    m.index.name = "month"
    return m


def dbie_monthly_wacr(raw: Path = RAW) -> pd.Series:
    d = pd.read_csv(raw / "rbi_dbie" / "war_call_money_rn.csv")
    d = d[d["hi_lw_avg_rates_rn"] == "AVERAGE_RATE"]
    s = d.set_index(pd.PeriodIndex(pd.to_datetime(d["time_period"]), freq="M"))["obs_value"]
    return s[~s.index.duplicated()]


def dbie_monthly_tbill_sgl(raw: Path = RAW, bucket: str = "15TO91") -> pd.Series:
    d = pd.read_csv(raw / "rbi_dbie" / "yield_tb_rn.csv")
    d = d[d["mat_term_d_rn"] == bucket]
    s = d.set_index(pd.PeriodIndex(pd.to_datetime(d["time_period"]), freq="M"))["obs_value"]
    return s[~s.index.duplicated()]
