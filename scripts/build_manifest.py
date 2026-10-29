"""Write data/raw/MANIFEST.csv: one row per raw file with source, URL, retrieval time (UTC + IST),
units, period, sha256 and status notes.

Retrieval time is recorded once. If a file already has a MANIFEST row with the same sha256, its recorded time is kept;
only new or changed files get their file mtime. So `liqfc run` in a fresh clone (where every mtime is the checkout
time) leaves MANIFEST.csv byte-identical."""
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
MON = "jan feb mar apr may jun jul aug sep oct nov dec".split()

UNUSABLE = {"r151_major_monetary_policy_rates_and_reserve_requirements_bank.csv": "UNUSED: value columns empty in mirror",
            "r113_91d_tbill_auctions.csv": "UNUSED: decimal points corrupted in mirror",
            "avg_nfty_rn.csv": "UNUSED: duplicated 2026-06 row, missing 2026-07; NSE official files used instead",
            "com_paper_rn.csv": "cross-check only (ends 2026-04)", "war_call_money_rn.csv": "cross-check only (monthly WACR, ends 2026-03)",
            "yield_tb_rn.csv": "cross-check only (SGL T-bill yields, ends 2026-06)"}


MANIFEST = RAW / "MANIFEST.csv"
PREV = {r["file"]: r for r in csv.DictReader(open(MANIFEST, newline=""))} if MANIFEST.exists() else {}


def row(p: Path, source, url, units, period, notes=""):
    b = p.read_bytes()
    sha, rel = hashlib.sha256(b).hexdigest(), str(p.relative_to(ROOT))
    old = PREV.get(rel)
    if old and old["sha256"] == sha:
        utc, ist = old["retrieved_utc"], old["retrieved_ist"]
    else:
        t = dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc)
        utc, ist = t.isoformat(timespec="seconds"), t.astimezone(IST).isoformat(timespec="seconds")
    return {"file": rel, "source": source, "url": url, "retrieved_utc": utc, "retrieved_ist": ist, "units": units, "period": period,
            "bytes": len(b), "sha256": sha, "notes": notes}


rows = []
for p in sorted((RAW / "amfi").glob("am*repo.xls")):
    m, y = p.name[2:5], p.name[5:9]
    head = p.read_bytes()[:4]
    kind = "xlsx(zip) named .xls" if head == b"PK\x03\x04" else ("xls(OLE2)" if head == b"\xd0\xcf\x11\xe0" else "NOT A WORKBOOK")
    pre = int(y) * 12 + MON.index(m) < 2019 * 12 + 3
    rows.append(row(p, "AMFI monthly report (official)", f"https://www.amfiindia.com/spages/{p.name}", "INR crore; folios count",
                    f"{y}-{MON.index(m)+1:02d}", kind + ("; pre-2019 sponsor-wise layout, no separate Liquid/Overnight rows - excluded" if pre else "")
                    + "; first-release (provisional) figures"))
rows.append(row(RAW / "amfi_dl_log.txt", "AMFI download log", "https://www.amfiindia.com/spages/am{mon}{yyyy}repo.xls", "-", "2015-01..2026-12",
                "42 requests returned 404 HTML (Jan2015-Mar2018, Oct-Dec2026); those files were not kept; loader also rejects non-workbooks by magic bytes"))
for p in sorted((RAW / "rbi_dbie").glob("*.csv")):
    meta = p.with_suffix(".meta.json")
    url = json.loads(meta.read_text()).get("url", "") if meta.exists() else "https://data-api.dbie.rbihub.in/api/tables/.../csv"
    rows.append(row(p, "RBI DBIE via dbie.rbihub.in (THIRD-PARTY mirror, not confirmed official)", url, "INR crore; % p.a.",
                    "see file", UNUSABLE.get(p.name, "USED; spot-checked vs official RBI press release 15-Sep-2026 (exact match)")))
idx = {l.split("|")[0]: l.strip().split("|") for l in (RAW / "rbi_mmo/index.txt").read_text().splitlines() if "|" in l}
for p in sorted((RAW / "rbi_mmo").glob("prid_*.html")):
    pid = p.stem.split("_")[1]
    rows.append(row(p, "RBI press release 'Money Market Operations' (official)", f"https://www.rbi.org.in/scripts/BS_PressReleaseDisplay.aspx?prid={pid}",
                    "INR crore; % p.a.", idx.get(pid, ["", ""])[1], "USED for days after mirror end (17-Sep-2026); "
                    "no releases found for 23, 26-28, 30 Sep 2026 in prid 63599-63760"))
p = RAW / "rbi_mmo_crosscheck/prid_63598.html"
rows.append(row(p, "RBI press release (official)", "https://www.rbi.org.in/scripts/BS_PressReleaseDisplay.aspx?prid=63598", "INR crore; % p.a.",
                "2026-09-15", "cross-check of mirror only"))
for d, src, units in [("nse", "NSE official index close archive", "index points"), ("fbil", "FBIL T-Bill benchmark public archive (official)", "% p.a.")]:
    log = {r[0]: r for r in csv.reader(open(RAW / d / "fetch_log.csv"))}
    for p in sorted((RAW / d).iterdir()):
        if p.name == "fetch_log.csv":
            continue
        r = log.get(p.name)
        rows.append(row(p, src, r[1] if r else "", units, p.stem.split("_")[-1], f"fetch_log: {r[-1]}" if r else ""))
with open(MANIFEST, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), "rows")
