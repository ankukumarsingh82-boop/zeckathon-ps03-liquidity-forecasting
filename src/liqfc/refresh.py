"""Incremental refresh of raw data from official public sources (no API keys).

* AMFI monthly report workbooks: months after the latest workbook on disk up to last month. A response is saved
  only if it is a real Excel workbook (OLE2 / ZIP magic bytes); 404/HTML pages are logged as not yet published.
* NSE official daily index-close archive: month-end files after the latest one on disk.
* FBIL T-Bill benchmark archive: previous + current month re-fetched (current month is partial).
* RBI press releases "Money Market Operations as on <date>": press-release ids after the latest one on disk.
The third-party DBIE mirror (dbie.rbihub.in) is deliberately NOT refreshed; later days come from official RBI releases.
"""
from __future__ import annotations

import calendar
import datetime as dt
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

MON = "jan feb mar apr may jun jul aug sep oct nov dec".split()
UA = {"User-Agent": "Mozilla/5.0 (liqfc local research tool)"}
MAGIC = (b"\xd0\xcf\x11\xe0", b"PK\x03\x04")


def _utc():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _months(a: tuple[int, int], b: tuple[int, int]):
    y, m = a
    while (y, m) <= b:
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def _next(y, m):
    return (y + 1, 1) if m == 12 else (y, m + 1)


def _prev(y, m):
    return (y - 1, 12) if m == 1 else (y, m - 1)


def latest_amfi(raw: Path) -> tuple[int, int]:
    best = (2019, 3)
    for p in (raw / "amfi").glob("am*repo.xls"):
        if p.read_bytes()[:4] in MAGIC:
            best = max(best, (int(p.name[5:9]), MON.index(p.name[2:5]) + 1))
    return best


def fetch_amfi(raw: Path, url_tpl: str, today: dt.date, session=requests) -> list[dict]:
    out = []
    last_complete = _prev(today.year, today.month)
    for y, m in _months(_next(*latest_amfi(raw)), last_complete):
        fn = f"am{MON[m-1]}{y}repo.xls"
        url = url_tpl.format(mon=MON[m - 1], yyyy=y)
        try:
            r = session.get(url, headers=UA, timeout=30)
            body, code = r.content, r.status_code
        except requests.RequestException as e:
            body, code = b"", f"ERR {type(e).__name__}"
        ok = code == 200 and body[:4] in MAGIC
        if ok:
            (raw / "amfi" / fn).write_bytes(body)
        status = "saved" if ok else f"not published / not a workbook (HTTP {code})"
        with open(raw / "amfi_dl_log.txt", "a") as f:
            f.write(f"{fn} {code} {len(body)} {_utc()} {'SAVED' if ok else 'REJECTED'}\n")
        out.append({"source": "AMFI", "file": fn, "status": status, "url": url})
    if not out:
        y, m = latest_amfi(raw)
        out.append({"source": "AMFI", "file": f"am{MON[m-1]}{y}repo.xls", "status": "up to date (latest complete month already on disk)"})
    return out


def fetch_nse(raw: Path, url_tpl: str, today: dt.date, session=requests) -> list[dict]:
    d = raw / "nse"
    have = sorted(dt.datetime.strptime(p.stem[-8:], "%d%m%Y").date() for p in d.glob("ind_close_all_*.csv"))
    start = (have[-1].year, have[-1].month) if have else (2019, 1)
    out = []
    for y, m in _months(_next(*start), _prev(today.year, today.month)):
        last = calendar.monthrange(y, m)[1]
        saved = None
        for dd in range(last, last - 8, -1):
            day = dt.date(y, m, dd)
            if day.weekday() >= 5:
                continue
            fn = f"ind_close_all_{day:%d%m%Y}.csv"
            url = url_tpl.format(ddmmyyyy=f"{day:%d%m%Y}")
            try:
                r = session.get(url, headers=UA, timeout=30)
            except requests.RequestException:
                continue
            if r.status_code == 200 and r.content[:10].startswith(b"Index Name"):
                (d / fn).write_bytes(r.content)
                with open(d / "fetch_log.csv", "a") as f:
                    f.write(f"{fn},{url},200,{len(r.content)},{hashlib.sha256(r.content).hexdigest()},{_utc()}\n")
                saved = fn
                break
        out.append({"source": "NSE", "file": saved or f"{y}-{m:02d}", "status": "saved" if saved else "not found"})
    if not out:
        out.append({"source": "NSE", "file": f"ind_close_all_{have[-1]:%d%m%Y}.csv" if have else "-", "status": "up to date (last month-end already on disk)"})
    return out


def fetch_fbil(raw: Path, url_tpl: str, today: dt.date, session=requests) -> list[dict]:
    d = raw / "fbil"
    out = []
    for y, m in [_prev(today.year, today.month), (today.year, today.month)]:
        a, b = dt.date(y, m, 1), dt.date(y, m, calendar.monthrange(y, m)[1])
        url = url_tpl.format(a=a, b=b)
        try:
            r = session.get(url, headers={**UA, "Referer": "https://www.fbil.org.in/"}, timeout=40)
            js = r.json()
            assert isinstance(js, list)
        except Exception as e:  # noqa: BLE001
            out.append({"source": "FBIL", "file": f"tbill_{y}{m:02d}.json", "status": f"failed ({type(e).__name__}); kept old file"})
            continue
        fn = d / f"tbill_{y}{m:02d}.json"
        fn.write_bytes(r.content)
        with open(d / "fetch_log.csv", "a") as f:
            f.write(f"{fn.name},{url},{len(js)},{hashlib.sha256(r.content).hexdigest()},{_utc()}\n")
        out.append({"source": "FBIL", "file": fn.name, "status": f"saved ({len(js)} records)"})
    return out


def _mmo_title(html: str):
    m = re.search(r"Money Market Operations as on [A-Za-z]+ \d+, 20\d\d", html)
    return m.group(0) if m else None


def fetch_rbi_mmo(raw: Path, url_tpl: str, scan: int, session=requests, workers: int = 6) -> list[dict]:
    d = raw / "rbi_mmo"
    idx = d / "index.txt"
    lines = [l for l in idx.read_text().splitlines() if "|" in l] if idx.exists() else []
    start = max([int(l.split("|")[0]) for l in lines] + [int(p.stem.split("_")[1]) for p in d.glob("prid_*.html")] + [63598])

    def get(pid):
        try:
            r = session.get(url_tpl.format(prid=pid), headers=UA, timeout=30)
            return pid, r.text if r.status_code == 200 else ""
        except requests.RequestException:
            return pid, ""

    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(get, range(start + 1, start + 1 + scan)))
    out = []
    for pid, html in res:
        t = _mmo_title(html)
        if t:
            (d / f"prid_{pid}.html").write_text(html, encoding="utf-8")
            with open(idx, "a") as f:
                f.write(f"{pid}|{t}|{_utc()}\n")
            out.append({"source": "RBI", "file": f"prid_{pid}.html", "status": t})
    out.append({"source": "RBI", "file": f"prid {start+1}..{start+scan}", "status": f"scanned, {len(out)} MMO releases saved"})
    return out


def refresh_all(root: Path, cfg: dict, today: dt.date | None = None, rbi: bool = True, log=print) -> list[dict]:
    raw, rc = root / "data/raw", cfg["refresh"]
    today = today or dt.date.today()
    res = []
    for name, fn in [("AMFI", lambda: fetch_amfi(raw, rc["amfi_url"], today)), ("NSE", lambda: fetch_nse(raw, rc["nse_url"], today)),
                     ("FBIL", lambda: fetch_fbil(raw, rc["fbil_url"], today))] + \
                    ([("RBI", lambda: fetch_rbi_mmo(raw, rc["rbi_press_url"], rc["rbi_prid_scan"]))] if rbi else []):
        log(f"refresh {name} ...")
        r = fn()
        for x in r:
            log(f"  {x['source']:5s} {x['file']}: {x['status']}")
        res += r
    return res
