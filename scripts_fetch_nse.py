"""Fetch NSE official daily index-close files (ind_close_all_DDMMYYYY.csv) for the last trading day of each month."""
import datetime as dt, hashlib, os, subprocess, calendar, csv
OUT = "/workspace/zeckathon-ps03/data/raw/nse"
log = open(os.path.join(OUT, "fetch_log.csv"), "a", newline="")
w = csv.writer(log)
for y in range(2019, 2027):
    for m in range(1, 13):
        if (y, m) < (2019, 1) or (y, m) > (2026, 9):
            continue
        last = calendar.monthrange(y, m)[1]
        for d in range(last, last - 8, -1):
            day = dt.date(y, m, d)
            if day.weekday() >= 5:
                continue
            fn = f"ind_close_all_{day:%d%m%Y}.csv"
            p = os.path.join(OUT, fn)
            if os.path.exists(p):
                break
            url = f"https://nsearchives.nseindia.com/content/indices/{fn}"
            r = subprocess.run(["curl", "-s", "-m", "25", "-A", "Mozilla/5.0", "-o", p, "-w", "%{http_code}", url], capture_output=True, text=True)
            ok = r.stdout == "200" and open(p, "rb").read(10).startswith(b"Index Name")
            if ok:
                w.writerow([fn, url, r.stdout, os.path.getsize(p), hashlib.sha256(open(p, "rb").read()).hexdigest(), dt.datetime.utcnow().isoformat() + "Z"]); log.flush()
                break
            if os.path.exists(p):
                os.remove(p)
print("DONE")
