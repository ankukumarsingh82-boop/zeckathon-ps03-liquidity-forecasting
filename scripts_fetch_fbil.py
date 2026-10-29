"""Fetch FBIL T-Bill benchmark rates (public archive endpoint used by fbil.org.in), month by month."""
import calendar, datetime as dt, hashlib, json, os, subprocess
OUT = "/workspace/zeckathon-ps03/data/raw/fbil"
logf = open(os.path.join(OUT, "fetch_log.csv"), "a")
for y in range(2019, 2027):
    for m in range(1, 13):
        if (y, m) < (2019, 1) or (y, m) > (2026, 10):
            continue
        a = dt.date(y, m, 1); b = dt.date(y, m, calendar.monthrange(y, m)[1])
        fn = os.path.join(OUT, f"tbill_{y}{m:02d}.json")
        if os.path.exists(fn) and os.path.getsize(fn) > 100:
            continue
        url = f"https://www.fbil.org.in/wasdm/tbill/fetchfiltered?fromDate={a}&toDate={b}&authenticated=false"
        r = subprocess.run(["curl", "-s", "-m", "40", "-A", "Mozilla/5.0", "-H", "Referer: https://www.fbil.org.in/", url], capture_output=True)
        try:
            d = json.loads(r.stdout); assert isinstance(d, list)
        except Exception:
            logf.write(f"{fn},{url},FAILED,{dt.datetime.utcnow().isoformat()}Z\n"); continue
        open(fn, "wb").write(r.stdout)
        logf.write(f"{os.path.basename(fn)},{url},{len(d)},{hashlib.sha256(r.stdout).hexdigest()},{dt.datetime.utcnow().isoformat()}Z\n"); logf.flush()
print("DONE")
