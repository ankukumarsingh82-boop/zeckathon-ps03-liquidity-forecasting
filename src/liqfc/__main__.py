"""Command-line entry point.

    python -m liqfc run [--skip-backtest]   ingest + validate + backtest + live forecast + dashboard/report tables
    python -m liqfc validate                ingest + validate only
    python -m liqfc refresh [--no-rbi] [--no-run]   fetch new official data, then rerun the pipeline
    python -m liqfc app [--port 8501]       launch the interactive local dashboard (Streamlit, localhost only)
    python -m liqfc docs                    rebuild reports/report.md, figures, report.pdf and deck.pdf from outputs/
    python -m liqfc test                    run the pytest suite
"""
from __future__ import annotations

import argparse
import os
import json
import runpy
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _cfg():
    return yaml.safe_load(open(ROOT / "config.yaml"))


def cmd_validate(_):
    from .amfi import load_amfi, validate_panel
    c = _cfg()["data"]
    panel, meta = load_amfi(ROOT / c["amfi_raw_dir"], c["categories"], c["panel_start"], c["release_lag_default_days"], c["identity_tol_crore"])
    rep = validate_panel(panel, c["categories"], c["panel_start"], c["identity_tol_crore"])
    print(json.dumps({"rows": len(panel), "first": str(panel.month.min()), "last": str(panel.month.max()),
                      "files": len(meta), "validation": rep}, indent=2))


def cmd_run(a):
    from .pipeline import main
    main(skip_backtest=a.skip_backtest)
    runpy.run_path(str(ROOT / "scripts/report_tables.py"), run_name="__main__")
    runpy.run_path(str(ROOT / "scripts/build_report.py"), run_name="__main__")
    runpy.run_path(str(ROOT / "scripts/build_manifest.py"), run_name="__main__")


def cmd_docs(_):
    runpy.run_path(str(ROOT / "scripts/report_tables.py"), run_name="__main__")
    runpy.run_path(str(ROOT / "scripts/build_report.py"), run_name="__main__")
    runpy.run_path(str(ROOT / "scripts/build_docs.py"), run_name="__main__")


def cmd_refresh(a):
    from .refresh import refresh_all
    res = refresh_all(ROOT, _cfg(), rbi=not a.no_rbi)
    (ROOT / "outputs").mkdir(exist_ok=True)
    json.dump(res, open(ROOT / "outputs/refresh_log.json", "w"), indent=1)
    if not a.no_run:
        a.skip_backtest = False
        cmd_run(a)


def cmd_app(a):
    port = a.port or _cfg()["app"]["port"]
    # exec (not a child process) so Ctrl-C / SIGTERM to `python -m liqfc app` stops the server itself
    os.chdir(ROOT)
    os.execv(sys.executable, [sys.executable, "-m", "streamlit", "run", str(ROOT / "app/streamlit_app.py"),
                              "--server.port", str(port), "--server.address", a.address, "--server.headless", "true",
                              "--browser.gatherUsageStats", "false"])


def cmd_test(a):
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", "-q", "-p", "no:warnings", *getattr(a, "pytest_args", [])], cwd=ROOT))


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m liqfc", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")
    r = sub.add_parser("run"); r.add_argument("--skip-backtest", action="store_true", help="reuse outputs/forecasts_backtest_full.csv")
    sub.add_parser("validate")
    sub.add_parser("docs", help="rebuild report.md, figures, report.pdf, deck.pdf")
    f = sub.add_parser("refresh"); f.add_argument("--no-rbi", action="store_true"); f.add_argument("--no-run", action="store_true")
    ap = sub.add_parser("app"); ap.add_argument("--port", type=int); ap.add_argument("--address", default="127.0.0.1")
    t = sub.add_parser("test", help="run pytest; extra args are passed through")
    a, extra = p.parse_known_args(argv)
    if extra and a.cmd != "test":
        p.error(f"unrecognized arguments: {' '.join(extra)}")
    a.pytest_args = extra
    if a.cmd is None:
        a = p.parse_args(["run"])
    {"run": cmd_run, "validate": cmd_validate, "refresh": cmd_refresh, "app": cmd_app, "test": cmd_test, "docs": cmd_docs}[a.cmd](a)


if __name__ == "__main__":
    main()
