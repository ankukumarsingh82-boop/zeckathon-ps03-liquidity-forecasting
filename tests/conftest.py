import sys
import warnings
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def cfg():
    return yaml.safe_load(open(ROOT / "config.yaml"))


@pytest.fixture(scope="session")
def panel(cfg):
    from liqfc.amfi import load_amfi
    p, _ = load_amfi(ROOT / cfg["data"]["amfi_raw_dir"], cfg["data"]["categories"], cfg["data"]["panel_start"],
                     cfg["data"]["release_lag_default_days"], cfg["data"]["identity_tol_crore"])
    return p


@pytest.fixture(scope="session")
def amfi_meta(cfg):
    from liqfc.amfi import load_amfi
    return load_amfi(ROOT / cfg["data"]["amfi_raw_dir"], cfg["data"]["categories"], cfg["data"]["panel_start"])[1]


@pytest.fixture(scope="session")
def drivers():
    from liqfc.macro import build_monthly_drivers
    return build_monthly_drivers()
