"""Calendar + origin-time features. Every function takes data already truncated to the origin's information set."""
from __future__ import annotations

import numpy as np
import pandas as pd

ADV_TAX_MONTHS = {6, 9, 12, 3}      # advance-tax instalments due 15 Jun / 15 Sep / 15 Dec / 15 Mar (Income Tax Dept)


def calendar_frame(months: pd.PeriodIndex) -> pd.DataFrame:
    mo = np.asarray(months.month)
    return pd.DataFrame({
        "adv_tax": np.isin(mo, list(ADV_TAX_MONTHS)).astype(int),     # also = calendar quarter-end months
        "fy_end": (mo == 3).astype(int),                                 # March: FY-end + final advance-tax instalment
        "fy_start": (mo == 4).astype(int),
        "q_start": np.isin(mo, [1, 7, 10]).astype(int),                 # first month of Q2-Q4 (re-investment after tax)
        "month_sin": np.sin(2 * np.pi * mo / 12), "month_cos": np.cos(2 * np.pi * mo / 12),
    }, index=months)


def sarimax_exog(months: pd.PeriodIndex) -> pd.DataFrame:
    c = calendar_frame(months)
    return pd.DataFrame({"qe_jun_sep_dec": c["adv_tax"] - c["fy_end"], "march": c["fy_end"],
                         "april": c["fy_start"], "q_start": c["q_start"]}, index=months).astype(float)


def driver_features(drv: pd.DataFrame) -> pd.DataFrame:
    """Monthly driver features from a driver table truncated at the origin (index = month)."""
    d = drv.sort_index()
    f = pd.DataFrame(index=d.index)
    f["wacr"] = d["wacr"]
    f["d_wacr_3m"] = d["wacr"] - d["wacr"].shift(3)
    f["term_spread"] = d["tbill_3m"] - d["wacr"]                       # T-bill 3M (FBIL) minus overnight call rate
    f["net_liq_inj_lakhcr"] = d["net_liq"] / 1e5                         # RBI LAF today's-ops net injection (+) / absorption (-)
    f["nifty_ret_1m"] = np.log(d["nifty_close"] / d["nifty_close"].shift(1))
    f["nifty_ret_3m"] = np.log(d["nifty_close"] / d["nifty_close"].shift(3))
    return f


DRIVER_GROUPS = {
    "rates": ["wacr", "d_wacr_3m", "term_spread"],
    "liquidity": ["net_liq_inj_lakhcr"],
    "equity": ["nifty_ret_1m", "nifty_ret_3m"],
}
