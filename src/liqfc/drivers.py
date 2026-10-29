"""Descriptive driver analysis (in-sample, HAC standard errors) with rate-regime break robustness.
Out-of-sample driver evidence comes from the LightGBM feature-group ablation in the backtest."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .features import calendar_frame, driver_features

BREAKS = {"post_sdf_2022_04": pd.Period("2022-04", "M"),       # SDF introduced as LAF floor, 8 Apr 2022
          "post_mibor_2025_08": pd.Period("2025-08", "M")}     # FBIL MIBOR methodology change, 4 Aug 2025


def calendar_driver_regression(panel: pd.DataFrame, drivers: pd.DataFrame) -> pd.DataFrame:
    out = []
    df_d = driver_features(drivers)
    for cat, g in panel.groupby("category"):
        s = g.set_index("month").sort_index()
        y = (s["net"] / s["aum"].shift(1) * 100).rename("flow_pct_aum")   # net flow as % of previous AUM
        cal = calendar_frame(s.index)
        X0 = pd.DataFrame({"qe_jun_sep_dec": cal["adv_tax"] - cal["fy_end"], "march_fy_end": cal["fy_end"],
                           "april": cal["fy_start"], "q_start_jul_oct_jan": cal["q_start"]}, index=s.index)
        drv = df_d.reindex(s.index)
        specs = {
            "calendar": X0,
            "calendar+drivers(contemporaneous)": X0.join(drv[["wacr", "term_spread", "net_liq_inj_lakhcr", "nifty_ret_1m"]]),
            "calendar+drivers+breaks": X0.join(drv[["wacr", "term_spread", "net_liq_inj_lakhcr", "nifty_ret_1m"]])
                .assign(**{k: (s.index >= v).astype(int) for k, v in BREAKS.items()}),
        }
        for name, X in specs.items():
            d = pd.concat([y, X], axis=1).dropna()
            res = sm.OLS(d["flow_pct_aum"], sm.add_constant(d.drop(columns="flow_pct_aum"))).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
            for v in res.params.index:
                out.append({"category": cat, "spec": name, "variable": v, "coef_pct_of_aum": res.params[v],
                            "hac_se": res.bse[v], "p_value": res.pvalues[v], "n": int(res.nobs), "r2": res.rsquared,
                            "sample": f"{d.index.min()}..{d.index.max()}"})
    return pd.DataFrame(out)
