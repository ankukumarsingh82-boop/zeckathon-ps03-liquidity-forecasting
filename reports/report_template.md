# PS 03: Forecasting Liquid & Overnight net flows and sizing the cash buffer

_All numbers below are produced by `python -m liqfc run` and rendered from `outputs/*.csv` by `scripts/report_tables.py` and `scripts/build_report.py` (this file is generated from `reports/report_template.md`; the numbers in sentences are filled in from the CSVs). Units are ₹ crore; negative net flow = net outflow._

> **Penalty definition (read this first).** The brief prices decision cost as "yield given up on idle cash + a 3× penalty on any redemption shortfall". This report uses **reading (a), primary:** each rupee short costs **3 × the monthly idle-cash spread** (3 × shortfall × s/1200, where s is FBIL 3M T-bill − WACR, floored at 0.10%). **Reading (b)** charges **3 × the shortfall amount itself** (₹3 per ₹1 short). Both are computed everywhere (`outputs/decision_costs.csv`, column `penalty_basis`; §3a; dashboard). {{fact:flip_sentence}} The forecast-based q75 buffer from `sarimax_cal` beats the naive baseline under **both** readings in both categories (Liquid {{cost:Liquid Fund|sarimax_cal|q75|spread|pct}} / {{cost:Liquid Fund|sarimax_cal|q75|amount|pct}}; Overnight {{cost:Overnight Fund|sarimax_cal|q75|spread|pct}} / {{cost:Overnight Fund|sarimax_cal|q75|amount|pct}}). {{fact:liquid_a_ci_note}} (Overnight: {{fact:overnight_sarimax_q75_ci}}). The choice between q75 and q95 depends on the reading.

## 1. Data
* **AMFI** monthly reports (official, `amfiindia.com/spages/am{mon}{yyyy}repo.xls`). There are 102 genuine workbooks. 75 are xlsx content saved under a .xls name and 27 are OLE2 xls, so the parser picks the engine from magic bytes. 42 URLs (Jan-2015 to Mar-2018 and Oct-Dec-2026) returned 404 HTML pages. Those pages were not kept, and the loader rejects any non-workbook and logs it.
  * Apr-2018 to Mar-2019 files use the old sponsor-wise layout, with a single combined Liquid/Money-Market row, so they are excluded. **The panel runs 2019-04 to 2026-09: 90 months × 2 categories.**
  * Validation passes: no duplicates or gaps, sales − redemptions = net within 0.1 cr (max residual ~1e-9), no negative gross flows, AUM-continuity unit checks hold, and the panel matches the auditor's independent extract on all 180 rows.
  * Six 2020 files have no unit text. Their units were confirmed through AUM continuity.
  * One robust outlier is flagged and kept: Overnight 2019-11.
  * AMFI figures are first releases and are treated as provisional.
* **Release dates** come from the "Released on" stamp: the lag after month-end is 8 to 13 days. Where a file has no stamp (2019-05, 2020-01, 2021-12 and 2025-09 to 2026-09) the date is set to month-end + 13 days, the maximum observed lag.
* **Drivers** (`data/raw/MANIFEST.csv` has URL, retrieval time in UTC and IST, sha256 and notes for {{fact:manifest_files}} files):
  * WACR: RBI daily call rates.
  * Net LAF liquidity: RBI Money Market Operations "net liquidity from today's operations", from Jul-2020 onward.
  * FBIL 3M T-bill benchmark: official public archive.
  * Nifty 50 month-end close: NSE official archive files.
  * Daily RBI series up to 17-Sep-2026 come from `dbie.rbihub.in`. This is a **third-party DBIE mirror, not confirmed official**. It matched the official RBI press release of 15-Sep-2026 exactly (test `test_mirror_matches_official_release_sep15`). Days after 17-Sep come from official RBI press releases. 23, 26–28 and 30 Sep 2026 were not found, so September liquidity averages 23 days.
  * DBIE mirror tables that were unusable (policy rates with empty values, T-bill auctions with corrupted decimals, a Nifty table with duplicate months) are listed in the manifest and not used.

## 2. Method
* **Classical models:**
  * `snaive`: y(t−12), with PIs from in-sample seasonal differences.
  * `snaive_ratio`: the same, applied to flow/lagged AUM and rescaled by AUM_T.
  * `ets_ratio`: ETS(A,N,A) on that ratio.
  * `sarimax_cal`: AR(1) on the ratio with calendar dummies (Jun/Sep/Dec advance-tax quarter-ends, March FY-end, April, other quarter starts).
* **ML model:** `lgbm_q`, quantile LightGBM with 9 quantiles and sorted outputs so they cannot cross. It is pooled over both categories and all horizons, fits each horizon directly, and predicts the target y(t+h)/AUM_t.
  * Features: target-month calendar, ratio lags, the same month last year (and the year before), 12-month AUM growth, WACR level and 3-month change, T-bill−WACR spread, net LAF, and Nifty 1-month and 3-month returns.
  * Hyper-parameters were fixed in `config.yaml` beforehand and not tuned.
* **Out-of-sample, horizon-specific PIs** (`*_oosq`): at each origin, the quantiles of that model's own past out-of-sample errors (scaled by AUM) for the same category and horizon, using only targets already released. A model gets these PIs once it has at least 12 such errors.
* **Rolling-origin backtest:** origins run from 2022-03 to 2026-08 and every model is refit at each origin. The headline window is the 24 targets Oct-2024 to Sep-2026 for h = 1, 2 and 3, which gives 72 forecasts per model per category.
* **Nested selection:** each category's model is chosen on pinball loss over Oct-2024 to Sep-2025 only, then reported on Oct-2025 to Sep-2026.
* **No-leakage checks** (tests):
  * Scrambling all data after the origin leaves every model's forecast unchanged.
  * ML features built from data truncated at T equal those built from the full data.
  * A driver whose `available_date` falls after the release date is dropped.
  * Backtest rows always have target > origin and as_of < target release.
  * Removing the truncation in `info_set` makes 6 of these tests fail (checked by deliberately breaking the code).

## 3. Results: accuracy, calibration and decision cost (24 targets × h = 1–3; decision uses h=2)
* **MAPE handling:** MAPE is reported on all points, as the spec requires. Net flows cross zero, so MAPE is dominated by small months: Overnight snaive is 497%, but only 133% once |y| < 2,000 cr is excluded. Read median APE, MASE and pinball alongside it.
* The seasonal-naive figures reproduce the auditor's reference values exactly: Liquid MAPE 137%, median APE 53%, MAE 33,199; Overnight MAPE 497%, median APE 81%, MAE 15,804.
* Coverage is pooled over 72 overlapping forecasts. Because those forecasts are not independent, the Wilson intervals are optimistic.

{{include:tables_metrics.md}}

![Empirical coverage of the 80% and 95% intervals with Wilson 95% CIs](figures/coverage.png)

![h=2 backtest of the selected models](figures/backtest_h2.png)

Per-horizon detail: reports/tables_by_h.md. Nested selection:

{{include:tables_nested.md}}

**What the results show.**
1. **Accuracy.** `sarimax_cal` is best for both categories on pooled pinball and MAE.
   * Liquid: pinball 6,563 vs 7,267 for snaive (−10%), MASE 0.78.
   * Overnight: pinball 3,172 vs 3,426 (−7%).
   * Its gain over seasonal-naive is modest, and Overnight MASE is about 1.0, so Overnight is essentially unforecastable beyond seasonality.
2. **The ML challenger does not beat the classical models.** Raw `lgbm_q` is the worst model on pinball and MAE for Liquid (MAE 42,429 vs 31,242), and its PIs under-cover: Overnight 80% PI covers 46% [35%, 57%]. The quantile GBM overfits on about 200 pooled training rows. Recalibrating its intervals with past out-of-sample errors (`lgbm_q_oosq`) repairs coverage (Liquid 81%) but not accuracy.
3. **Coverage.** The classical models are roughly calibrated at 80%: `sarimax_cal` gives Liquid 82% and Overnight 79%. snaive over-covers Liquid (92% / 100%), i.e. its intervals are too wide. Overnight 95% intervals tend to under-cover (83–94%).
4. **Nested selection** picks `sarimax_cal` for Liquid. For Overnight it picks `{{fact:overnight_sel_model}}`, which then ranks {{fact:overnight_sel_rank}} on the evaluation window (pinball {{fact:overnight_sel_pinball}}; best was {{fact:overnight_eval_best}}). With only 12 selection months the choice is noisy. I report this as it came out and did not re-pick on the test window.

**Decision layer, penalty reading (a)** (h=2, FBIL 3M − WACR spread; 90% moving-block-bootstrap CI of the 24-month total cost difference vs the baseline):

{{include:tables_decision.md}}

* **Under reading (a), the spec's 95th-percentile buffer is too conservative for Liquid.** Across the {{fact:n_models}} forecasters it costs {{fact:liquid_q95_range}} relative to the naive baseline, and the 90% CI excludes 0 on the costly side for {{fact:liquid_q95_n_ci_excl0_costlier}}. For Overnight, q95 changes cost by {{fact:overnight_q95_range}} ({{fact:overnight_q95_n_cheaper}} cheaper than baseline). Under reading (b) this conclusion reverses (§3a). Under this cost function the cost-minimising quantile is the newsvendor critical ratio 3/(3+1) = **0.75**. That value is set in `config.yaml` and was not tuned on the test window.
* **The q75 policy beats the baseline for Overnight.** With `sarimax_cal` the cost changes by {{cost:Overnight Fund|sarimax_cal|q75|spread|pct}}. Across the {{fact:n_models}} forecasters the 24-month cost changes by {{fact:overnight_q75_range}} ({{fact:overnight_q75_n_cheaper}} cheaper than baseline; 90% CI excludes 0 on the cheap side for {{fact:overnight_q75_n_ci_excl0_cheaper}}).
* **For Liquid, q75 + `sarimax_cal` changes cost by {{cost:Liquid Fund|sarimax_cal|q75|spread|pct}}** ({{fact:liquid_q75_label}}).
* The sign of these results holds when the spread is a constant 0.50% (`reports/tables_decision_sensitivity.md`). The h=1 rows there are an oracle-timing check only: the M−1 release arrives around the 10th of M, after the day-1 decision, so h=1 is not feasible live.
* **Costs are small in ₹** (tens of crore over 24 months) because the T-bill − WACR spread sat at the 0.10% floor in 18 of 24 months; WACR traded above the 3M T-bill. Rankings are therefore driven largely by shortfall versus idle amounts, not by the spread level.
* **Net, not gross, category outflow** is used. A single fund should scale the buffer by its share of category AUM.

## 3a. Penalty reading: both cost tables and whether the conclusion flips
**Reading (a), primary:** shortfall cost = 3 × shortfall × s/1200, so idle and shortfall are priced in the same unit (₹ of carry per month) and the penalty is three times the idle rate. **Reading (b):** shortfall cost = 3 × shortfall (₹). Idle cost is idle × s/1200 under both. h=2, realised spread, 24 months Oct-2024 to Sep-2026; full tables for six models in `reports/tables_penalty.md`.

{{include:tables_penalty_compact.md}}

**Does q75 vs q95 flip?**

{{include:tables_penalty_flip.md}}

![24-month cost vs buffer quantile under both penalty readings (h=2)](figures/cost_tau.png)

{{fact:flip_sentence}}

**Why (a) is primary.**
1. The brief puts the penalty next to "yield given up on idle cash", which is a rate. The natural reading is a 3× rate on the rupees that are short, so both terms are in the same unit. That gives a well-posed newsvendor problem whose optimal buffer is the 3/(3+1) = 0.75 quantile; `policy_quantiles` in `config.yaml` sets it, and it was not tuned on the test window.
2. Under (b) a rupee short costs ₹3, while a rupee idle costs s/1200 ≈ ₹0.00008 a month at the 0.10% floor, a ratio of about 36,000:1. The optimal quantile is then about 0.99997. Any buffer below the largest plausible outflow is dominated, the spread data play no role, and the stored 2.5–97.5% quantiles cannot even reach the optimum. A 300% charge on principal is also not what a shortfall costs a liquid fund, which can meet redemptions by selling T-bills/CPs or by short-term borrowing (SEBI (Mutual Funds) Regulations 1996, Reg. 44(2), allows borrowing up to 20% of net assets for up to 6 months for redemptions).

**What to do with it.** If the judges or the fund intend (b), use q95: for November 2026 that is the "Buffer q95" column in §5. Under (a), q75 is the cost-minimising rule. The q75 buffer from `sarimax_cal` beats the baseline under both readings.

## 4. Drivers
![Calendar effects on net flow as % of AUM](figures/drivers.png)

* **In-sample HAC regression** (`outputs/driver_regression.csv`, flow as % of prior AUM):
  * Liquid: Jun/Sep/Dec advance-tax quarter-ends −14.5 pp, March FY-end −22.6 pp, April +18.0 pp (all p < 0.001), R² = 0.63 with the calendar alone.
  * Adding contemporaneous WACR, spread, net LAF and Nifty raises Liquid R² only to 0.63–0.65.
  * Overnight: April (+25 pp) and quarter starts (+23 pp) matter. WACR (−3.7 pp per 1 pp, p = 0.01) and net LAF (+3.1 pp per ₹1 lakh cr injection, p = 0.02) are significant contemporaneously, R² = 0.40.
* **Out-of-sample** (LGBM feature-group ablation, `reports/tables_ablation.md`): removing the rates, liquidity or equity features does **not** worsen the forecasts. The calendar-only LGBM has the *lowest* pinball for both categories, so market drivers measured at the origin add no forecasting value beyond calendar effects and target lags at h = 1–3.
* **Regime-break robustness (auditor item 6):** WACR (RBI) is used, not FBIL MIBOR, so the 4-Aug-2025 MIBOR methodology change does not enter the features directly.
  * Break dummies for the SDF floor (8-Apr-2022) and post-Aug-2025 were added to the regression. Calendar coefficients are stable to within 1–3 pp.
  * Post-Aug-2025 is +5.1 pp for Liquid (p = 0.03), which is a level shift rather than a rate effect.

## 5. Live forecast: origin Sep-2026 (as-of 13-Oct-2026), next release = Oct-2026 data (~10-Nov-2026)

{{include:tables_live_selected.md}}

![Live forecast fans for the selected models](figures/live_fan.png)

All models' live forecasts: `reports/tables_live.md`.

**Recommended buffer for November 2026** (decision on 1-Nov using the h=2 forecast, q75 policy under penalty reading (a); q95 if reading (b) applies):
* **Liquid:** ₹{{fact:liquid_sarimax_cal_h2_buffer_q75}} cr from `sarimax_cal`. The q95 buffer is ₹{{fact:liquid_sarimax_cal_h2_buffer_q95}} cr; the baseline is ₹{{fact:liquid_sarimax_cal_h2_baseline_buffer_smly}} cr.
* **Overnight:** ₹{{fact:overnight_lgbm_q_oosq_h2_buffer_q75}} cr from `lgbm_q_oosq`, the nested-selection pick. The q95 buffer is ₹{{fact:overnight_lgbm_q_oosq_h2_buffer_q95}} cr; `sarimax_cal` would give ₹{{fact:overnight_sarimax_cal_h2_buffer_q75}} cr; the baseline is ₹{{fact:overnight_lgbm_q_oosq_h2_baseline_buffer_smly}} cr.

**October 2026 point forecast (one figure per category).** Rule, stated in `config.yaml` (`submission: rule: nested_selection, horizon: 1`) on 29-Oct-2026, before AMFI publishes October data (~10-Nov): each category's figure is the h=1 forecast of the model chosen by nested selection, the same model that sets the buffer.

* {{fact:liquid_submitted_line}}
* {{fact:overnight_submitted_line}}

**Why `lgbm_q_oosq` and not `sarimax_cal` for Overnight.** Nested selection picked `lgbm_q_oosq` on Oct-2024 to Sep-2025. On the later evaluation window it ranks {{fact:overnight_evalrank_lgbm_q_oosq}} and `sarimax_cal` ranks {{fact:overnight_evalrank_sarimax_cal}}. Switching to `sarimax_cal` because it did better on the evaluation window would mean choosing on the window that is meant to stay untouched, which is what nested selection exists to prevent. v0.2 quoted `sarimax_cal` for both categories without a stated rule; that is withdrawn. For reference only, `sarimax_cal` gives {{fact:overnight_sarimax_cal_h1_point}} cr for Overnight October (80% PI {{fact:overnight_sarimax_cal_h1_q_0.1}} to {{fact:overnight_sarimax_cal_h1_q_0.9}}).

December, an advance-tax month, is expected to see a large Liquid outflow: {{fact:liquid_sarimax_cal_h3_point}} cr, with a q75 buffer of ₹{{fact:liquid_sarimax_cal_h3_buffer_q75}} cr.

## 6. Data gaps and risks
* AMFI data are first releases, and 16 release dates are imputed (month-end + 13 d).
* The daily RBI series before 17-Sep-2026 come from a third-party DBIE mirror, verified on one day. 5 days are missing in Sep-2026. Net LAF is unavailable before Jul-2020 (LightGBM treats it as missing).
* The evaluation covers only 24 targets; coverage CIs are wide and the forecasts overlap.
* Nested selection on 12 months is unstable (see Overnight).
* The cost function uses an ex-post realised monthly spread that is floor-dominated.
* Category-level net flow is not a single fund's gross redemptions.
* The MIBOR/SDF breaks are tested only in-sample.

## 7. Auditor findings
| # | Finding | Verified? | Fix | Regression test | Effect |
|---|---|---|---|---|---|
| 1 | Apr-18 to Mar-19 files have only a combined Liquid/MM row | Yes: old sponsor-wise layout | Panel starts 2019-04 (`config.data.panel_start`); the parser marks that layout and returns no rows | `test_pre_2019_files_yield_no_category_rows`, `test_panel_starts_2019_04_with_90_months_each` | 90×2 panel, no category mixing |
| 2 | 42 .xls files are 404 HTML | Yes (download log) | Files deleted; loader raises `NotAWorkbook` on non-workbooks and logs `REJECTED` | `test_html_404_rejected_and_logged` | 102 real files, 0 rejected now on disk |
| 3 | xlsx named .xls; identity check | Yes: 75 ZIP, 27 OLE2 | Engine chosen by magic bytes; hard assert that sales−red−net is within 0.1 | `test_engine_from_magic_bytes_not_extension`, `test_flow_identity_within_tolerance`, `test_matches_independent_auditor_extract` | Max residual ~1e-9; all 180 rows match the auditor |
| 4 | Release lag 6–13 d | Yes: observed 8–13 d | Origin as_of = release stamp or month-end+13; drivers need `available_date` ≤ as_of; OOS-PI only uses released targets | `test_info_set_*`, `test_driver_not_available_by_release_is_dropped`, `test_forecast_invariant_to_future_data[*]`, `test_ml_features_equal_under_truncation`, `test_backtest_rows_are_out_of_sample`, `test_oos_calibration_only_uses_released_errors`, `test_release_dates_after_month_end` | Backtest is leakage-free by construction; mutation check confirms the tests bite |
| 5 | Report MAE, sMAPE/MASE, median APE | Agreed | `metrics.summarize` adds median APE, sMAPE, WAPE, MAE, RMSE, MASE, scaled pinball, MAPE excluding small months, and Wilson CIs | `test_point_metrics`, `test_coverage_and_wilson`, `test_pinball_sign_and_values` | snaive reproduces the auditor's numbers exactly |
| 6 | MIBOR methodology break 4-Aug-2025 | Yes, but MIBOR is not used (WACR is) | Break dummies (post-Aug-2025, post-SDF Apr-2022) in the driver regression; LGBM rate-feature ablation | (driver regression output) | Calendar effects stable; rates add no OOS value |

Other auditor points:
* No tuning on the test window: hyper-parameters are fixed and selection is nested.
* Horizon-specific out-of-sample PIs: the `_oosq` variants.
* Wilson CIs on coverage: added.
* Provisional-data note: added.
* The DBIE mirror is flagged as third-party.
* SDF break: dummy added.
* Dashboard numbers trace to CSVs: each block names its source file.
* The baseline uses actual(t−12): `test_policy_and_smly_baseline_use_actual_t_minus_12`.

## 8. Next steps
1. Score the October 2026 figures in §5 when AMFI publishes it (~10-Nov), and extend the evaluation with gross redemptions (a fund-level buffer needs gross, not net, flows).
2. Use a longer selection window (or forecast combination of `sarimax_cal` + `snaive_ratio`) in place of 12-month nested selection, which proved unstable for Overnight. Also add day-of-month intramonth timing of advance-tax outflows (around the 15th) for intra-month buffers.
3. Replace the third-party DBIE mirror with direct RBI DBIE downloads, and fill the 5 missing September days from RBI weekly statistical supplements.

## Local application
Everything above can be reproduced with `python -m liqfc run`. The interactive local dashboard (`python -m liqfc app`, http://127.0.0.1:8501) reads only `outputs/`. Its τ, horizon, spread and penalty controls recompute the decision costs from the backtest quantiles, actual flows and realised spreads. A sidebar switch selects penalty reading (a) or (b), and the Decision-cost tab shows both readings side by side. At τ = 0.75/0.95 with the default settings it reproduces `outputs/decision_costs.csv` exactly under both readings (tests/test_app.py). `python -m liqfc refresh` pulls new months from AMFI, NSE, FBIL and RBI. See README.
