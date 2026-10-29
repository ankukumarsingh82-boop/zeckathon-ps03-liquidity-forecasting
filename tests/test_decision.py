import pandas as pd
import pytest

from liqfc.decision import buffer_from_quantiles, cost, evaluate_policies

TAUS = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]


def qrow(vals):
    return {f"q_{t}": v for t, v in zip(TAUS, vals)}


def test_buffer_is_upper_quantile_of_outflow():
    r = qrow([-1000, -800, -600, -300, 0, 300, 600, 800, 1000])   # net-flow quantiles
    assert buffer_from_quantiles(r, 0.95, TAUS) == pytest.approx(800)   # 95% outflow quantile = -q_0.05(net)
    assert buffer_from_quantiles(r, 0.75, TAUS) == pytest.approx(300)
    r2 = qrow([100, 200, 300, 400, 500, 600, 700, 800, 900])          # all inflows -> zero buffer
    assert buffer_from_quantiles(r2, 0.95, TAUS) == 0.0
    with pytest.raises(ValueError):
        buffer_from_quantiles(r, 0.99, TAUS)


def test_cost_formula():
    # buffer 1200, outflow 1000, spread 1.2% p.a. -> idle 200*0.012/12 = 0.2
    c = cost(1200, 1000, 1.2, 3.0)
    assert float(c["idle_cost"]) == pytest.approx(0.2) and float(c["shortfall_cost"]) == 0
    # buffer 1000, outflow 1200 -> shortfall 3*200*0.012/12 = 0.6
    c = cost(1000, 1200, 1.2, 3.0)
    assert float(c["shortfall_cost"]) == pytest.approx(0.6) and float(c["idle_cost"]) == 0 and int(c["breach"]) == 1
    # inflow month: whole buffer idle
    assert float(cost(500, 0, 1.2)["total_cost"]) == pytest.approx(0.5)


def test_policy_and_smly_baseline_use_actual_t_minus_12():
    months = pd.period_range("2024-01", "2025-03", freq="M")
    panel = pd.DataFrame({"month": months, "category": "Liquid Fund", "net": [-(i + 1) * 100.0 for i in range(len(months))]})
    drivers = pd.DataFrame({"tbill_3m": 6.2, "wacr": 6.0}, index=months)
    fc = pd.DataFrame([{**qrow([-1000, -800, -600, -300, 0, 300, 600, 800, 1000]), "model": "m", "h": 2,
                        "target": pd.Period("2025-02", "M"), "category": "Liquid Fund", "actual": -1400.0}])
    out = evaluate_policies(fc, panel, drivers, 2, TAUS, [0.95], 3.0, 0.10)
    base = out[out.policy == "smly_baseline"].iloc[0]
    assert base["buffer"] == pytest.approx(200.0)                         # -net(2024-02) = 200
    assert base["outflow"] == pytest.approx(1400.0)
    assert base["shortfall_cost"] == pytest.approx(3 * 1200 * 0.2 / 100 / 12)
    pol = out[out.policy == "q95"].iloc[0]
    assert pol["buffer"] == pytest.approx(800) and pol["spread_pct"] == pytest.approx(0.2)
    # spread floor applies
    drivers2 = pd.DataFrame({"tbill_3m": 5.0, "wacr": 6.0}, index=months)
    assert evaluate_policies(fc, panel, drivers2, 2, TAUS, [0.95], 3.0, 0.10).iloc[0]["spread_pct"] == pytest.approx(0.10)


def test_paired_bootstrap_sign():
    from liqfc.decision import paired_vs_baseline
    t = pd.period_range("2024-01", "2025-12", freq="M")
    rows = [{"decision_h": 2, "spread_basis": "x", "category": "c", "model": "baseline", "policy": "smly_baseline", "target": m, "total_cost": 10.0} for m in t]
    rows += [{"decision_h": 2, "spread_basis": "x", "category": "c", "model": "m", "policy": "q75", "target": m, "total_cost": 7.0} for m in t]
    r = paired_vs_baseline(pd.DataFrame(rows), n_boot=200).iloc[0]
    assert r["total_diff_cr"] == pytest.approx(-72) and r["diff_ci90_hi"] < 0 and r["months_cheaper"] == 24


def test_penalty_bases_spread_and_amount():
    # buffer 1000, outflow 1200, spread 1.2% p.a.: shortfall 200
    a = cost(1000, 1200, 1.2, 3.0, penalty_basis="spread")
    b = cost(1000, 1200, 1.2, 3.0, penalty_basis="amount")
    assert float(a["shortfall_cost"]) == pytest.approx(3 * 200 * 1.2 / 1200)    # (a) 3x spread on the shortfall
    assert float(b["shortfall_cost"]) == pytest.approx(3 * 200)                 # (b) 3x the shortfall amount, in Rs
    assert float(cost(1000, 1200, 1.2, 3.0)["shortfall_cost"]) == float(a["shortfall_cost"])   # default = primary (a)
    # idle cost is the same under both readings
    assert float(cost(1500, 1200, 1.2, penalty_basis="amount")["idle_cost"]) == pytest.approx(300 * 1.2 / 1200)
    assert float(cost(1500, 1200, 1.2, penalty_basis="spread")["idle_cost"]) == pytest.approx(300 * 1.2 / 1200)
    with pytest.raises(ValueError):
        cost(1000, 1200, 1.2, penalty_basis="rupees")
