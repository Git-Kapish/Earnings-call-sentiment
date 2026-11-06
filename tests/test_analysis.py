import numpy as np
import pandas as pd

from ecs import analysis as A


def _synthetic(beta=0.01, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    lm = 0.5 * z + rng.normal(size=n) * 0.87
    d = pd.DataFrame({
        "ticker": rng.choice([f"T{i}" for i in range(50)], n),
        "sector": rng.choice(["A", "B", "C"], n),
        "yq": rng.choice(["2020Q1", "2020Q2", "2020Q3"], n),
        "d0": pd.to_datetime("2020-01-01") + pd.to_timedelta(rng.integers(0, 200, n), "D"),
        "fb_net": z, "lm_net": lm, "prior_ret": rng.normal(size=n) * 0.05,
        "pre_vol": rng.uniform(0.01, 0.03, n), "log_dvol": rng.normal(20, 1, n),
    })
    d["fb_net_z"] = (d.fb_net - d.fb_net.mean()) / d.fb_net.std()
    d["lm_net_z"] = (d.lm_net - d.lm_net.mean()) / d.lm_net.std()
    d["car_0_1"] = beta * d.fb_net_z + rng.normal(size=n) * 0.04
    return d


def test_regression_recovers_planted_effect_all_se_types():
    d = _synthetic()
    for cl in ("firm", "two_way", "hc1"):
        r = A.regress(d, ["fb_net_z"], "car_0_1", cl)
        assert abs(r.params["fb_net_z"] - 0.01) < 0.004 and r.tvalues["fb_net_z"] > 2
        assert int(r.nobs) == len(d)


def test_quintile_spread_positive_and_balanced():
    d = _synthetic()
    r = A.quintile_sort(d, "fb_net", "car_0_1")
    assert r["spread"] > 0 and r["t"] > 2
    assert r["table"]["count"].between(290, 310).all() and r["table"]["mean"].is_monotonic_increasing


def test_null_gives_small_t_and_horse_race_runs():
    d = _synthetic(beta=0.0)
    hr = A.horse_race(d, "car_0_1")
    assert len(hr) == 4 and hr.t.abs().max() < 3.5
    assert 0.3 < A.rank_corr(d) < 0.7


def test_winsorize_clips_tails():
    s = pd.Series(np.r_[np.zeros(98), [10, -10]])
    assert A.winsorize(s).abs().max() < 10
