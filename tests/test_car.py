import numpy as np
import pandas as pd

from ecs.events import abnormal_returns, car


def _series(jump_at=None, jump=0.0, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=400)
    mkt = pd.Series(rng.normal(0, 0.01, 400), idx)
    ret = 0.0002 + 0.9 * mkt  # noiseless market-model returns
    if jump_at is not None:
        ret.iloc[jump_at] += jump
    return ret, mkt, idx


def test_injected_jump_equals_car():
    ret, mkt, idx = _series(jump_at=300, jump=0.05)
    r = abnormal_returns(ret, mkt, idx[300])
    assert abs(r["beta"] - 0.9) < 1e-9 and abs(r["alpha"] - 0.0002) < 1e-9
    assert abs(car(r["ar"], 0, 1) - 0.05) < 1e-9
    assert abs(car(r["ar"], 0, 5) - 0.05) < 1e-9
    assert abs(car(r["ar"], -5, -1)) < 1e-9


def test_jump_on_day1_only_in_longer_windows():
    ret, mkt, idx = _series(jump_at=301, jump=0.03)
    r = abnormal_returns(ret, mkt, idx[300])
    assert abs(car(r["ar"], 0, 0)) < 1e-9 and abs(car(r["ar"], 0, 1) - 0.03) < 1e-9


def test_market_adjusted_model():
    ret, mkt, idx = _series(jump_at=300, jump=0.04)
    r = abnormal_returns(ret, mkt, idx[300], model="market_adjusted")
    assert abs(r["ar"].loc[0] - (0.0002 - 0.1 * mkt.iloc[300] + 0.04)) < 1e-9
