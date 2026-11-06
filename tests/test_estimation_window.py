import numpy as np
import pandas as pd
import pytest

from ecs.events import abnormal_returns, estimation_slice


def test_window_ends_strictly_before_day0():
    for p0 in [300, 500, 1000]:
        sl = estimation_slice(p0, (-250, -30))
        assert sl.stop - 1 < p0 and sl.stop - 1 == p0 - 30 and sl.start == p0 - 250


def test_bad_window_rejected():
    with pytest.raises(AssertionError):
        estimation_slice(300, (-250, 0))


def test_post_event_data_cannot_leak_into_alpha_beta():
    """Wrecking every return at/after day 0 must not change the fitted alpha/beta."""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=400)
    mkt = pd.Series(rng.normal(0, 0.01, 400), idx)
    ret = 0.0003 + 1.2 * mkt + pd.Series(rng.normal(0, 0.005, 400), idx)
    d0 = idx[300]
    a = abnormal_returns(ret, mkt, d0)
    ret2 = ret.copy()
    ret2.iloc[300:] += 0.5
    b = abnormal_returns(ret2, mkt, d0)
    assert a["alpha"] == b["alpha"] and a["beta"] == b["beta"]


def test_too_little_estimation_data_returns_none():
    idx = pd.bdate_range("2020-01-01", periods=400)
    s = pd.Series(0.001, idx)
    assert abnormal_returns(s, s, idx[100]) is None  # only ~70 usable estimation days
