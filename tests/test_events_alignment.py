import pandas as pd
import pytest

from ecs.events import day0

# Sessions around a Friday->Monday and the July-4th 2024 holiday (Thu Jul 4 closed)
IDX = pd.DatetimeIndex(pd.bdate_range("2024-06-24", "2024-07-12").difference([pd.Timestamp("2024-07-04")]))
D = pd.Timestamp


def test_before_open_is_call_date():
    assert day0(D("2024-06-25"), "before_open", IDX) == D("2024-06-25")


def test_after_close_is_next_trading_day():
    assert day0(D("2024-06-25"), "after_close", IDX) == D("2024-06-26")


def test_friday_after_close_is_monday():
    assert day0(D("2024-06-28"), "after_close", IDX) == D("2024-07-01")


def test_holiday_skipped():
    assert day0(D("2024-07-03"), "after_close", IDX) == D("2024-07-05")  # Jul 4 closed
    assert day0(D("2024-07-04"), "before_open", IDX) == D("2024-07-05")  # call on the holiday itself
    assert day0(D("2024-06-29"), "before_open", IDX) == D("2024-07-01")  # Saturday call


def test_during_rules_differ():
    assert day0(D("2024-06-25"), "during", IDX, "prd") == D("2024-06-26")
    assert day0(D("2024-06-25"), "during", IDX, "alt") == D("2024-06-25")


def test_unknown_never_guessed():
    assert day0(D("2024-06-25"), "unknown", IDX, "prd") is None
    assert day0(D("2024-06-25"), "unknown", IDX, "alt") is None
    assert day0(D("2024-06-25"), "unknown", IDX, "same") == D("2024-06-25")
    assert day0(D("2024-06-25"), "unknown", IDX, "next") == D("2024-06-26")


def test_past_end_of_data_and_bad_rule():
    assert day0(D("2024-07-12"), "after_close", IDX) is None
    with pytest.raises(ValueError):
        day0(D("2024-06-25"), "before_open", IDX, "bogus")
