"""opening_hours parser. The rule under test: never answer "open" on a guess."""

from __future__ import annotations

from datetime import datetime

from seruen.core.hours import CLOSED, OPEN, UNKNOWN, hours_status

# 2026-10-19 is a Monday
MONDAY_10 = datetime(2026, 10, 19, 10, 0)
MONDAY_20 = datetime(2026, 10, 19, 20, 0)
MONDAY_01 = datetime(2026, 10, 19, 1, 0)
SATURDAY_11 = datetime(2026, 10, 24, 11, 0)
SATURDAY_20 = datetime(2026, 10, 24, 20, 0)
SUNDAY_12 = datetime(2026, 10, 25, 12, 0)


def test_24_7_is_always_open():
    assert hours_status("24/7", MONDAY_01).status == OPEN
    assert hours_status("24/7", SATURDAY_20).status == OPEN


def test_weekday_range():
    result = hours_status("Mo-Fr 09:00-18:00", MONDAY_10)
    assert result.status == OPEN
    assert result.confidence == "high"
    assert hours_status("Mo-Fr 09:00-18:00", MONDAY_20).status == CLOSED
    assert hours_status("Mo-Fr 09:00-18:00", SUNDAY_12).status == CLOSED


def test_two_rules():
    raw = "Mo-Fr 09:00-18:00; Sa 10:00-16:00"
    assert hours_status(raw, MONDAY_10).status == OPEN
    assert hours_status(raw, SATURDAY_11).status == OPEN
    assert hours_status(raw, SATURDAY_20).status == CLOSED
    assert hours_status(raw, SUNDAY_12).status == CLOSED


def test_every_day():
    raw = "Mo-Su 10:00-22:00"
    assert hours_status(raw, MONDAY_10).status == OPEN
    assert hours_status(raw, SUNDAY_12).status == OPEN
    assert hours_status(raw, MONDAY_01).status == CLOSED


def test_crosses_midnight():
    raw = "Mo-Su 20:00-02:00"
    assert hours_status(raw, MONDAY_20).status == OPEN
    # 01:00 belongs to the range that opened the previous evening
    assert hours_status(raw, MONDAY_01).status == OPEN
    assert hours_status(raw, MONDAY_10).status == CLOSED


def test_multiple_intervals_one_day():
    raw = "Mo-Fr 09:00-12:00,13:00-18:00"
    assert hours_status(raw, MONDAY_10).status == OPEN
    assert hours_status(raw, datetime(2026, 10, 19, 12, 30)).status == CLOSED
    assert hours_status(raw, datetime(2026, 10, 19, 14, 0)).status == OPEN


def test_day_list_and_explicit_off():
    assert hours_status("Mo,We,Fr 10:00-19:00", MONDAY_10).status == OPEN
    assert hours_status("Mo,We,Fr 10:00-19:00", datetime(2026, 10, 20, 10, 0)).status == CLOSED
    assert hours_status("Mo-Su 10:00-22:00; Su off", SUNDAY_12).status == CLOSED


def test_holiday_rule_is_ignored_not_fatal():
    assert hours_status("Mo-Su 10:00-22:00; PH off", MONDAY_10).status == OPEN


def test_holiday_selector_inside_a_day_list():
    # "Mo-Su,PH 10:00-23:00" is very common in the Astana data
    assert hours_status("Mo-Su,PH 10:00-23:00", MONDAY_10).status == OPEN
    assert hours_status("Mo-Su,PH 10:00-23:00", MONDAY_01).status == CLOSED
    # a holidays-only tag says nothing about a Monday, so the answer is "unknown":
    # claiming "closed" would wrongly drop the place from the results
    assert hours_status("PH 10:00-23:00", MONDAY_10).status == UNKNOWN


def test_comma_separates_whole_rules_too():
    raw = "Mo-Th 11:00-23:00, Fr,Sa 12:00-24:00, Su 11:00-23:00"
    assert hours_status(raw, MONDAY_10).status == CLOSED  # opens at 11:00
    assert hours_status(raw, datetime(2026, 10, 19, 12, 0)).status == OPEN
    assert hours_status(raw, SATURDAY_11).status == CLOSED  # Sa opens at 12:00
    assert hours_status(raw, SATURDAY_20).status == OPEN
    assert hours_status(raw, SUNDAY_12).status == OPEN


def test_space_after_day_list_comma():
    assert hours_status("Su, Tu-Fr 11:00-19:00", datetime(2026, 10, 20, 12, 0)).status == OPEN
    assert hours_status("Su, Tu-Fr 11:00-19:00", MONDAY_10).status == CLOSED


def test_open_ended_range_stays_unknown():
    # "08:00+" has no closing time — we must not guess one
    assert hours_status("Mo-Su,PH 08:00+", MONDAY_10).status == UNKNOWN


def test_missing_and_unparsable_never_say_open():
    for raw in (None, "", "   ", "sunrise-sunset", "Mo-Fr 09:00+", "круглосуточно", "by appointment"):
        result = hours_status(raw, MONDAY_10)
        assert result.status == UNKNOWN, raw
        assert result.confidence == "low"


def test_minutes_until_close_is_reported():
    result = hours_status("Mo-Fr 09:00-18:00", MONDAY_10)
    assert result.minutes_until_close == 8 * 60
