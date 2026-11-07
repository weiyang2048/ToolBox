"""Tests for toolbox.DataZoo.market.eastern — the app's one
Eastern-time convention (user direction 2026-11-01)."""
import datetime

import pandas as pd

from toolbox.DataZoo.market import eastern


def test_asof_stamp_renders_est_after_dst_ends():
    # 2026-11-01 19:36 UTC is 2:36 PM in New York, on standard time.
    when = datetime.datetime(2026, 11, 1, 19, 36,
                             tzinfo=datetime.timezone.utc)
    assert eastern.asof_stamp(when) == "Sun Nov 1, 2:36 PM EST"


def test_asof_stamp_renders_edt_in_summer():
    when = datetime.datetime(2026, 7, 15, 16, 5,
                             tzinfo=datetime.timezone.utc)
    assert eastern.asof_stamp(when) == "Wed Jul 15, 12:05 PM EDT"


def test_asof_stamp_naive_is_eastern_wall_time():
    when = datetime.datetime(2026, 10, 29, 12, 0)
    assert eastern.asof_stamp(when) == "Thu Oct 29, 12:00 PM EDT"


def test_format_day_unpadded_without_platform_flags():
    # Hand-composed (2026-11-07): "%-d" raises ValueError
    # under Windows strftime; the output must match the
    # glibc rendering exactly, single- and double-digit.
    assert eastern.format_day(
        datetime.date(2026, 11, 2)) == "Mon Nov 2, 2026"
    assert eastern.format_day(
        datetime.date(2026, 11, 15)) == "Sun Nov 15, 2026"
    assert eastern.format_month_day(
        datetime.date(2026, 11, 2)) == "Nov 2, 2026"


def test_asof_stamp_midnight_and_noon_hours():
    # The hand-composed 12-hour clock: 0 -> 12 AM, 12 -> 12 PM.
    base = datetime.datetime(2026, 11, 2, tzinfo=eastern.ET)
    assert eastern.asof_stamp(base.replace(hour=0, minute=5)) \
        == "Mon Nov 2, 12:05 AM EST"
    assert eastern.asof_stamp(base.replace(hour=12, minute=5)) \
        == "Mon Nov 2, 12:05 PM EST"


def test_utc_index_to_et_dates_moves_midnight_utc_back_a_day():
    # A bar stamped 2026-11-02 00:00 UTC starts 2026-11-01 19:00
    # EST, so its Eastern date is 2026-11-01 — in both DST regimes.
    idx = pd.DatetimeIndex(["2026-11-02", "2026-07-15"])
    out = eastern.utc_index_to_et_dates(idx)
    assert [str(d.date()) for d in out] == ["2026-11-01",
                                            "2026-07-14"]
