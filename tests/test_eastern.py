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


def test_utc_index_to_et_dates_moves_midnight_utc_back_a_day():
    # A bar stamped 2026-11-02 00:00 UTC starts 2026-11-01 19:00
    # EST, so its Eastern date is 2026-11-01 — in both DST regimes.
    idx = pd.DatetimeIndex(["2026-11-02", "2026-07-15"])
    out = eastern.utc_index_to_et_dates(idx)
    assert [str(d.date()) for d in out] == ["2026-11-01",
                                            "2026-07-14"]
