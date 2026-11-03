"""midterm: the shipped average-path series (package data)
and the tab renderer. Offline — the live current-cycle
fetch is not exercised here (no network in tests)."""

import pytest

from toolbox.DataZoo.market.midterm import (
    midterm_series,
    render_midterms,
)

COLS = ["day", "us_mean", "us_q25", "us_q75",
        "xus_mean", "xus_q25", "xus_q75"]


def test_series_loads_with_505_rows():
    df = midterm_series()
    assert list(df.columns) == COLS
    assert len(df) == 505
    assert df["day"].min() == -252 and df["day"].max() == 252
    assert df["day"].is_unique


def test_day_zero_row_is_all_100():
    row = midterm_series().set_index("day").loc[0]
    assert row.tolist() == pytest.approx([100.0] * 6, abs=1e-9)


def test_endpoints_match_the_study():
    df = midterm_series().set_index("day")
    # +14.6% (US) and +10.8% (ex-US composite) mean 12m-after.
    assert df.loc[252, "us_mean"] == pytest.approx(114.6, abs=0.2)
    assert df.loc[252, "xus_mean"] == pytest.approx(110.8, abs=0.2)


def test_render_without_current_has_no_overlay():
    out = render_midterms(midterm_series(), {})
    assert "S&amp;P 500, 19 midterms" in out
    assert "five-market composite, 15 midterms" in out
    assert "2026 (into today's vote)" not in out
    assert "+14.6%" in out and "19 of 19" in out
    assert "-6.1%" in out  # composite 6m-before, from the CSV


def test_render_with_current_draws_overlay():
    cur = {"us": [98.0, 99.0, 100.0], "us_start": -2}
    out = render_midterms(midterm_series(), cur)
    assert "2026 (into today's vote)" in out
