"""election_cycle: the shipped average-path series
(package data) and the tab renderer. Offline — the live
current-cycle fetch is not exercised here (no network in
tests)."""

import pytest

from toolbox.DataZoo.market.election_cycle import (
    election_series,
    render_election_cycle,
)

KEYS = ["us", "eu", "jp", "hk"]
COLS = ["day"] + [f"{k}_{s}" for k in KEYS
                  for s in ("mean", "q25", "q75")]


def test_series_loads_with_1009_rows():
    df = election_series()
    assert list(df.columns) == COLS
    assert len(df) == 1009
    assert df["day"].min() == -504 and df["day"].max() == 504
    assert df["day"].is_unique


def test_day_zero_row_is_all_100():
    row = election_series().set_index("day").loc[0]
    assert row.tolist() == pytest.approx([100.0] * 12, abs=1e-9)


def test_endpoints_match_the_study():
    df = election_series().set_index("day")
    # +24m-after mean endpoints: US +19.4%, Europe +14.3%,
    # Japan +18.7%, Hong Kong +18.1%.
    for key, want in [("us", 119.4), ("eu", 114.3),
                      ("jp", 118.7), ("hk", 118.1)]:
        assert df.loc[504, f"{key}_mean"] == pytest.approx(
            want, abs=0.2)


def test_render_without_current_has_no_overlay():
    out = render_election_cycle(election_series(), {})
    assert "S&amp;P 500, 37 elections" in out
    assert "Hang Seng, 17 elections" in out
    assert "2026 (into today's vote)" not in out
    assert "+19.4%" in out and "31 of 37" in out
    assert "-2.9%" in out  # Europe 6m-before


def test_render_with_current_draws_overlay():
    cur = {"us": [98.0, 99.0, 100.0], "us_start": -2}
    out = render_election_cycle(election_series(), cur)
    assert "2026 (into today's vote)" in out
