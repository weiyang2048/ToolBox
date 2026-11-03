"""election_cycle: the shipped average-path series
(package data) and the tab renderer. Offline — the live
current-cycle fetch is not exercised here (no network in
tests)."""

import pytest

from toolbox.DataZoo.market.election_cycle import (
    election_series,
    render_election_cycle,
)

KEYS = ["mid_us", "mid_xus", "pres_us", "pres_xus"]
COLS = ["day"] + [f"{k}_{s}" for k in KEYS
                  for s in ("mean", "q25", "q75")]
LABEL = "2026 (into today's vote)"


def test_series_loads_with_505_rows():
    df = election_series()
    assert list(df.columns) == COLS
    assert len(df) == 505
    assert df["day"].min() == -252 and df["day"].max() == 252
    assert df["day"].is_unique


def test_day_zero_row_is_all_100():
    row = election_series().set_index("day").loc[0]
    assert row.tolist() == pytest.approx([100.0] * 12, abs=1e-9)


def test_endpoints_match_the_study():
    df = election_series().set_index("day")
    # +12m-after mean endpoints: midterm US +14.6%,
    # midterm ex-US +10.8%, presidential US +8.1%,
    # presidential ex-US +13.4%.
    for key, want in [("mid_us", 114.6), ("mid_xus", 110.8),
                      ("pres_us", 108.1), ("pres_xus", 113.4)]:
        assert df.loc[252, f"{key}_mean"] == pytest.approx(
            want, abs=0.2)


def test_render_without_current_has_no_overlay():
    out = render_election_cycle(election_series(), {})
    assert ">Midterm elections<" in out
    assert ">Presidential elections<" in out
    assert "S&amp;P 500, 19 midterms" in out
    assert "S&amp;P 500, 20 elections" in out
    assert LABEL not in out
    assert "+14.6%" in out and "19 of 19" in out
    assert "-6.1%" in out  # midterm ex-US 6m-before
    assert "+13.4%" in out and "13 of 15" in out


def test_render_with_current_overlays_midterms_only():
    cur = {"us": [98.0, 99.0, 100.0], "us_start": -2,
           "xus": [97.0, 99.0, 100.0], "xus_start": -2}
    out = render_election_cycle(election_series(), cur)
    head, tail = out.split(">Presidential elections<", 1)
    # Both midterm panels draw the labelled overlay; the
    # presidential section draws none.
    assert head.count(LABEL) == 2
    assert LABEL not in tail
