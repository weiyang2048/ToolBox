"""election_cycle: the shipped average-path series
(package data) and the tab renderer. Offline — the live
current-cycle fetch is not exercised here (no network in
tests)."""

import math

import pytest

from toolbox.DataZoo.market.election_cycle import (
    chart_figs,
    election_series,
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


def _html(blocks):
    return "".join(v for k, v in blocks if k == "html")


def _figs(blocks):
    return [v for k, v in blocks if k == "fig"]


def _ann_texts(fig):
    return [a.text or "" for a in fig.layout.annotations]


def test_figs_without_current_have_no_overlay():
    blocks = chart_figs(election_series(), {})
    out = _html(blocks)
    assert ">Midterm elections<" in out
    assert ">Presidential elections<" in out
    figs = _figs(blocks)
    assert len(figs) == 4
    titles = [t for f in figs for t in _ann_texts(f)]
    assert any("S&P 500, 19 midterms" in t for t in titles)
    assert any("S&P 500, 20 elections" in t for t in titles)
    for f in figs:
        assert all(t.name != "2026" for t in f.data)
        assert not any(LABEL in t for t in _ann_texts(f))
    assert "+14.6%" in out and "19 of 19" in out
    assert "-6.1%" in out  # midterm ex-US 6m-before
    assert "+13.4%" in out and "13 of 15" in out


def test_figs_with_current_overlay_midterms_only():
    cur = {"us": [98.0, 99.0, 100.0], "us_start": -2,
           "xus": [97.0, 99.0, 100.0], "xus_start": -2}
    figs = _figs(chart_figs(election_series(), cur))
    # Both midterm panels draw the labelled overlay; the
    # presidential panels draw none.
    for f in figs[:2]:
        assert any(t.name == "2026" for t in f.data)
        assert any(LABEL in t for t in _ann_texts(f))
    for f in figs[2:]:
        assert all(t.name != "2026" for t in f.data)
        assert not any(LABEL in t for t in _ann_texts(f))


# Per panel, in tab order: (title fragment, the four chip
# values in chronological chip order — 12m before, 6m
# before, 6m after, 12m after — and the positive count,
# which both "after" chips carry).
PANELS = [
    ("19 midterms, 1950–2022",
     ["+4.0%", "+1.3%", "+13.4%", "+14.6%"], "19 of 19"),
    ("15 midterms, 1966–2022",
     ["+1.1%", "-6.1%", "+11.7%", "+10.8%"], "13 of 15"),
    ("20 elections, 1948–2024",
     ["+9.5%", "+4.6%", "+4.0%", "+8.1%"], "13 of 20"),
    ("15 elections, 1968–2024",
     ["+13.0%", "+3.6%", "+9.8%", "+13.4%"], "13 of 15"),
]
CHIP_LABELS = ("12m before", "6m before",
               "6m after", "12m after")


def test_chips_four_per_panel_chronological():
    blocks = chart_figs(election_series(), {})
    chips = [v for k, v in blocks
             if k == "html" and "12m before" in v]
    assert len(chips) == 4
    for seg, (_, values, positive) in zip(chips, PANELS):
        order = [seg.index(s) for s in CHIP_LABELS]
        assert order == sorted(order)
        for v in values:
            assert v in seg
        assert seg.count(positive) == 2


def _live_like_overlay(hot):
    """A 2026-like overlay: 253 closes over days
    -252..0, wavy through the year with a hot final ~90
    days — the regime whose tail (97-104) the old label
    plate hid — tapering to the last close at 100."""
    vals = [92 + 5 * math.sin(i / 29.0) + i * 0.015
            for i in range(253)]
    for i in range(163, 249):
        vals[i] = hot
    for i in range(249, 253):
        vals[i] = hot - (hot - 100.0) * (i - 248) / 4
    vals[-1] = 100.0
    return vals


def test_label_plate_clears_overlay_and_band():
    cur = {"us": _live_like_overlay(103.7), "us_start": -252,
           "xus": _live_like_overlay(106.0), "xus_start": -252}
    df = election_series()
    figs = _figs(chart_figs(df, cur))
    days = df["day"].tolist()
    for f, key in zip(figs[:2], ("mid_us", "mid_xus")):
        ann = next(a for a in f.layout.annotations
                   if a.text == LABEL)
        # The plate is right-anchored at the overlay's
        # end (day 0 here) and its base sits at the
        # ceiling over its span: nothing it spans — the
        # overlay or the band's top over the last 60
        # days — passes above it.
        assert ann.x == 0 and ann.xanchor == "right"
        q75 = df[key + "_q75"].tolist()
        span_top = max(
            cur["us" if key == "mid_us" else "xus"][-61:]
            + [q75[days.index(d)] for d in range(-60, 1)])
        assert ann.y >= span_top - 1e-9
        # And the axis extends past the ceiling (the
        # 90% rule), so the plate stays inside the plot.
        lo, hi = f.layout.yaxis.range
        assert ann.y - lo <= 0.90 * (hi - lo) + 1e-9


# --- The anchored live overlay (2026-11-04): the current
# cycle is indexed at the election-day close and extends
# past day 0 as post-vote sessions complete.

import pandas as pd

from toolbox.DataZoo.market.election_cycle import (
    _anchor_cycle,
    _composite,
)


def _closes():
    # 90 sessions ending Fri Nov 6, 2026, so the election
    # (Tue Nov 3) sits 3 sessions before the end; give the
    # election session a distinctive close of 200.
    idx = pd.bdate_range(end="2026-11-06", periods=90)
    vals = [100.0 + i for i in range(len(idx))]
    pos = idx.get_loc(pd.Timestamp("2026-11-03"))
    vals[pos] = 200.0
    return pd.Series(vals, index=idx), pos


def test_anchor_cycle_indexes_at_the_election_close():
    closes, pos = _closes()
    start, vals = _anchor_cycle(closes)
    # Day 0 sits at position -start in the values and is
    # exactly 100 (indexed at the election close).
    assert vals[-start] == pytest.approx(100.0)
    assert start == -pos  # full run-up kept (pos < 252)
    # Sessions after the vote extend the path past 0.
    assert start + len(vals) - 1 == 3
    # Values are ratios to the election close of 200.
    assert vals[-1] == pytest.approx(
        closes.iloc[-1] / 200.0 * 100)


def test_anchor_cycle_none_when_the_series_is_stale():
    # 70 sessions ending more than two weeks before the
    # vote: the last session is no election-day anchor.
    idx = pd.bdate_range(end="2026-10-16", periods=70)
    closes = pd.Series([100.0] * len(idx), index=idx)
    assert _anchor_cycle(closes) is None


def test_anchor_cycle_uses_the_previous_session_when_closed():
    # A market closed on election day itself (no Nov 3
    # session) anchors to its Nov 2 session: day 0 is
    # that close, and the path ends at day 0.
    idx = pd.bdate_range(end="2026-11-02", periods=90)
    closes = pd.Series([100.0 + i for i in range(len(idx))],
                       index=idx)
    start, vals = _anchor_cycle(closes)
    assert vals[-start] == pytest.approx(100.0)
    assert start + len(vals) - 1 == 0


def test_composite_ends_at_the_second_latest_leg():
    a = (-2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])   # days -2..3
    b = (-2, [2.0, 2.0, 2.0, 2.0, 2.0])        # days -2..2
    start, comp = _composite([a, b])
    assert start == -2
    # Day +3 has only one leg: the composite stops at +2.
    assert len(comp) == 5
    assert comp[0] == pytest.approx(1.5)
    assert comp[-1] == pytest.approx(3.5)


def test_live_label_once_past_the_vote():
    cur = {"us": [98.0, 99.0, 100.0, 101.0, 102.0, 103.0],
           "us_start": -2,
           "xus": [97.0, 99.0, 100.0, 100.5, 101.0, 101.5],
           "xus_start": -2}
    figs = _figs(chart_figs(election_series(), cur))
    for f in figs[:2]:
        texts = _ann_texts(f)
        assert any("2026 (live)" in t for t in texts)
        assert not any(LABEL in t for t in texts)
    for f in figs[2:]:
        assert not any("2026 (live)" in t
                       for t in _ann_texts(f))
