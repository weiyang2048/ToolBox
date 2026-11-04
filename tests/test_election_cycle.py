"""election_cycle: the shipped average-path series
(package data) and the tab renderer. Offline — the live
current-cycle fetch is not exercised here (no network in
tests)."""

import math
import re

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
    out = render_election_cycle(election_series(), {})
    starts = [out.index(title) for title, _, _ in PANELS]
    for i, (_, values, positive) in enumerate(PANELS):
        end = starts[i + 1] if i + 1 < len(PANELS) else None
        seg = out[starts[i]:end]
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


def _plate(svg):
    m = re.search(r'<rect x="(-?[\d.]+)" y="(-?[\d.]+)" '
                  r'width="(-?[\d.]+)" height="20" rx="4" '
                  r'fill="#171512"', svg)
    assert m, "overlay label plate missing"
    x, y, w = (float(m.group(i)) for i in (1, 2, 3))
    return x, y, x + w, y + 20.0


def _path_points(svg, marker):
    m = re.search(r'<path d="([^"]+)"[^>]*' + marker, svg)
    assert m, f"path with {marker} missing"
    return [(float(a), float(b)) for a, b in
            re.findall(r"[ML] (-?[\d.]+) (-?[\d.]+)",
                       m.group(1))]


def test_label_plate_clears_overlay_and_band():
    cur = {"us": _live_like_overlay(103.7), "us_start": -252,
           "xus": _live_like_overlay(106.0), "xus_start": -252}
    out = render_election_cycle(election_series(), cur)
    blocks = re.findall(r"<svg.*?</svg>", out, re.S)
    assert len(blocks) == 4
    for svg in blocks[:2]:  # the two midterm panels
        px0, py0, px1, py1 = _plate(svg)
        # Right-aligned at day 0 (x=527): the plate's
        # right edge is the text's end (521) + 8px pad.
        assert abs(px1 - 529.0) < 1.5
        line = _path_points(svg, 'stroke="#7FB3D5"')
        # The overlay runs the full window into day 0.
        assert len(line) == 253
        assert line[-1][0] >= 526.0
        band = _path_points(svg, 'fill-opacity="0.16"')
        for pts in (line, band):
            for x, y in pts:
                if px0 - 0.5 <= x <= px1 + 0.5:
                    # Nothing the plate spans passes under
                    # it: every such point sits at least
                    # 3px below the plate's bottom edge.
                    assert y >= py1 + 3.0


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


def test_render_live_label_once_past_the_vote():
    cur = {"us": [98.0, 99.0, 100.0, 101.0, 102.0, 103.0],
           "us_start": -2,
           "xus": [97.0, 99.0, 100.0, 100.5, 101.0, 101.5],
           "xus_start": -2}
    out = render_election_cycle(election_series(), cur)
    head, tail = out.split(">Presidential elections<", 1)
    assert head.count("2026 (live)") == 2
    assert LABEL not in head
    assert "2026 (live)" not in tail
