"""The shared warped-axis panel builder (plotly_panels):
the warp geometry and the window-adaptive tick rule the
Markets tabs share (user directions 2026-11-10)."""
import numpy as np
import pandas as pd

from toolbox.DataZoo.market import plotly_panels as pp


def test_warp_tail_takes_its_share():
    dates = pd.bdate_range("2026-01-01", periods=120)
    idx, cut, warp, warped = pp.warp_setup(dates)
    assert warped
    xs = warp(idx)
    assert xs[0] == 0.0 and xs[-1] == 1.0
    assert abs(warp([cut])[0] - (1 - pp.TAIL_SHARE)) < 1e-9


def test_short_window_stays_linear():
    dates = pd.bdate_range("2026-10-20", periods=10)
    _idx, _cut, warp, warped = pp.warp_setup(dates)
    assert not warped
    axis, _w, _wd = pp.x_axis(dates)
    assert axis["tickvals"] == [0.0, 0.5, 1.0]


def test_month_ticks_on_a_one_year_window():
    dates = pd.bdate_range("2025-11-03", periods=252)
    axis, _w, _wd = pp.x_axis(dates)
    labels = axis["ticktext"]
    assert labels[-2] == (dates[-1] - pd.Timedelta(days=30)
                          ).strftime("%b %d")
    assert any(lbl.endswith(" 26") for lbl in labels[:-2])


def test_year_ticks_on_a_decade_window():
    # The demo finding: ~120 month labels crammed into a
    # 10y window's compressed head garble; long windows
    # tick Jan 1 years instead, seam and last date kept.
    dates = pd.bdate_range("2016-11-01", periods=2600)
    axis, _w, _wd = pp.x_axis(dates)
    labels = axis["ticktext"]
    years = [lbl for lbl in labels if len(lbl) == 4]
    assert years == [str(y) for y in range(2017, 2027)]
    assert labels[-1] == dates[-1].strftime("%b %d")
    assert len(labels) == len(years) + 2


# ---------------------------------------------------------------------------
# Known-value tests for warped_panel itself (idea search #2,
# candidate 3): the figure's data is asserted against the
# documented warp geometry recomputed here with plain timedelta
# arithmetic — structural golden values, not image bytes.
# ---------------------------------------------------------------------------


def _panel_kwargs(**over):
    kw = dict(height=321, y_ticks=[50.0, 100.0], y_range=[40.0, 120.0],
              fmt_y=lambda v: f"{v:.0f}",
              end_labels={"A": "A end", "B": "B end"})
    kw.update(over)
    return kw


def test_warped_panel_short_window_golden():
    # 10 business days spanning 11 calendar days: under the
    # 30-day tail the axis stays linear, so every x is the
    # calendar-day offset over the 11-day span (weekend days
    # count — the warp is on day counts, not sessions).
    dates = pd.bdate_range("2026-01-05", periods=10)
    a = [100.0 + i for i in range(10)]
    b = [50.0] * 10
    fig = pp.warped_panel(
        dates, {"A": a, "B": b},
        {"A": "#111111", "B": "#222222"},
        {"A": "Alpha Full", "B": "Beta Full"},
        {"A": "Alpha", "B": "Beta"},
        zero=True, **_panel_kwargs())
    d = fig.to_dict()
    span = (dates[-1] - dates[0]).days
    expect_x = [(t - dates[0]).days / span for t in dates]

    lines = [t for t in d["data"] if t.get("mode") == "lines"]
    assert [t["name"] for t in lines] == ["Alpha", "Beta"]
    for trace, vals in zip(lines, (a, b), strict=True):
        assert list(trace["y"]) == vals
        assert np.allclose(trace["x"], expect_x, atol=1e-12)
    assert lines[0]["customdata"][0] == "Jan 05, 2026"
    assert lines[0]["customdata"][-1] == "Jan 16, 2026"
    assert "Alpha Full" in lines[0]["hovertemplate"]

    layout = d["layout"]
    assert layout["height"] == 321
    assert layout["margin"] == {"l": 10, "r": 26, "t": 10, "b": 10}
    assert layout["xaxis"]["range"] == [0.0, 1.0]
    assert layout["xaxis"]["ticktext"] == [
        t.strftime("%b %d")
        for t in (dates[0], dates[0] + (dates[-1] - dates[0]) / 2,
                  dates[-1])]
    assert layout["yaxis"]["range"] == [40.0, 120.0]
    assert list(layout["yaxis"]["tickvals"]) == [50.0, 100.0]

    # zero=True draws exactly one shape: the dashed zero line.
    # No seam on an unwarped window.
    shapes = layout.get("shapes", [])
    assert len(shapes) == 1
    assert shapes[0]["y0"] == 0 and shapes[0]["line"]["dash"] == "dash"

    annotations = layout.get("annotations", [])
    chips = {an["text"]: an for an in annotations if an["x"] == 0.0}
    assert chips["50"]["y"] == 50.0 and chips["100"]["y"] == 100.0
    # Endpoint chips: labels sit at their own values when the
    # separation allows; the label x is the warp of 3 days
    # before the last date (8 of 11 days in).
    ends = {an["text"]: an for an in annotations
            if an["text"] in ("A end", "B end")}
    assert ends["A end"]["y"] == 109.0 and ends["B end"]["y"] == 50.0
    assert abs(ends["A end"]["x"] - 8 / 11) < 1e-12
    markers = [t for t in d["data"] if t.get("mode") == "markers"]
    assert sorted(t["y"][0] for t in markers) == [50.0, 109.0]
    assert all(list(t["x"]) == [1.0] for t in markers)


def test_warped_panel_warped_window_golden():
    # 61 daily points, span 60 days: [day 0, day 30] maps onto
    # [0, 0.75] and [day 30, day 60] onto [0.75, 1] — the
    # piecewise-linear day-count warp from the module docstring,
    # recomputed here point by point.
    dates = pd.date_range("2026-01-01", periods=61, freq="D")
    vals = [float(i) for i in range(61)]
    fig = pp.warped_panel(
        dates, {"A": vals}, {"A": "#111111"}, {"A": "Alpha Full"},
        {"A": "Alpha"}, height=200, y_ticks=[0.0, 60.0],
        y_range=[0.0, 60.0], fmt_y=lambda v: f"{v:.0f}",
        end_labels={"A": "end"})
    d = fig.to_dict()
    trace = d["data"][0]
    expect_x = [day / 30 * 0.75 if day <= 30
                else 0.75 + (day - 30) / 30 * 0.25
                for day in range(61)]
    assert np.allclose(trace["x"], expect_x, atol=1e-12)
    assert list(trace["y"]) == vals

    layout = d["layout"]
    assert list(layout["xaxis"]["tickvals"]) == [0.0, 0.75, 1.0]
    assert layout["xaxis"]["ticktext"] == [
        dates[0].strftime("%b %y"), dates[30].strftime("%b %d"),
        dates[-1].strftime("%b %d")]
    # The dotted seam sits at 1 - TAIL_SHARE with its note.
    shapes = layout.get("shapes", [])
    assert len(shapes) == 1
    assert shapes[0]["x0"] == 1 - pp.TAIL_SHARE
    assert shapes[0]["line"]["dash"] == "dot"
    notes = [an["text"] for an in layout.get("annotations", [])]
    assert "last 30 days \u2192" in notes
