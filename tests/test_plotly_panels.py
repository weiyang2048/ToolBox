"""The shared warped-axis panel builder (plotly_panels):
the warp geometry and the window-adaptive tick rule the
Markets tabs share (user directions 2026-11-10)."""
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
