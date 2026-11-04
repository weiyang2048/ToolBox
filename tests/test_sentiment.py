"""Tests for toolbox.DataZoo.market.sentiment (network-free)."""
import pandas as pd

from toolbox.DataZoo.market import sentiment as sm


def test_fetch_sentiment_progress_callback(monkeypatch):
    # The progress hook (the dashboard's sentiment bar): one
    # call before the sources, then one per source attempt,
    # labelled, in source order.
    import numpy as np

    idx = pd.date_range("2026-01-01", periods=40, freq="D")
    series = pd.Series(np.linspace(30.0, 50.0, 40), index=idx)
    for fn in ("fetch_cnn", "fetch_crypto", "fetch_gold"):
        monkeypatch.setattr(sm, fn, lambda s=series: s)
    calls = []
    out = sm.fetch_sentiment(
        progress=lambda d, t, s: calls.append((d, t, s)))
    assert out is not None
    assert calls == [(0, 3, ""), (1, 3, "CNN Fear & Greed"),
                     (2, 3, "Crypto Fear & Greed"), (3, 3, "Gold")]


def test_fetch_cnn_builds_on_canonical_graph(monkeypatch):
    import datetime

    now_ms = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
    payload = {"fear_and_greed_historical": {"data": [
        {"x": now_ms - 86_400_000, "y": 41.6},
        {"x": now_ms, "y": 42.4},
    ]}}
    monkeypatch.setattr(sm, "fetch_cnn_graph", lambda timeout=30: payload)
    out = sm.fetch_cnn()
    assert out.tolist() == [42, 42]  # int-rounded, as before


def test_fetch_crypto_builds_on_canonical_entries(monkeypatch):
    entries = [
        {"timestamp": "1767225600", "value": "30",
         "value_classification": "Fear"},
        {"timestamp": "1767312000", "value": "34",
         "value_classification": "Fear"},
    ]
    monkeypatch.setattr(
        sm, "fetch_crypto_entries", lambda limit, timeout=30: entries)
    out = sm.fetch_crypto()
    assert out.tolist() == [30, 34]


def test_fetch_crypto_dates_values_by_index_day(monkeypatch):
    # alternative.me stamps at 00:00 UTC of the index day — a
    # day-key: the 2026-11-02 stamp IS the Nov 2 reading, as the
    # source itself presents it. (Dating it in Eastern, as an
    # instant, labelled the current reading as yesterday's —
    # reversed 2026-11-02 after the user caught it.)
    entries = [
        {"timestamp": "1793491200", "value": "34",
         "value_classification": "Fear"},
        {"timestamp": "1793577600", "value": "37",
         "value_classification": "Fear"},
    ]
    monkeypatch.setattr(
        sm, "fetch_crypto_entries", lambda limit, timeout=30: entries)
    out = sm.fetch_crypto()
    assert [str(d.date()) for d in out.index] == ["2026-11-01",
                                                  "2026-11-02"]
    assert out.iloc[-1] == 37


def test_fetch_gold_dates_bars_in_eastern(monkeypatch):
    # Yahoo stamps GC=F daily bars at midnight ET (04:00 UTC). Dated in
    # Eastern, the Friday bar stays Friday; a Chicago conversion would
    # push it back to Thursday 11 PM and mislabel it. fetch_gold rides
    # the canonical yahoo_daily since 2026-11-02; its UTC-normalized
    # dates are the same exchange dates for midnight-ET stamps.
    import datetime

    from toolbox.DataZoo.market import yahoo_prices as yp

    fri_et = datetime.datetime(2026, 10, 30, 4, tzinfo=datetime.timezone.utc)
    thu_et = fri_et - datetime.timedelta(days=1)
    payload = {"chart": {"result": [{
        "timestamp": [int(thu_et.timestamp()), int(fri_et.timestamp())],
        "indicators": {"quote": [{"close": [4031.4, 3898.9]}],
                       "adjclose": [{"adjclose": [4031.4, 3898.9]}]},
    }]}}
    monkeypatch.setattr(yp, "chart_payload",
                        lambda t, a, b: payload)
    out = sm.fetch_gold()
    assert [str(d.date()) for d in out.index] == ["2026-10-29", "2026-10-30"]
    assert out.iloc[-1] == 3898.9


def test_zone_boundaries():
    assert sm.zone(0) == "Extreme Fear"
    assert sm.zone(25) == "Extreme Fear"
    assert sm.zone(26) == "Fear"
    assert sm.zone(45) == "Fear"
    assert sm.zone(46) == "Neutral"
    assert sm.zone(55) == "Neutral"
    assert sm.zone(56) == "Greed"
    assert sm.zone(75) == "Greed"
    assert sm.zone(76) == "Extreme Greed"
    assert sm.zone(100) == "Extreme Greed"


def _series(vals):
    idx = pd.date_range("2026-01-01", periods=len(vals), freq="D")
    return pd.Series(vals, index=idx, dtype=float)


def test_crypto_evening_index_is_dated_in_eastern(monkeypatch):
    # The index for tomorrow's UTC day is published at 19:00
    # Eastern tonight; on an Eastern evening it is tonight's
    # reading and is dated tonight (user direction 2026-11-02:
    # "let's use EST") — never a day ahead, never (the morning
    # bug) a day behind its source day.
    import datetime

    from toolbox.DataZoo.market.eastern import ET

    today_et = datetime.datetime.now(ET).date()
    tomorrow = today_et + datetime.timedelta(days=1)
    stamp = int(datetime.datetime(
        tomorrow.year, tomorrow.month, tomorrow.day,
        tzinfo=datetime.timezone.utc).timestamp())
    rows = [{"value": "40", "timestamp": str(stamp - 86400)},
            {"value": "26", "timestamp": str(stamp)}]
    monkeypatch.setattr(sm, "fetch_crypto_entries",
                        lambda limit, timeout=30: rows)
    out = sm.fetch_crypto()
    assert out.index[-1].date() == today_et
    assert out.iloc[-1] == 26


def test_cap_to_eastern_today_merges_the_future_point():
    # The cap itself (fetch_gold's live bar rides it): a final
    # point dated tomorrow moves to today's Eastern date and
    # wins over today's own earlier point; history untouched.
    import datetime

    import pandas as pd

    from toolbox.DataZoo.market.eastern import ET

    today_et = datetime.datetime.now(ET).date()
    tomorrow = today_et + datetime.timedelta(days=1)
    s = pd.Series([3900.0, 3956.0],
                  index=pd.to_datetime(
                      [today_et.isoformat(), tomorrow.isoformat()]))
    out = sm._cap_to_eastern_today(s)
    assert out.index[-1].date() == today_et
    assert out.iloc[-1] == 3956.0
    assert len(out) == 1
    # A series already within Eastern today is returned as-is.
    assert sm._cap_to_eastern_today(s.iloc[:1]).iloc[-1] == 3900.0


def test_fetch_sentiment_degrades_per_source(monkeypatch):
    monkeypatch.setattr(sm, "fetch_cnn", lambda: _series([40] * 40))
    monkeypatch.setattr(sm, "fetch_crypto",
                        lambda: (_ for _ in ()).throw(RuntimeError("x")))
    monkeypatch.setattr(sm, "fetch_gold", lambda: _series([3900] * 40))
    out = sm.fetch_sentiment()
    assert set(out["series"]) == {"cnn", "gold"}
    assert out["latest"]["cnn"] == {"value": 40.0, "zone": "Fear"}
    assert out["latest"]["gold"] == {"value": 3900.0}
    assert "crypto" not in out["latest"]
    # No fetch-time stamp: the tab dates each series by its
    # own last measurement (user direction 2026-11-02).
    assert "asof" not in out


def test_fetch_sentiment_none_when_all_fail(monkeypatch):
    for fn in ("fetch_cnn", "fetch_crypto", "fetch_gold"):
        monkeypatch.setattr(sm, fn,
                            lambda: (_ for _ in ()).throw(RuntimeError()))
    assert sm.fetch_sentiment() is None


def test_chart_png_renders():
    series = {"cnn": _series([20 + i % 50 for i in range(120)]),
              "crypto": _series([30 + i % 40 for i in range(120)]),
              "gold": _series([3000 + i * 5 for i in range(120)])}
    png = sm.chart_png(series)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png) > 20000


def _capture(monkeypatch):
    import matplotlib.pyplot as plt

    captured = {}
    orig = plt.subplots

    def fake(*a, **k):
        fig, ax = orig(*a, **k)
        captured["fig"] = fig
        return fig, ax

    monkeypatch.setattr(plt, "subplots", fake)
    return captured


def test_chart_png_headroom_above_100(monkeypatch):
    # Live fix 2026-11-02: the index axis runs to 115 (headroom
    # over the 0-100 scale; bands stay 0-100) and gold sits on
    # its own 0-100 axis with its peak normalized to GOLD_PEAK,
    # below the band ceiling — normalizing the peak to exactly
    # 100 put it on the ceiling / top tick and the spikes read
    # as truncated on the served page.
    captured = _capture(monkeypatch)
    series = {"cnn": _series([20 + i % 50 for i in range(120)]),
              "crypto": _series([30 + i % 40 for i in range(120)]),
              "gold": _series([3000 + i * 5 for i in range(120)])}
    sm.chart_png(series)
    axes = captured["fig"].axes
    assert axes[0].get_ylim() == (0.0, 115.0)
    assert axes[1].get_ylim() == (0.0, 100.0)
    peak = max(axes[1].lines[0].get_ydata())
    assert abs(peak - sm.GOLD_PEAK) < 1e-6


def test_chart_png_gold_axis_top_headroom(monkeypatch):
    # The gold axis top (in dollars, via the plotted mapping)
    # must exceed the series max by ~15% of the series range.
    captured = _capture(monkeypatch)
    gold = _series([3000.0 + i * 20 for i in range(101)])  # ..5000
    sm.chart_png({"gold": gold})
    ax2 = captured["fig"].axes[-1]
    top_norm = ax2.get_ylim()[1]
    peak_norm = max(ax2.lines[0].get_ydata())
    gmin, gmax = float(gold.min()), float(gold.max())
    axis_top_usd = gmin + (gmax - gmin) * top_norm / peak_norm
    headroom = (axis_top_usd - gmax) / (gmax - gmin)
    assert axis_top_usd > gmax + 0.15 * (gmax - gmin)
    assert 0.14 < headroom < 0.20
    assert ax2.get_yticklabels()[-1].get_text() == "$5,000"


def test_spread_labels():
    # Converged desireds are pushed apart, order preserved.
    out = sm._spread_labels([26.0, 29.0])
    assert out[1] - out[0] >= sm.LABEL_MIN_SEP
    assert out[0] == 26.0
    # Well-spread desireds pass through untouched.
    assert sm._spread_labels([23.0, 63.0]) == [23.0, 63.0]
    # A stack at the top clamps into the band, still separated.
    out = sm._spread_labels([110.0, 111.0])
    assert out[1] == sm.LABEL_HI
    assert out[1] - out[0] >= sm.LABEL_MIN_SEP


def test_chart_png_endpoint_labels_separate_when_converged(monkeypatch):
    # Live defect 2026-11-02: crypto 26 vs CNN 23 at identical
    # (+8, +8) offsets drew on top of each other and past the
    # right spine. Labels now sit inside at spread positions.
    captured = _capture(monkeypatch)
    cnn = _series([40.0] * 119 + [23.0])
    crypto = _series([40.0] * 119 + [26.0])
    sm.chart_png({"cnn": cnn, "crypto": crypto})
    ax = captured["fig"].axes[0]
    labels = {t.get_text(): t.get_position()[1]
              for t in ax.texts if "·" in t.get_text()}
    y23 = next(y for txt, y in labels.items() if txt.startswith("23 ·"))
    y26 = next(y for txt, y in labels.items() if txt.startswith("26 ·"))
    assert y26 > y23
    assert y26 - y23 >= sm.LABEL_MIN_SEP
    assert sm.LABEL_LO <= y23
    assert y26 <= sm.LABEL_HI


def test_chart_png_endpoint_labels_natural_when_spread(monkeypatch):
    captured = _capture(monkeypatch)
    cnn = _series([40.0] * 119 + [60.0])
    crypto = _series([40.0] * 119 + [20.0])
    sm.chart_png({"cnn": cnn, "crypto": crypto})
    ax = captured["fig"].axes[0]
    labels = {t.get_text(): t.get_position()[1]
              for t in ax.texts if "·" in t.get_text()}
    y60 = next(y for txt, y in labels.items() if txt.startswith("60 ·"))
    y20 = next(y for txt, y in labels.items() if txt.startswith("20 ·"))
    assert y60 == 63.0  # value + 3, unadjusted
    assert y20 == 23.0
