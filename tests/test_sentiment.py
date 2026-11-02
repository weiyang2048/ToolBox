"""Tests for toolbox.DataZoo.market.sentiment (network-free)."""
import pandas as pd

from toolbox.DataZoo.market import sentiment as sm


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
    # push it back to Thursday 11 PM and mislabel it.
    import datetime

    fri_et = datetime.datetime(2026, 10, 30, 4, tzinfo=datetime.timezone.utc)
    thu_et = fri_et - datetime.timedelta(days=1)
    payload = {"chart": {"result": [{
        "timestamp": [int(thu_et.timestamp()), int(fri_et.timestamp())],
        "indicators": {"quote": [{"close": [4031.4, 3898.9]}]},
    }]}}
    monkeypatch.setattr(sm, "_get_json", lambda url, headers=None: payload)
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
    assert out["asof"].endswith((" EST", " EDT"))


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


def test_chart_png_headroom_above_100(monkeypatch):
    # Live fix 2026-11-02: a series max at 100 (gold normalizes
    # its peak to exactly 100) clipped at the plot's top edge.
    # Both axes run to 115; the zone bands stay 0-100.
    import matplotlib.pyplot as plt

    captured = {}
    orig = plt.subplots

    def fake(*a, **k):
        fig, ax = orig(*a, **k)
        captured["fig"] = fig
        return fig, ax

    monkeypatch.setattr(plt, "subplots", fake)
    series = {"cnn": _series([20 + i % 50 for i in range(120)]),
              "crypto": _series([30 + i % 40 for i in range(120)]),
              "gold": _series([3000 + i * 5 for i in range(120)])}
    sm.chart_png(series)
    axes = captured["fig"].axes
    assert axes[0].get_ylim() == (0.0, 115.0)
    assert axes[1].get_ylim() == (0.0, 115.0)
