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
    assert out["asof"].endswith(" CT")


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
