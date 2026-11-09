"""Tests for toolbox.DataZoo.market (sentiment, indices, prices, housing).

Every network source is mocked — the tests are deterministic, offline,
and safe for CI. Asserted behaviors were probe-verified against the
real implementations first.
"""

import datetime

import numpy as np
import pandas as pd

from toolbox.DataZoo.market import CNN_SERIES_LABELS, FearGreed, StockIndices
from toolbox.DataZoo.market import fear_greed as fg_module
from toolbox.DataZoo.market import stock_indices as si_module
from toolbox.DataZoo.market import yahoo_prices as yp_module
from toolbox.DataZoo.market.yahoo_prices import (
    download_close_prices,
    normalize_close_prices,
    yahoo_daily,
)
from toolbox.DataZoo.market.zillow_data import ZillowZipData

UTC = datetime.timezone.utc


def _ts_ms(year, month, day, hour=12):
    dt = datetime.datetime(year, month, day, hour, tzinfo=UTC)
    return int(dt.timestamp() * 1000)


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _fail_get(*args, **kwargs):
    raise RuntimeError("network unavailable")


# --- package surface ------------------------------------------------------


def test_market_reexports_and_labels():
    assert CNN_SERIES_LABELS["fear_and_greed_historical"] == (
        "Fear & Greed Index (0-100)"
    )
    # The headline index is the first series, in display order.
    assert next(iter(CNN_SERIES_LABELS)) == "fear_and_greed_historical"
    assert len(CNN_SERIES_LABELS) == 10


# --- normalize_close_prices ----------------------------------------------


def test_normalize_scales_each_column_by_first_valid_price():
    prices = pd.DataFrame({"A": [np.nan, 2.0, 4.0], "B": [10.0, 20.0, 5.0]})
    out = normalize_close_prices(prices)
    assert np.isnan(out["A"].iloc[0])
    np.testing.assert_allclose(out["A"].iloc[1:], [1.0, 2.0])
    np.testing.assert_allclose(out["B"], [1.0, 2.0, 0.5])


def test_normalize_empty_frame_passes_through():
    assert normalize_close_prices(pd.DataFrame()).empty


def test_normalize_all_nan_column_stays_nan():
    out = normalize_close_prices(pd.DataFrame({"A": [np.nan, np.nan]}))
    assert out["A"].isna().all()


# --- download_close_prices (history mocked) -------------------------------


def _history_frame():
    idx = pd.date_range("2024-01-01", periods=2, tz="America/New_York")
    columns = pd.MultiIndex.from_tuples([("Close", "AAA"), ("Close", "BBB")])
    return pd.DataFrame([[100.0, 50.0], [110.0, 55.0]], index=idx, columns=columns)


def test_download_no_tickers_returns_empty_without_fetch(monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("history must not be fetched for no tickers")

    monkeypatch.setattr(yp_module, "_history", _forbidden)
    assert download_close_prices([]).empty


def test_download_dedupes_tickers_keeps_close_and_strips_tz(monkeypatch):
    seen = {}

    def fake_history(tickers, period, start=None):
        seen["tickers"] = tickers
        return _history_frame()

    monkeypatch.setattr(yp_module, "_history", fake_history)
    out = download_close_prices(["AAA", "BBB", "AAA"])
    assert seen["tickers"] == ["AAA", "BBB"]
    assert list(out.columns) == ["AAA", "BBB"]
    assert out.index.tz is None
    np.testing.assert_allclose(out["AAA"], [100.0, 110.0])


def test_download_normalize_scales_to_first_print(monkeypatch):
    monkeypatch.setattr(yp_module, "_history",
                        lambda t, p, start=None: _history_frame())
    out = download_close_prices(["AAA", "BBB"], normalize=True)
    np.testing.assert_allclose(out.iloc[0], [1.0, 1.0])
    np.testing.assert_allclose(out["BBB"], [1.0, 1.1])


def test_download_empty_history_returns_empty(monkeypatch):
    # Empty batch AND an empty chart path: nothing to return.
    # (An empty batch alone no longer ends the call — it is
    # the extreme stub, refilled from the chart path; see
    # test_download_empty_batch_is_refilled in
    # test_yahoo_prices.py.)
    monkeypatch.setattr(yp_module, "_history",
                        lambda t, p, start=None: pd.DataFrame())
    monkeypatch.setattr(yp_module, "yahoo_daily",
                        lambda t, p1=0, p2=None, include_partial=False:
                        pd.Series(dtype=float))
    assert download_close_prices(["AAA"]).empty


# --- yahoo_daily / fetch_json (raw chart API) --------------------------------


def _chart_payload():
    def _ts(year, month, day):
        dt = datetime.datetime(year, month, day, 14, 30, tzinfo=UTC)
        return int(dt.timestamp())

    return {"chart": {"result": [{
        "timestamp": [_ts(2024, 1, 2), _ts(2024, 1, 3), _ts(2024, 1, 4)],
        "indicators": {"adjclose": [{"adjclose": [100.0, None, 102.5]}]},
    }]}}


def test_yahoo_daily_parses_adjclose_drops_nan(monkeypatch):
    seen = {}

    def fake_fetch(url, timeout, user_agent=None):
        seen["url"] = url
        return _chart_payload()

    monkeypatch.setattr(yp_module, "fetch_json", fake_fetch)
    out = yahoo_daily("SPY", p1=0, p2=1_800_000_000)
    assert "chart/SPY" in seen["url"]
    assert list(out.index) == [pd.Timestamp("2024-01-02"),
                               pd.Timestamp("2024-01-04")]
    assert out.index.tz is None
    np.testing.assert_allclose(out, [100.0, 102.5])


def test_fetch_json_parses_body_and_sends_user_agent(monkeypatch):
    import io
    import urllib.request

    captured = {}

    class _Resp:
        def __enter__(self):
            return io.BytesIO(b'{"a": 1}')

        def __exit__(self, *exc):
            return False

    def fake_urlopen(req, timeout=None):
        captured["ua"] = req.get_header("User-agent")
        captured["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert yp_module.fetch_json("https://example.test/x", 7) == {"a": 1}
    assert captured == {"ua": "toolbox-datazoo", "timeout": 7}


# --- StockIndices ----------------------------------------------------------


def test_stock_indices_available_and_labels():
    si = StockIndices()
    assert "sp500" in si.available
    assert len(si.available) == 19
    assert si.label(" SP500 ") == "S&P 500"
    assert si.label("nope") == "nope"


def test_stock_indices_unknown_name_returns_empty():
    si = StockIndices()
    assert si.get("nope") == []
    frame = si.get("nope", as_frame=True)
    assert frame.empty
    assert list(frame.columns) == ["ticker", "name"]


def test_stock_indices_get_converts_dots_to_hyphens(monkeypatch):
    si = StockIndices()
    table = pd.DataFrame(
        {"Symbol": ["BRK.B", "AAPL"], "Security": ["Berkshire", "Apple Inc"]}
    )
    monkeypatch.setattr(si, "_table", lambda name: table)
    assert si.get("sp500") == ["BRK-B", "AAPL"]
    frame = si.get("sp500", as_frame=True)
    assert frame.to_dict("list") == {
        "ticker": ["BRK-B", "AAPL"],
        "name": ["Berkshire", "Apple Inc"],
    }


def test_stock_indices_frame_falls_back_to_ticker_as_name(monkeypatch):
    si = StockIndices()
    monkeypatch.setattr(si, "_table", lambda name: pd.DataFrame({"Symbol": ["X.Y"]}))
    frame = si.get("sp500", as_frame=True)
    assert frame.to_dict("list") == {"ticker": ["X-Y"], "name": ["X-Y"]}


def test_stock_indices_fetch_failure_returns_empty(monkeypatch):
    monkeypatch.setattr(si_module.requests, "get", _fail_get)
    si = StockIndices()
    assert si.get("sp500") == []


# --- FearGreed: CNN --------------------------------------------------------


def _cnn_payload():
    return {
        "fear_and_greed_historical": {
            "data": [
                {"x": _ts_ms(2024, 1, 1, 12), "y": 10.0},
                {"x": _ts_ms(2024, 1, 1, 18), "y": 20.0},
                {"x": _ts_ms(2024, 1, 2, 12), "y": 30.0},
            ],
            "score": 55.0,
            "rating": "Greed",
        },
        # A component without a data array is skipped.
        "market_momentum_sp500": {"data": [], "score": 1, "rating": "x"},
    }


def test_cnn_history_keeps_last_point_per_day(monkeypatch):
    monkeypatch.setattr(
        fg_module.requests, "get", lambda *a, **k: _Response(_cnn_payload())
    )
    history = FearGreed().cnn_history()
    assert len(history) == 1
    entry = history[0]
    assert entry["key"] == "fear_and_greed_historical"
    assert entry["label"] == "Fear & Greed Index (0-100)"
    assert entry["y"] == [20.0, 30.0]
    assert entry["x"] == [
        datetime.datetime(2024, 1, 1, tzinfo=UTC),
        datetime.datetime(2024, 1, 2, tzinfo=UTC),
    ]
    assert entry["score"] == 55.0
    assert entry["rating"] == "Greed"


def test_cnn_history_dataframe_shape(monkeypatch):
    monkeypatch.setattr(
        fg_module.requests, "get", lambda *a, **k: _Response(_cnn_payload())
    )
    df = FearGreed().cnn_history(return_df=True)
    assert list(df.index) == ["2024-01-01", "2024-01-02"]
    assert list(df.columns) == ["fear_and_greed_historical"]
    assert df.iloc[0, 0] == 20.0


def test_cnn_history_failure_returns_empty(monkeypatch):
    monkeypatch.setattr(fg_module.requests, "get", _fail_get)
    fg = FearGreed()
    assert fg.cnn_history() == []
    assert fg.cnn_history(return_df=True).empty


# --- FearGreed: crypto -----------------------------------------------------


def _crypto_payload():
    return {
        "data": [
            {
                "timestamp": "1704153600",
                "value": "40",
                "value_classification": "Fear",
            },
            {
                "timestamp": "1704067200",
                "value": "30",
                "value_classification": "Fear",
            },
        ]
    }


def test_crypto_history_sorts_oldest_first(monkeypatch):
    monkeypatch.setattr(
        fg_module.requests, "get", lambda *a, **k: _Response(_crypto_payload())
    )
    out = FearGreed().crypto_history(limit=2)
    assert out["key"] == "crypto_fear_and_greed"
    assert out["y"] == [30, 40]
    assert out["score"] == 40
    assert out["rating"] == "Fear"
    assert out["x"][0] < out["x"][1]


def test_crypto_history_failure_returns_empty_dict(monkeypatch):
    monkeypatch.setattr(fg_module.requests, "get", _fail_get)
    assert FearGreed().crypto_history() == {}


def test_fear_greed_history_combines_cnn_and_crypto(monkeypatch):
    def fake_get(url, **kwargs):
        if "alternative.me" in url:
            return _Response(_crypto_payload())
        return _Response(_cnn_payload())

    monkeypatch.setattr(fg_module.requests, "get", fake_get)
    df = FearGreed().fear_greed_history(limit=2)
    assert list(df.index) == ["2024-01-01", "2024-01-02"]
    assert set(df.columns) == {"fear_and_greed_historical", "crypto_fear_and_greed"}
    np.testing.assert_allclose(df["crypto_fear_and_greed"], [30.0, 40.0])


# --- ZillowZipData ----------------------------------------------------------


def test_zillow_normalize_stringifies_ids_and_pads_zips():
    raw = pd.DataFrame({"RegionID": [123, 456], "RegionName": ["1234", "90210"]})
    out = ZillowZipData._normalize_zip_frame(raw)
    assert out["RegionID"].tolist() == ["123", "456"]
    assert out["RegionName"].tolist() == ["01234", "90210"]
    # The caller's frame is not mutated.
    assert raw["RegionName"].tolist() == ["1234", "90210"]


def test_zillow_urls_are_https_csvs():
    for url in (ZillowZipData.ZHVI_URL, ZillowZipData.ZHVF_URL):
        assert url.startswith("https://")
        assert url.endswith(".csv")


def test_download_ticker_sectors_maps_and_marks_missing(monkeypatch):
    import sys
    import types

    from toolbox.DataZoo.market.yahoo_prices import download_ticker_sectors

    infos = {
        "ASML": {"sector": "Technology"},
        "HSBC": {"sector": "Financial Services"},
        "IXC": {},  # a fund: no sector reported
    }

    class _FakeTicker:
        def __init__(self, symbol):
            self._symbol = symbol

        @property
        def info(self):
            if self._symbol == "BOOM":
                raise RuntimeError("network down")
            return infos[self._symbol]

    fake_yf = types.ModuleType("yfinance")
    fake_yf.Ticker = _FakeTicker
    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)

    seen = []
    out = download_ticker_sectors(
        ["ASML", "HSBC", "IXC", "BOOM", "ASML"],
        progress=lambda d, t, label: seen.append((d, t)))
    assert list(out.index) == ["ASML", "HSBC", "IXC", "BOOM"]
    assert out["ASML"] == "Technology"
    assert out["HSBC"] == "Financial Services"
    assert out["IXC"] is None  # missing surfaces as missing
    assert out["BOOM"] is None  # a failed read, never a guess
    assert seen[0] == (0, 4) and seen[-1] == (4, 4)
