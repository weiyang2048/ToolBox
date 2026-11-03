"""The shared Yahoo-leg retry (yahoo_prices.with_retry):
transient HTTP errors are retried with backoff, permanent
ones are not, and a final failure names its leg. This is the
canonical home of the helper the power law, the dollar and
US vs ex-US fetchers all share; the power law's own tests
cover the same semantics through its module alias."""
import time

import pytest

from toolbox.DataZoo.market import yahoo_prices as yp


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)


def test_with_retry_recovers_from_transient_429():
    from urllib.error import HTTPError

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise HTTPError("u", 429, "Too Many Requests",
                            None, None)
        return "ok"

    assert yp.with_retry(flaky, "leg-x") == "ok"
    assert calls["n"] == 3


def test_with_retry_names_the_leg_on_final_failure():
    def bad():
        raise ValueError("boom")

    with pytest.raises(RuntimeError, match="leg-y: boom"):
        yp.with_retry(bad, "leg-y")


def test_with_retry_exhausts_attempts_on_persistent_429():
    from urllib.error import HTTPError

    calls = {"n": 0}

    def limited():
        calls["n"] += 1
        raise HTTPError("u", 429, "Too Many Requests",
                        None, None)

    with pytest.raises(RuntimeError, match="leg-w: HTTP"):
        yp.with_retry(limited, "leg-w")
    assert calls["n"] == 3


def test_with_retry_does_not_retry_permanent_http_errors():
    from urllib.error import HTTPError

    calls = {"n": 0}

    def gone():
        calls["n"] += 1
        raise HTTPError("u", 404, "Not Found", None, None)

    with pytest.raises(RuntimeError, match="leg-z"):
        yp.with_retry(gone, "leg-z")
    assert calls["n"] == 1


# --- chart_payload: the shared raw-chart cache (2026-11-02) ---

def _payload(p1, closes=(1.0, 2.0, 3.0)):
    ts = [p1 + 86400 * i for i in range(len(closes))]
    return {"chart": {"result": [{
        "timestamp": ts,
        "indicators": {
            "quote": [{"close": list(closes)}],
            "adjclose": [{"adjclose": list(closes)}]}}]}}


@pytest.fixture(autouse=True)
def _clear_chart_cache():
    yp._chart_cache.clear()
    yp._chart_inflight.clear()
    yield
    yp._chart_cache.clear()
    yp._chart_inflight.clear()


def _counting_fetch(calls, fail_first=False):
    def fake(url, timeout, user_agent=None):
        calls["n"] += 1
        if fail_first and calls["n"] == 1:
            raise ValueError("boom")
        p1 = int(url.split("period1=")[1].split("&")[0])
        return _payload(p1)
    return fake


def test_chart_payload_serves_a_narrower_window_from_cache(
        monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(yp, "fetch_json", _counting_fetch(calls))
    wide = yp.chart_payload("XXX", p1=1000, p2=2000)
    narrow = yp.chart_payload("XXX", p1=1500, p2=2000)
    assert narrow is wide
    assert calls["n"] == 1


def test_chart_payload_refetches_for_a_wider_window(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(yp, "fetch_json", _counting_fetch(calls))
    yp.chart_payload("XXX", p1=1500, p2=2000)
    yp.chart_payload("XXX", p1=1000, p2=2000)
    assert calls["n"] == 2


def test_chart_payload_refetches_after_the_ttl(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(yp, "fetch_json", _counting_fetch(calls))
    now = [10_000.0]
    monkeypatch.setattr(yp.time, "monotonic", lambda: now[0])
    yp.chart_payload("XXX", p1=1000, p2=2000)
    now[0] += yp._CHART_TTL_SECONDS + 1
    yp.chart_payload("XXX", p1=1000, p2=2000)
    assert calls["n"] == 2


def test_chart_payload_never_caches_a_failure(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(
        yp, "fetch_json", _counting_fetch(calls, fail_first=True))
    with pytest.raises(ValueError, match="boom"):
        yp.chart_payload("XXX", p1=1000, p2=2000)
    assert yp.chart_payload("XXX", p1=1000, p2=2000) is not None
    assert calls["n"] == 2


def test_chart_payload_fetches_once_under_concurrency(
        monkeypatch):
    import threading

    calls = {"n": 0}

    def slow(url, timeout, user_agent=None):
        calls["n"] += 1
        time.sleep(0.2)
        return _payload(1000)

    monkeypatch.setattr(yp, "fetch_json", slow)
    out = []
    threads = [threading.Thread(
        target=lambda: out.append(
            yp.chart_payload("XXX", p1=1000, p2=2000)))
        for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert calls["n"] == 1
    assert len(out) == 4 and all(o is out[0] for o in out)


# --- _repaired_adjclose: Yahoo's unfilled trailing bar (2026-11-02) ---

def _result(adj, closes=None, meta=None, events=None):
    import datetime as _dt

    # Fri Oct 30 and Mon Nov 2, 2026, 14:30 UTC session stamps.
    ts = [int(_dt.datetime(2026, 10, 30, 14, 30,
                           tzinfo=_dt.timezone.utc).timestamp()),
          int(_dt.datetime(2026, 11, 2, 14, 30,
                           tzinfo=_dt.timezone.utc).timestamp())]
    r = {"timestamp": ts,
         "indicators": {
             "quote": [{"close": list(closes or adj)}],
             "adjclose": [{"adjclose": list(adj)}]}}
    if meta is not None:
        r["meta"] = meta
    if events is not None:
        r["events"] = events
    return r


def _stamp(y, m, d, hh=0):
    import datetime as _dt

    return int(_dt.datetime(y, m, d, hh, tzinfo=_dt.timezone.utc)
               .timestamp())


_CLOSE_META = {"regularMarketPrice": 29.41,
               "regularMarketTime": _stamp(2026, 11, 2, 21)}


def test_repaired_adjclose_fills_the_unfilled_bar():
    r = _result([29.47, None], meta=_CLOSE_META)
    out = yp._repaired_adjclose(r)
    assert out[0] == 29.47
    assert abs(out[1] - 29.41) < 1e-9  # ratio 1.0 at the anchor bar


def test_repaired_adjclose_scales_by_the_standing_ratio():
    r = _result([20.0, None], closes=[25.0, None], meta=_CLOSE_META)
    out = yp._repaired_adjclose(r)
    assert abs(out[1] - 29.41 * 20.0 / 25.0) < 1e-9


def test_repaired_adjclose_skips_a_dividend_on_the_bar():
    r = _result([29.47, None], meta=_CLOSE_META,
                events={"dividends": {"x": {"date": _stamp(2026, 11, 2)}}})
    assert yp._repaired_adjclose(r)[1] is None


def test_repaired_adjclose_skips_a_stale_meta():
    stale = {"regularMarketPrice": 29.41,
             # Nov 2 00:00 UTC is still Nov 1 in New York.
             "regularMarketTime": _stamp(2026, 11, 2)}
    r = _result([29.47, None], meta=stale)
    assert yp._repaired_adjclose(r)[1] is None


def test_repaired_adjclose_noops_on_a_filled_bar():
    r = _result([29.47, 29.41], meta=_CLOSE_META)
    assert yp._repaired_adjclose(r) == [29.47, 29.41]


def _crypto_result():
    # BTC-USD, Oct 30 - Nov 3 2026: the Nov 2 bar entirely
    # unfilled, the Nov 3 bar live; bars are contiguous, so
    # Nov 3's open is Nov 2's close.
    r = _result([77198.21, None])
    r["timestamp"] = [_stamp(2026, 10, 30), _stamp(2026, 10, 31),
                      _stamp(2026, 11, 1), _stamp(2026, 11, 2),
                      _stamp(2026, 11, 3)]
    r["indicators"]["quote"][0]["close"] = [
        77198.21, 77434.75, 78248.45, None, 76813.70]
    r["indicators"]["quote"][0]["open"] = [
        77775.11, 77198.21, 77434.75, None, 76683.89]
    r["indicators"]["adjclose"][0]["adjclose"] = [
        77198.21, 77434.75, 78248.45, None, 76813.70]
    r["meta"] = {"instrumentType": "CRYPTOCURRENCY",
                 "regularMarketPrice": 76813.70,
                 "regularMarketTime": _stamp(2026, 11, 3, 2)}
    return r


def test_repaired_adjclose_fills_a_crypto_interior_bar():
    out = yp._repaired_adjclose(_crypto_result())
    assert out[3] == 76683.89  # the next bar's open, ratio 1
    assert out[4] == 76813.70  # the live bar untouched


def test_repaired_adjclose_leaves_an_equity_interior_bar():
    r = _crypto_result()
    r["meta"] = {"instrumentType": "EQUITY",
                 "regularMarketPrice": 76813.70,
                 "regularMarketTime": _stamp(2026, 11, 3, 2)}
    assert yp._repaired_adjclose(r)[3] is None


# --- the intraday fallback: an index's unfilled session (2026-11-02) ---

@pytest.fixture(autouse=True)
def _clear_intraday_cache():
    yp._INTRADAY_CACHE.clear()
    yield
    yp._INTRADAY_CACHE.clear()


def _dxy_daily_payload():
    # DX-Y.NYB as served on Mon Nov 2, 2026, ~21:15 ET: Fri
    # filled, a null Sunday placeholder, Monday's bar null,
    # and a live pseudo-bar stamped at the fetch instant.
    ts = [_stamp(2026, 10, 30, 4), _stamp(2026, 11, 1, 4),
          _stamp(2026, 11, 2, 5), _stamp(2026, 11, 3, 2)]
    vals = [101.88, None, None, 101.829]
    return {"chart": {"result": [{
        "timestamp": ts,
        "meta": {"instrumentType": "INDEX",
                 "regularMarketPrice": 101.829,
                 "regularMarketTime": _stamp(2026, 11, 3, 2)},
        "indicators": {
            "quote": [{"close": list(vals),
                       "open": list(vals)}],
            "adjclose": [{"adjclose": list(vals)}]}}]}}


def _intraday_payload():
    # Monday hourly bars: 09:00/10:00/11:00 ET under the
    # 16:00 cutoff, plus a 16:00 ET bar (excluded by it).
    ts = [_stamp(2026, 11, 2, 14), _stamp(2026, 11, 2, 15),
          _stamp(2026, 11, 2, 16), _stamp(2026, 11, 2, 21)]
    return {"chart": {"result": [{
        "timestamp": ts,
        "indicators": {"quote": [
            {"close": [101.50, 101.70, 101.82, 999.0]}]}}]}}


def test_yahoo_daily_fills_an_unfilled_session_from_intraday(
        monkeypatch):
    monkeypatch.setattr(yp, "chart_payload",
                        lambda t, a, b: _dxy_daily_payload())
    monkeypatch.setattr(yp, "fetch_json",
                        lambda url, timeout, user_agent=None:
                        _intraday_payload())
    s = yp.yahoo_daily("DX-Y.NYB", include_partial=True)
    assert list(s.index.strftime("%Y-%m-%d")) == [
        "2026-10-30", "2026-11-02", "2026-11-03"]
    assert abs(float(s.loc["2026-11-02"]) - 101.82) < 1e-9


def test_yahoo_daily_survives_an_intraday_failure(monkeypatch):
    monkeypatch.setattr(yp, "chart_payload",
                        lambda t, a, b: _dxy_daily_payload())

    def boom(url, timeout, user_agent=None):
        raise ValueError("no intraday")

    monkeypatch.setattr(yp, "fetch_json", boom)
    s = yp.yahoo_daily("DX-Y.NYB", include_partial=True)
    assert list(s.index.strftime("%Y-%m-%d")) == [
        "2026-10-30", "2026-11-03"]


def test_patch_tail_fills_from_yahoo_daily(monkeypatch):
    import pandas as pd

    idx = pd.to_datetime(["2026-10-30", "2026-11-02"])
    frame = pd.DataFrame({"SPY": [755.72, float("nan")]}, index=idx)
    monkeypatch.setattr(
        yp, "yahoo_daily",
        lambda t, p1=0, p2=None, include_partial=False:
        pd.Series([752.10], index=pd.to_datetime(["2026-11-02"])))
    out = yp._patch_tail(frame)
    assert float(out.loc["2026-11-02", "SPY"]) == 752.10
    assert float(out.loc["2026-10-30", "SPY"]) == 755.72


def _batch_frame(cols):
    """A yfinance-shaped batch history: (Price, Ticker) columns."""
    import pandas as pd

    idx = pd.bdate_range("2026-06-01", periods=100)
    data = {}
    for tic, vals in cols.items():
        data[("Close", tic)] = vals
    frame = pd.DataFrame(data, index=idx)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    return frame


def _full_series(v0=10.0):
    import pandas as pd

    idx = pd.bdate_range("2026-06-01", periods=100)
    return pd.Series([v0 + i * 0.1 for i in range(100)], index=idx)


def test_download_refills_a_stub_column(monkeypatch):
    # The batch returns BBB as a 3-row stub (a throttled leg);
    # the chart path carries its full history, which must
    # replace the stub (the 2026-11-02 IXN incident).
    import numpy as np
    import pandas as pd

    full = [100.0 + i for i in range(100)]
    stub = [5.0, 5.1, 5.2] + [float("nan")] * 97
    monkeypatch.setattr(yp, "_history",
                        lambda t, p: _batch_frame({"AAA": full,
                                                   "BBB": stub}))
    monkeypatch.setattr(
        yp, "yahoo_daily",
        lambda t, p1=0, p2=None, include_partial=False:
        _full_series(50.0))
    out = yp.download_close_prices(["AAA", "BBB"], period="5y")
    assert int(out["BBB"].notna().sum()) == 100
    assert float(out["BBB"].iloc[0]) == 50.0
    assert int(out["AAA"].notna().sum()) == 100


def test_download_refills_a_missing_column(monkeypatch):
    import pandas as pd

    full = [100.0 + i for i in range(100)]
    monkeypatch.setattr(yp, "_history",
                        lambda t, p: _batch_frame({"AAA": full}))
    monkeypatch.setattr(
        yp, "yahoo_daily",
        lambda t, p1=0, p2=None, include_partial=False:
        _full_series(50.0))
    out = yp.download_close_prices(["AAA", "BBB"], period="5y")
    assert "BBB" in out.columns
    assert int(out["BBB"].notna().sum()) == 100


def test_download_keeps_a_genuinely_young_ticker(monkeypatch):
    # A young ticker's batch column is short and its refill is
    # the same short history: nothing is replaced or padded.
    import numpy as np
    import pandas as pd

    full = [100.0 + i for i in range(100)]
    young = [7.0] * 40 + [float("nan")] * 60
    monkeypatch.setattr(yp, "_history",
                        lambda t, p: _batch_frame({"AAA": full,
                                                   "NEW": young}))
    idx = pd.bdate_range("2026-06-01", periods=100)
    monkeypatch.setattr(
        yp, "yahoo_daily",
        lambda t, p1=0, p2=None, include_partial=False:
        pd.Series([7.0] * 40, index=idx[:40]))
    out = yp.download_close_prices(["AAA", "NEW"], period="5y")
    assert int(out["NEW"].notna().sum()) == 40
    assert float(out["NEW"].dropna().iloc[0]) == 7.0


def test_download_survives_a_failed_refill(monkeypatch):
    import pandas as pd

    full = [100.0 + i for i in range(100)]
    monkeypatch.setattr(yp, "_history",
                        lambda t, p: _batch_frame({"AAA": full}))

    def boom(t, p1=0, p2=None, include_partial=False):
        raise RuntimeError("chart down")

    monkeypatch.setattr(yp, "yahoo_daily", boom)
    out = yp.download_close_prices(["AAA", "BBB"], period="5y")
    assert "BBB" not in out.columns
    assert int(out["AAA"].notna().sum()) == 100


def test_yahoo_daily_slices_a_wider_cached_window(monkeypatch):
    # Bars at p1-1d, p1, p1+1d, p2, p2+1d in the cached payload;
    # only the ones inside [p1, p2] may come back.
    p1, p2 = 1_700_000_000, 1_700_000_000 + 2 * 86400
    ts = [p1 - 86400, p1, p1 + 86400, p2, p2 + 86400]
    payload = {"chart": {"result": [{
        "timestamp": ts,
        "indicators": {"adjclose": [
            {"adjclose": [9.0, 1.0, 2.0, 3.0, 9.0]}]}}]}}
    monkeypatch.setattr(yp, "chart_payload",
                        lambda t, a, b: payload)
    s = yp.yahoo_daily("XXX", p1=p1, p2=p2, include_partial=True)
    assert list(s.values) == [1.0, 2.0, 3.0]
