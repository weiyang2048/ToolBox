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
