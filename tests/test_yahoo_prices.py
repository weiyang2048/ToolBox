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
