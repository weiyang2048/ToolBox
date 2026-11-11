"""prediction_markets: the pure transforms over Gamma
event JSON — exclusion filters, the leading-outcome
pick, the dynamic next-Fed selection, and the render
blocks. Offline: fixtures only, no API calls."""

import json
from datetime import datetime, timedelta, timezone

from toolbox.DataZoo.market.prediction_markets import (
    _fed_feature,
    _leading,
    _row,
    _rows,
    chart_blocks,
)

NOW = datetime(2026, 11, 10, tzinfo=timezone.utc)


def _market(label, prob, chg=0.0, token="1"):
    return {"groupItemTitle": label,
            "outcomePrices": json.dumps([str(prob),
                                         str(1 - prob)]),
            "oneDayPriceChange": chg,
            "clobTokenIds": json.dumps([token, "2"])}


def _event(title, markets, tags=(), end=None, vol24=1000):
    return {"title": title, "markets": markets,
            "tags": [{"slug": t} for t in tags],
            "endDate": end, "volume24hr": vol24}


def test_leading_picks_the_highest_priced_outcome():
    ev = _event("Fed Decision in December?", [
        _market("25 bps decrease", 0.01),
        _market("25 bps increase", 0.85, chg=0.06),
        _market("No change", 0.04)])
    assert _leading(ev) == ("25 bps increase", 0.85, 0.06)


def test_leading_binary_market_is_yes():
    ev = _event("Will X happen?", [
        {"outcomePrices": json.dumps(["0.3", "0.7"]),
         "oneDayPriceChange": -0.02}])
    assert _leading(ev) == ("Yes", 0.3, -0.02)


def test_row_excludes_sports_and_entertainment():
    game = _event("Lakers vs. Celtics", [_market("Lakers", 0.5)],
                  tags=("sports", "nba"))
    assert _row(game) is None
    stream = _event("Streamer finishes GTA marathon",
                    [_market("Yes", 0.9)], tags=("gaming",))
    assert _row(stream) is None
    pol = _event("Presidential Election Winner 2028",
                 [_market("JD Vance", 0.17)], tags=("politics",))
    r = _row(pol)
    assert r["lead"] == "JD Vance" and r["prob"] == 0.17


def test_rows_caps_and_skips_unpriceable():
    bad = _event("Broken", [{"outcomePrices": "n/a"}])
    good = _event("Fine", [_market("Yes", 0.5)])
    assert [r["title"] for r in _rows([bad, good, good], 1)] \
        == ["Fine"]


def _fed(title, end, outcomes):
    return _event(title, [_market(lbl, p) for lbl, p in outcomes],
                  end=end)


def test_fed_feature_picks_the_next_meeting():
    dec = _fed("Fed Decision in December?",
               "2026-12-10T04:59:00Z",
               [("No change", 0.04), ("25 bps increase", 0.85)])
    jan = _fed("Fed Decision in January?",
               "2027-01-28T04:59:00Z",
               [("No change", 0.6)])
    other = _event("Another Fed rate hike in 2026?",
                   [_market("Yes", 0.2)],
                   end="2026-12-31T00:00:00Z")
    fed = _fed_feature([jan, other, dec], now=NOW)
    assert fed["heading"] == "Fed decision — December 2026 FOMC"
    # Outcomes sorted by probability, descending.
    assert [o[0] for o in fed["outcomes"]] == [
        "25 bps increase", "No change"]


def test_fed_feature_skips_a_resolved_meeting():
    past = _fed("Fed Decision in October?",
                "2026-10-29T04:59:00Z", [("No change", 0.9)])
    jan = _fed("Fed Decision in January?",
               "2027-01-28T04:59:00Z", [("No change", 0.6)])
    fed = _fed_feature([past, jan], now=NOW)
    assert fed["heading"].startswith("Fed decision — January 2027")
    assert _fed_feature([past], now=NOW) is None


def _payload(with_fed=True):
    row = {"title": "Presidential Election Winner 2028",
           "lead": "JD Vance", "prob": 0.17, "chg": -0.002,
           "vol24": 1_000_000}
    fed = None
    if with_fed:
        t0 = NOW - timedelta(days=30)
        fed = {"heading": "Fed decision — December 2026 FOMC",
               "outcomes": [("25 bps increase", 0.85, 0.06, "tok")],
               "vol24": 7_000_000,
               "hist": {"25 bps increase":
                        [(t0, 0.7), (NOW, 0.85)]}}
    return {"fed": fed, "top": [row],
            "cats": {"Politics": [row]}}


def test_blocks_order_fed_first_then_boards_then_note():
    blocks = chart_blocks(_payload())
    kinds = [k for k, _ in blocks]
    assert kinds == ["html", "html", "fig", "html",
                     "html", "html"]
    html = "".join(v for k, v in blocks if k == "html")
    assert "Fed decision — December 2026 FOMC" in html
    assert "Most traded — ex-sports" in html
    assert "Politics" in html and "JD Vance" in html
    assert "85.0%" in html


def test_blocks_without_fed_have_no_figure():
    blocks = chart_blocks(_payload(with_fed=False))
    assert all(k == "html" for k, _ in blocks)
    html = "".join(v for _, v in blocks)
    assert "Fed decision —" not in html
    assert "Most traded — ex-sports" in html
