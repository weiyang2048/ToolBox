"""Tests for toolbox.DataZoo.market.edgar_13f (SEC 13F-HR parsing).

Offline: only the pure parsing/scoring helpers and the parameterization
plumbing are exercised — no SEC or OpenFIGI traffic.
"""

from toolbox.DataZoo.market import edgar_13f as e

XML = """<informationTable>
  <infoTable>
    <nameOfIssuer>APPLE INC</nameOfIssuer>
    <titleOfClass>COM</titleOfClass>
    <cusip>037833100</cusip>
    <value>1500</value>
    <shrsOrPrnAmt><sshPrnamt>100</sshPrnamt></shrsOrPrnAmt>
  </infoTable>
  <infoTable>
    <nameOfIssuer>APPLE INC</nameOfIssuer>
    <titleOfClass>COM</titleOfClass>
    <cusip>037833100</cusip>
    <value>500</value>
    <shrsOrPrnAmt><sshPrnamt>40</sshPrnamt></shrsOrPrnAmt>
  </infoTable>
  <infoTable>
    <nameOfIssuer>TESLA INC</nameOfIssuer>
    <titleOfClass>COM</titleOfClass>
    <cusip>88160R101</cusip>
    <value>900</value>
    <shrsOrPrnAmt><sshPrnamt>10</sshPrnamt></shrsOrPrnAmt>
    <putCall>Put</putCall>
  </infoTable>
  <infoTable>
    <nameOfIssuer>ZERO CORP</nameOfIssuer>
    <titleOfClass>COM</titleOfClass>
    <cusip>000000000</cusip>
    <value>0</value>
    <shrsOrPrnAmt><sshPrnamt>5</sshPrnamt></shrsOrPrnAmt>
  </infoTable>
</informationTable>"""


def test_sec_headers_carries_caller_user_agent():
    headers = e.sec_headers("My App admin@example.com")
    assert headers["User-Agent"] == "My App admin@example.com"
    assert headers["Accept-Encoding"] == "gzip, deflate"


def test_load_13f_investors_from_mapping_filters_cikless():
    out = e.load_13f_investors(
        {"funds": {"name": "Fund A", "cik": "123"}, "junk": {"name": "No CIK"}}
    )
    assert list(out) == ["funds"]


def test_parse_holdings_xml_skips_puts_and_zero_values():
    rows = e._parse_holdings_xml(XML)
    assert [r["cusip"] for r in rows] == ["037833100", "037833100"]
    assert rows[0]["issuer"] == "APPLE INC"
    assert rows[0]["value"] == 1500 and rows[0]["shares"] == 100


def test_aggregate_sums_by_cusip_sorted_by_value():
    rows = e._aggregate(e._parse_holdings_xml(XML))
    assert len(rows) == 1
    assert rows[0]["value"] == 2000 and rows[0]["shares"] == 140


def test_pick_figi_ticker_prefers_us_common_stock():
    records = [
        {"ticker": "AAPL", "marketSector": "Equity",
         "securityType": "Common Stock", "exchCode": "US"},
        {"ticker": "APC", "marketSector": "Equity",
         "securityType": "Common Stock", "exchCode": "GR"},
        {"ticker": "AAPL", "marketSector": "Equity",
         "securityType": "Option", "exchCode": "US"},
    ]
    assert e._pick_figi_ticker(records) == "AAPL"


def test_format_13f_asof_note():
    meta = {"report_period": "2026-09-30", "filing_date": "2026-11-14"}
    assert e.format_13f_asof_note(meta) == "Holdings as of Sep 30, 2026 · filed Nov 14, 2026"
    assert e.format_13f_asof_note(None) == ""


# ---------------------------------------------------------------------------
# Known-value test for load_13f_lens (idea search #2, candidate 3).
# The network seams are fixtured at exactly what the function
# reads: the SEC submissions response (via _sec_get) and the two
# on-disk caches (holdings_<cik>.json, cusip_tickers.json). All
# expected values are hand-computed from the fixture rows.
# ---------------------------------------------------------------------------

import json

import pytest

_LENS_ROWS = [
    {"cusip": "037833100", "issuer": "APPLE INC", "value": 3000, "shares": 200},
    {"cusip": "037833104", "issuer": "APPLE INC", "value": 2500, "shares": 150},
    {"cusip": "594918104", "issuer": "MICROSOFT CORP", "value": 2000, "shares": 50},
    {"cusip": "88160R101", "issuer": "TESLA INC", "value": 1000, "shares": 10},
]
_LENS_ACCESSION = "0000000123-26-000001"


def test_load_13f_lens_hand_computed(tmp_path, monkeypatch):
    # Cache the aggregated holdings (the shape _aggregate emits:
    # per-CUSIP sums, value-descending) and the CUSIP -> ticker
    # map, so no OpenFIGI traffic is needed either.
    (tmp_path / "holdings_123.json").write_text(json.dumps(
        {"cik": "123", "accession": _LENS_ACCESSION,
         "filing_date": "2026-11-14", "report_period": "2026-09-30",
         "rows": _LENS_ROWS}))
    (tmp_path / "cusip_tickers.json").write_text(json.dumps(
        {"037833100": "AAPL", "037833104": "AAPL",
         "594918104": "MSFT", "88160R101": "TSLA"}))
    submissions = {"filings": {"recent": {
        "form": ["13F-HR"], "accessionNumber": [_LENS_ACCESSION],
        "filingDate": ["2026-11-14"], "reportDate": ["2026-09-30"]}}}

    class _Resp:
        def json(self):
            return submissions

    def fake_sec_get(url, user_agent):
        assert "submissions" in url  # the only live read left
        return _Resp()

    monkeypatch.setattr(e, "_sec_get", fake_sec_get)
    sections, labels, weights, asof = e.load_13f_lens(
        {"fund_a": {"name": "Fund A (Class I)", "cik": "123"}},
        "Test test@example.com", top_n=3, cache_dir=tmp_path)

    # Hand computation. Filing total = 3000+2500+2000+1000 = 8500
    # (the denominator is the whole filing, including the row
    # skipped below). Candidates walk value-descending: AAPL
    # (3000) in; the second APPLE cusip also maps to AAPL and is
    # skipped as a duplicate; MSFT (2000) and TSLA (1000) fill
    # top_n = 3. Weights are 100 * value / 8500; labels are the
    # title-cased issuer of the first row per ticker; the section
    # title drops the " (Class I)" suffix.
    total = 8500.0
    assert sections == {"Fund A": ["AAPL", "MSFT", "TSLA"]}
    assert weights == {"Fund A": {
        "AAPL": pytest.approx(100.0 * 3000 / total),
        "MSFT": pytest.approx(100.0 * 2000 / total),
        "TSLA": pytest.approx(100.0 * 1000 / total),
    }}
    assert labels == {"AAPL": "Apple Inc", "MSFT": "Microsoft Corp",
                      "TSLA": "Tesla Inc"}
    assert asof["Fund A"]["accession"] == _LENS_ACCESSION
    assert asof["Fund A"]["report_period"] == "2026-09-30"
    assert asof["Fund A"]["filing_date"] == "2026-11-14"
