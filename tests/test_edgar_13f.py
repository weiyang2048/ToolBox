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
