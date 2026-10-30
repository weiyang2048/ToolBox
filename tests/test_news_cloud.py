"""News word cloud: date parsing, topic extraction (lexicon +
mining, boilerplate filter), and the embeddable frame HTML.
All offline — synthetic headlines only."""
from toolbox.DataZoo.market.news_cloud import (
    _family_of, _mine_phrases, _parse_date, cloud_html, extract_topics)


def _heads(title, n, source="Test"):
    return [{"title": title, "source": source, "published": None}
            for _ in range(n)]


def test_parse_date_formats():
    assert _parse_date("Fri, 30 Oct 2026 05:00:00 GMT") is not None
    assert _parse_date("2026-10-30T05:00:00Z") is not None  # ISO, Z
    assert _parse_date("2026-10-30T14:00:00+09:00") is not None
    assert _parse_date("not a date") is None
    assert _parse_date(None) is None


def test_lexicon_topics_and_groups():
    heads = (_heads("Federal Reserve holds rates steady as inflation cools", 3)
             + _heads("Bitcoin slides below $100,000 in crypto selloff", 2))
    topics = {t["term"]: t for t in extract_topics(heads)}
    assert topics["Federal Reserve"]["group"] == "macro"
    assert topics["Bitcoin"]["group"] == "crypto"
    assert topics["Federal Reserve"]["weight"] == 3


def test_mining_finds_uncovered_phrase():
    heads = _heads("Jio Platforms unveils new data centre plan", 5)
    mined = _mine_phrases(heads, known_terms=[])
    assert any(m["term"] == "Jio Platforms" for m in mined)


def test_mining_filters_boilerplate():
    heads = _heads("Q2 results 2026: Acme Corp share price jumps 4%", 6)
    terms = [m["term"] for m in _mine_phrases(heads, known_terms=[])]
    assert not any("Results" in t or "Share Price" in t for t in terms)


def test_family_classification():
    assert _family_of("bitcoin etf inflows") == "crypto"
    assert _family_of("oil prices climb") == "commodities"
    assert _family_of("some unclassifiable phrase") == "neutral"


def test_cloud_html_frame():
    topics = [{"term": "Midterms", "weight": 18, "group": "macro",
               "example_headline": "x"},
              {"term": "Intel", "weight": 11, "group": "tech",
               "example_headline": "y"}]
    html = cloud_html(topics, {"left": "As of test", "right": "Size"})
    assert "nc-frame" in html and "Midterms" in html
    assert "linear-gradient" in html and "As of test" in html
    # The leader renders larger than the smaller term.
    assert html.index("Midterms") > -1 and "font-size:46px" in html


def test_mining_dedupes_hyphen_variants_of_lexicon_terms():
    heads = _heads("Tech Sell-Off deepens as chip stocks slide", 6)
    mined = _mine_phrases(heads, known_terms=["Tech selloff"])
    assert not any("sell" in m["term"].lower() for m in mined)
