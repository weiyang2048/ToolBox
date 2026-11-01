"""News word cloud: date parsing, topic extraction (lexicon +
mining, boilerplate filter), and the embeddable frame HTML.
All offline — synthetic headlines only."""
from toolbox.DataZoo.market.news_cloud import (
    LEXICON,
    _display,
    _family_of,
    _matches,
    _mine_phrases,
    _parse_date,
    _paywalled,
    _tokens,
    cloud_html,
    extract_topics,
)


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
    # Since the 2026-11-01 fragment suppression, the miner
    # returns the maximal phrase, not its bigram fragment.
    assert any(m["term"].startswith("Jio Platforms")
               for m in mined)


def test_tokens_keep_ampersand_entities_whole():
    toks = _tokens("S&P 500 slips as AT&T weighs M&A curbs in China")
    assert "s&p" in toks and "at&t" in toks and "m&a" in toks
    assert "s" not in toks and "p" not in toks
    assert _tokens("China's exports rise") == ["china", "exports", "rise"]
    assert _tokens("Procter & Gamble rises") == [
        "procter", "&", "gamble", "rises"]


def test_display_ampersand_initialisms():
    assert _display(["s&p", "500"]) == "S&P 500"
    assert _display(["at&t"]) == "AT&T"
    assert _display(["procter", "&", "gamble"]) == "Procter & Gamble"
    assert _display(["tech", "selloff"]) == "Tech Selloff"


def test_mining_renders_sp_500_not_p_500():
    heads = _heads("S&P 500 falls as megacap tech slides again", 6)
    terms = [m["term"] for m in _mine_phrases(heads, known_terms=[])]
    # Maximal-phrase mining (2026-11-01): the topic is the
    # whole "S&P 500 Falls"; the point of this test stands —
    # the tokenizer never splits "S&P" into a "P 500" fragment.
    assert "S&P 500 Falls" in terms
    assert "P 500" not in terms
    assert not any(t.startswith("P 500") for t in terms)


def test_mining_spaced_ampersand_trigram():
    heads = _heads("Procter & Gamble raises outlook on demand", 6)
    terms = [m["term"] for m in _mine_phrases(heads, known_terms=[])]
    assert "Procter & Gamble" in terms


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
    # The leader renders at the embed ceiling, larger than the rest.
    assert html.index("Midterms") > -1 and "font-size:60px" in html


def test_mining_dedupes_hyphen_variants_of_lexicon_terms():
    heads = _heads("Tech Sell-Off deepens as chip stocks slide", 6)
    mined = _mine_phrases(heads, known_terms=["Tech selloff"])
    terms = [m["term"] for m in mined]
    assert "Tech Sell-Off" not in terms and "Tech Selloff" not in terms


def test_render_html_without_chromium_returns_false(monkeypatch):
    import toolbox.DataZoo.market.news_cloud as nc
    monkeypatch.setattr(nc, "CHROMIUM", "/nonexistent/chromium")
    topics = [{"term": "Midterms", "weight": 5, "group": "macro",
               "example_headline": "x"}]
    assert nc.render_html(topics, "/tmp/never.png") is False


def test_build_cloud_survives_renderer_failure(monkeypatch, tmp_path):
    import toolbox.DataZoo.market.news_cloud as nc
    heads = [{"title": "Bitcoin rallies hard", "source": "T",
              "published": None}] * 3
    monkeypatch.setattr(nc, "fetch_headlines",
                        lambda hours=24, now=None: (heads, {"T": 3}))
    monkeypatch.setattr(nc, "render_html",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
    summary = nc.build_cloud(png_path=str(tmp_path / "c.png"),
                             json_path=str(tmp_path / "c.json"))
    assert summary["renderer"] == "none"
    assert summary["topics"] and summary["caption"]["left"]
    # The date and the counts carry the accent spans, unescaped.
    assert 'class="nc-hl"' in summary["caption"]["left"]


def test_sizes_spread_is_dramatic():
    from toolbox.DataZoo.market.news_cloud import _sizes
    topics = [{"term": "big", "weight": 18}, {"term": "mid", "weight": 9},
              {"term": "small", "weight": 2}]
    s = _sizes(topics, max_px=60)
    assert s["big"] == 60
    # Exponent 1.35 over the 16px floor (raised from 9px on
    # 2026-10-31): half the leader's weight lands ~33px, the
    # tail sits just above the floor.
    assert 30 <= s["mid"] <= 36 and 16 <= s["small"] <= 20
    assert s["big"] - s["small"] >= 40



def test_mining_suppresses_bigram_fragments_of_a_trigram():
    # The 2026-11-01 live case: every "protests turn" and
    # "turn violent" headline was the single phrase "protests
    # turn violent" — only the whole phrase is a topic.
    heads = _heads("Protests turn violent in France", 4)
    mined = {m["term"] for m in _mine_phrases(heads, known_terms=[])}
    assert "Protests Turn Violent" in mined
    assert "Protests Turn" not in mined
    assert "Turn Violent" not in mined


def test_mining_keeps_bigram_with_independent_usage():
    # A bigram that also stands on its own (outside the longer
    # phrase) at min_weight or more survives alongside it.
    heads = (_heads("Protests turn violent in France", 4)
             + _heads("Turn violent clashes erupt downtown", 4))
    mined = {m["term"] for m in _mine_phrases(heads, known_terms=[])}
    assert "Turn Violent" in mined
    assert "Protests Turn" not in mined
    # The trigram (weight 4) folds into the heavier bigram
    # (weight 8) via the overlap dedupe — one topic, not two.
    assert "Protests Turn Violent" not in mined


def test_topics_carry_source_links():
    heads = ([{"title": "Bitcoin rally lifts crypto sentiment",
               "source": "Test", "link": "https://example.com/btc",
               "published": None} for _ in range(3)]
             + [{"title": "Jio Platforms unveils new data centre plan",
                 "source": "Test", "link": "https://example.com/jio",
                 "published": None} for _ in range(5)])
    topics = {t["term"]: t for t in extract_topics(heads)}
    # Lexicon topic: the first matching headline's article.
    assert topics["Bitcoin"]["link"] == "https://example.com/btc"
    # Mined topic: its example headline's article.
    mined = [t for t in topics.values() if t.get("mined")]
    assert any(t["link"] == "https://example.com/jio" for t in mined)


def test_lexicon_patterns_respect_word_boundaries():
    # Substring false friends (seen live 2026-11-01: "AWS"
    # matched "lawsuit" and "withdrawals" and rendered as a
    # cloud topic): short patterns are word-boundary regexes.
    def hit(term, text):
        pats = [p for t, p, _ in LEXICON if t == term][0]
        return any(_matches(p, text.lower()) for p in pats)
    assert hit("AWS", "AWS outage hits us-east")
    assert not hit("AWS", "Judge dismisses DOJ lawsuit")
    assert not hit("AWS", "Binance resumes withdrawals")
    assert hit("Intel", "Intel unveils new chip")
    assert not hit("Intel", "Market intelligence report")
    assert hit("Crypto", "Ethereum rallies")
    assert not hit("Crypto", "Whether the Fed cuts rates")
    assert hit("Tech selloff", "Stocks rout deepens")
    assert not hit("Tech selloff", "The usual route home")


def test_paywalled_source_classification():
    assert _paywalled("WSJ Markets") and _paywalled("FT Markets")
    assert _paywalled("Bloomberg Markets")
    assert _paywalled("MarketWatch") and _paywalled("Nikkei Asia")
    assert _paywalled("The Wall Street Journal")
    assert not _paywalled("CNBC Markets")
    assert not _paywalled("BBC Business")
    assert not _paywalled("Google News Business (US)")
    assert not _paywalled("Reuters")


def test_topic_links_prefer_non_paywalled_sources():
    heads = ([{"title": "Bitcoin rally lifts crypto sentiment",
               "source": "WSJ Markets",
               "link": "https://example.com/wsj-btc",
               "published": None} for _ in range(2)]
             + [{"title": "Bitcoin rally lifts crypto sentiment",
                 "source": "CNBC Markets",
                 "link": "https://example.com/cnbc-btc",
                 "published": None} for _ in range(2)])
    topics = {t["term"]: t for t in extract_topics(heads)}
    # The paywalled hit comes first in the pool, but the
    # topic's link comes from the free source.
    assert topics["Bitcoin"]["link"] == "https://example.com/cnbc-btc"


def test_topic_links_fall_back_to_paywalled_when_only_source():
    heads = [{"title": "Bitcoin rally lifts crypto sentiment",
              "source": "Bloomberg Markets",
              "link": "https://example.com/bbg-btc",
              "published": None} for _ in range(3)]
    topics = {t["term"]: t for t in extract_topics(heads)}
    assert topics["Bitcoin"]["link"] == "https://example.com/bbg-btc"


def test_mined_links_prefer_non_paywalled_sources():
    heads = ([{"title": "Jio Platforms unveils new data centre plan",
               "source": "FT Markets",
               "link": "https://example.com/ft-jio",
               "published": None} for _ in range(3)]
             + [{"title": "Jio Platforms unveils new data centre plan",
                 "source": "BBC Business",
                 "link": "https://example.com/bbc-jio",
                 "published": None} for _ in range(2)])
    mined = _mine_phrases(heads, known_terms=[])
    jio = [m for m in mined if m["term"].startswith("Jio Platforms")]
    assert jio and jio[0]["link"] == "https://example.com/bbc-jio"
