"""News word cloud: date parsing, topic extraction (lexicon +
mining, boilerplate filter), and the embeddable frame HTML.
All offline — synthetic headlines only."""
from toolbox.DataZoo.market.news_cloud import (
    LEXICON,
    _canon_of_term,
    _display,
    _family_of,
    _fold_containment,
    _fold_windows,
    _join_windows,
    _matches,
    _mine_phrases,
    _parse_date,
    _paywalled,
    _shape_ok,
    _story_clusters,
    _template_bound,
    _tokens,
    cloud_html,
    extract_topics,
    singularize,
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
    # Distinct headlines per topic: since 2026-11-02 weights
    # count story clusters, so identical headlines are one story.
    heads = (_heads("Federal Reserve holds rates steady as inflation cools", 1)
             + _heads("Federal Reserve officials signal patience on rate moves", 1)
             + _heads("Markets rally after Federal Reserve decision surprises traders", 1)
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


def test_singularize_rules_and_protections():
    # Regular plurals fold to the singular.
    assert singularize("midterms") == "midterm"
    assert singularize("elections") == "election"
    assert singularize("tariffs") == "tariff"
    assert singularize("rallies") == "rally"
    assert singularize("dismisses") == "dismiss"
    # Meaning-bearing tokens are never touched: entities and
    # figures keep their exact form (S&P 500 stays S&P 500).
    assert singularize("s&p") == "s&p"
    assert singularize("at&t") == "at&t"
    assert singularize("m&a") == "m&a"
    assert singularize("500") == "500"
    # Not plurals despite the trailing s; too-short tokens are
    # left alone as well.
    assert singularize("news") == "news"
    assert singularize("business") == "business"
    assert singularize("gas") == "gas"


def test_canon_key_protects_sp_500():
    assert _canon_of_term("S&P 500") == ("s&p", "500")
    assert _canon_of_term("Midterms") == ("midterm",)
    assert _canon_of_term("Treasury yields") == ("treasury", "yield")


def test_mining_pools_plural_surface_variants():
    heads = (_heads("Senate race tightens in Michigan", 1)
             + _heads("Senate race draws millions", 1)
             + _heads("Senate races heat up", 1))
    topics = {t["term"]: t for t in extract_topics(heads)}
    # One topic for both surface forms: the weight is the union
    # of their headlines, the label the most frequent surface.
    assert topics["Senate Race"]["weight"] == 3
    assert topics["Senate Race"]["surfaces"] == {
        "senate race": 2, "senate races": 1}
    assert "Senate Races" not in topics


def test_containment_fold_merges_midterm_fragment():
    heads = (_heads("Midterm elections draw record turnout", 1)
             + _heads("Midterm elections reshape Congress", 1)
             + _heads("Midterm elections test both parties", 1)
             + _heads("Midterms could flip the House", 1))
    topics = extract_topics(heads)
    # The mined "Midterm Elections" (3 headlines, every one of
    # them inside the lexicon topic's 4) folds into "Midterms":
    # one topic, weight = the union (4), not the sum (7).
    assert [t["term"] for t in topics] == ["Midterms"]
    assert topics[0]["weight"] == 4
    assert set(topics[0]["merged_from"]) == {
        "Midterms", "Midterm Elections"}


def _fold_pair(members_x, members_y, wx=10, wy=7):
    topics = [
        {"term": "Midterms", "weight": wx, "group": "macro",
         "canon": ("midterm",), "example_headline": "", "link": ""},
        {"term": "Midterm Elections", "weight": wy, "group": "neutral",
         "canon": ("midterm", "election"),
         "example_headline": "", "link": ""},
    ]
    heads = [{"title": "H{}".format(i), "source": "Test", "link": "",
              "published": None} for i in range(12)]
    members = {"Midterms": members_x, "Midterm Elections": members_y}
    return _fold_containment(topics, heads, members)


def test_fold_guard_blocks_low_containment():
    # Token-subset pair, but only 5 of the smaller topic's 7
    # headlines sit inside the larger's (71% < 80%): no fold —
    # token inclusion alone must not swallow a topic.
    out = _fold_pair(set(range(10)), {0, 1, 2, 3, 4, 10, 11})
    assert [t["term"] for t in out] == [
        "Midterms", "Midterm Elections"]


def test_fold_merges_at_full_containment():
    out = _fold_pair(set(range(10)), set(range(7)))
    assert [t["term"] for t in out] == ["Midterms"]
    assert out[0]["weight"] == 10  # the union, not 10 + 7


def test_fold_merges_at_the_80_percent_boundary():
    # Exactly 80% of the smaller set (4 of 5) contained: folds.
    out = _fold_pair({0, 1, 2, 3, 4}, {0, 1, 2, 3, 5}, wx=5, wy=5)
    assert len(out) == 1
    assert out[0]["weight"] == 6


def test_join_windows_overlap_only():
    assert _join_windows(("what", "is", "behind"),
                         ("behind", "the", "move")) == (
        "what", "is", "behind", "the", "move")
    # Reverse orientation joins the same way.
    assert _join_windows(("behind", "the", "move"),
                         ("what", "is", "behind")) == (
        "what", "is", "behind", "the", "move")
    # Pure containment is not a window join (the containment
    # fold owns it): the join would not extend the longer gram.
    assert _join_windows(("midterm",), ("midterm", "election")) is None
    # No shared edge token: no join.
    assert _join_windows(("senate", "race"), ("wall", "street")) is None


def _window_topics():
    return [
        {"term": "Silver Lining", "weight": 3, "group": "neutral",
         "canon": ("silver", "lining"), "mined": True,
         "example_headline": "", "link": "",
         "surfaces": {"silver lining": 3}},
        {"term": "Lining Up", "weight": 3, "group": "neutral",
         "canon": ("lining", "up"), "mined": True,
         "example_headline": "", "link": "",
         "surfaces": {"lining up": 3}},
    ]


def test_window_fold_merges_template_fragments():
    # Every headline carries both windows, joined contiguously:
    # one topic, the joined phrase, weight = the union.
    heads = _heads("The Silver Lining Up North", 3)
    out = _fold_windows(_window_topics(), heads, [])
    assert [t["term"] for t in out] == ["Silver Lining Up"]
    assert out[0]["weight"] == 3
    assert set(out[0]["merged_from"]) == {"Silver Lining", "Lining Up"}
    assert "silver lining up" in out[0]["surfaces"]


def test_window_fold_blocked_below_containment():
    # The windows share only 3 of their 5 headlines each (60% <
    # the 80% guard): token overlap alone must not merge them.
    heads = (_heads("Silver Lining Up For Grabs", 3)
             + _heads("Silver Lining For Miners", 2)
             + _heads("Lining Up For Tickets", 2))
    out = _fold_windows(_window_topics(), heads, [])
    assert {t["term"] for t in out} == {"Silver Lining", "Lining Up"}


def test_window_fold_blocked_when_not_contiguous():
    # Identical headline sets, but the joined phrase never
    # occurs contiguously — the two grams are separate phrases
    # that merely share a word: no fold.
    heads = _heads("Silver Lining Appears While Lining Up Forms", 3)
    out = _fold_windows(_window_topics(), heads, [])
    assert {t["term"] for t in out} == {"Silver Lining", "Lining Up"}


def test_window_fold_merges_live_template():
    # The live 2026-11-01 case end to end: Yahoo's explainer
    # template mined as two overlapping windows of one phrase.
    heads = [
        {"title": "What Is Behind The Move In Cameco Stock?",
         "source": "Yahoo ^GSPC", "published": None},
        {"title": "What Is Behind The Move In Robinhood Stock?",
         "source": "Yahoo ^GSPC", "published": None},
        {"title": "What Is Behind The Move In Global Payments Stock?",
         "source": "Yahoo ^GSPC", "published": None},
    ]
    topics = {t["term"]: t for t in extract_topics(heads)}
    assert topics["What Is Behind The Move"]["weight"] == 3
    assert "What Is Behind" not in topics
    assert "Behind The Move" not in topics


def test_window_fold_blocked_by_fragment_gate():
    # The windows' join ("Broader Market Some") ends on a
    # determiner — a fragment the shape gate rejects — so the
    # fold is an ordinary miss and both windows survive.
    topics = [
        {"term": "Broader Market", "weight": 3, "group": "neutral",
         "canon": ("broader", "market"), "mined": True,
         "example_headline": "", "link": "",
         "surfaces": {"broader market": 3}},
        {"term": "Market Some", "weight": 3, "group": "neutral",
         "canon": ("market", "some"), "mined": True,
         "example_headline": "", "link": "",
         "surfaces": {"market some": 3}},
    ]
    heads = _heads("Broader Market Some Facts Emerge", 3)
    out = _fold_windows(topics, heads, [])
    assert {t["term"] for t in out} == {
        "Broader Market", "Market Some"}


# ------------------------------------------------------------------
# Robustness gates (2026-11-02): shape, outlet, template, stories
# ------------------------------------------------------------------

def test_shape_gate_units():
    assert not _shape_ok(("their", "own"))       # possessive edge
    assert not _shape_ok(("make", "final"))      # verb + adjective
    assert not _shape_ok(("$20", "billion"))     # bare quantity
    assert _shape_ok(("oil", "prices"))
    assert _shape_ok(("strait", "of", "hormuz"))  # interior stopword
    # The window fold owns question/preposition edges: its
    # adjudicated phrases must pass the gate.
    assert _shape_ok(("behind", "the", "move"))


def test_shape_gate_fragment_shapes_2026_11_02():
    # The three fragment topics that rendered live on
    # 2026-11-02, checked on their canonical forms.
    from toolbox.DataZoo.market.news_cloud import _canon_of_term
    assert not _shape_ok(_canon_of_term("Least 24"))   # number tail
    assert not _shape_ok(_canon_of_term("Some Facts"))  # determiner
    assert not _shape_ok(_canon_of_term("30 Years"))   # bare duration
    assert not _shape_ok(("least", "24"))
    assert not _shape_ok(("some", "fact"))
    assert not _shape_ok(("30", "year"))
    # Headed phrases with numbers/durations inside survive.
    assert _shape_ok(("s&p", "500"))          # & head, number tail
    assert _shape_ok(("10", "year", "yield"))  # duration + noun head
    assert _shape_ok(("30", "year", "treasury"))


def test_shape_gate_filters_mined_fragments():
    heads = (_heads("Funds Make Final Push Into Bonds", 1)
             + _heads("Banks Make Final Push Into Loans", 1)
             + _heads("Funds Make Final Call On Bonds", 1)
             + _heads("Insurers Hold Their Own Reserves", 3)
             + _heads("Mega Fund Raises $20 Billion For Buyouts", 3)
             + _heads("Chile Copper Exports Climb Again", 3))
    terms = {t["term"] for t in _mine_phrases(heads, [])}
    assert "Make Final" not in terms
    assert "Their Own" not in terms
    assert "$20 Billion" not in terms
    # A headed noun phrase survives (the window fold may extend
    # it to the maximal phrase).
    assert any(t.startswith("Chile Copper") for t in terms)


def test_outlet_gate():
    one = _heads("Chile Copper Exports Climb", 3, source="A")
    other = _heads("Local Bakery Wins Sourdough Award", 3,
                   source="B")
    terms = {t["term"] for t in _mine_phrases(one + other, [])}
    assert not any("Chile Copper" in t for t in terms)  # 1 outlet
    two = one + _heads("Chile Copper Exports Climb", 2, source="B")
    terms = {t["term"] for t in _mine_phrases(two, [])}
    assert any(t.startswith("Chile Copper") for t in terms)


def test_template_demotion():
    titles = ["Acme stock hits 52-week low at 10.00 USD",
              "Beta Corp stock hits 52-week low at 22.50 USD",
              "Gamma Inc stock hits 52-week low at 3.75 USD",
              "Delta LLC stock hits 52-week low at 8.10 USD"]
    assert _template_bound(titles)
    heads = [{"title": t, "source": "Wire", "published": None}
             for t in titles]
    terms = {t["term"] for t in extract_topics(heads)}
    assert "Hits 52-Week Low" not in terms
    # A repeated story keeps its entities: not template-bound.
    same_story = ["Fed cuts rates amid inflation fight",
                  "Fed cuts rates amid inflation fight",
                  "Fed cuts rates amid inflation fight"]
    assert not _template_bound(same_story)


def test_story_clusters_rare_token_veto():
    pair = [{"title": "US Equity Indexes Rise Amid Sinking Crude "
                      "Oil Treasury Yields", "source": "Y"},
            {"title": "US Equity Indexes Rise Amid Sinking Crude "
                      "Oil", "source": "Y"}]
    siblings = [{"title": "What Is Behind The Move In Roblox "
                          "Stock?", "source": "Y"},
                {"title": "What Is Behind The Move In Kinross "
                          "Gold Stock?", "source": "Y"}]
    # Padding lifts the shared market vocabulary out of the
    # rare band, as in a real pool.
    padding = _heads("Treasury yields steady as investors await "
                     "data", 12)
    heads = pair + siblings + padding
    clusters = _story_clusters(heads)
    assert clusters[0] == clusters[1]   # one story, counted once
    assert clusters[2] != clusters[3]   # template siblings stay apart


def test_fold_fragment_gate_shapes():
    # The three fragments that rendered live on 2026-11-02
    # ("Kills 26", "Dip Than Broader", "Broader Market Some")
    # each pass one of the shape gate's old holes; the gate
    # now closes all three, and the adjudicated phrases the
    # window fold owns still pass.
    assert not _shape_ok(_canon_of_term("Kills 26"))
    assert not _shape_ok(("kill", "26"))        # verb + count
    assert not _shape_ok(_canon_of_term("Dip Than Broader"))
    assert not _shape_ok(("dip", "than", "broader"))
    assert not _shape_ok(_canon_of_term("Broader Market Some"))
    assert not _shape_ok(("broader", "market", "some"))
    assert _shape_ok(_canon_of_term("What Is Behind The Move"))
    assert _shape_ok(("s&p", "500"))  # & head keeps number tail


def test_fold_fragment_gate_filters_live_fragments():
    # End to end on the live fragments' own headline families
    # (the Yahoo explainer template + the Ukraine count
    # headlines, verbatim patterns from the 2026-11-02 pool).
    y = "Yahoo ^GSPC"
    heads = [
        {"title": "AeroVironment (AVAV) Sees a More Significant "
                  "Dip Than Broader Market: Some Facts to Know",
         "source": y, "published": None},
        {"title": "Wix.com (WIX) Sees a More Significant Dip "
                  "Than Broader Market: Some Facts to Know",
         "source": "Yahoo ^IXIC", "published": None},
        {"title": "Intrusion Inc. (INTZ) Sees a More Significant "
                  "Dip Than Broader Market: Some Facts to Know",
         "source": y, "published": None},
        {"title": "Russian bombing kills 26 people at Ukraine "
                  "aid centre, Kyiv says", "source": "Wire UK",
         "published": None},
        {"title": "Russian bombing of Ukrainian aid centre "
                  "kills 26 people", "source": "Wire CA",
         "published": None},
        {"title": "Russia kills 26 in strike on Ukraine aid "
                  "centre", "source": "Wire US", "published": None},
    ]
    terms = {t["term"] for t in _mine_phrases(heads, [])}
    assert "Kills 26" not in terms
    assert "Dip Than Broader" not in terms
    assert "Broader Market Some" not in terms


def test_ahead_of_earnings_is_scaffolding_not_a_topic():
    # 2026-11-02, user-caught on the live cloud: "ahead" was
    # missing from STOPWORDS, so the trailing PP of earnings-
    # preview headlines ("... on Apple ahead of earnings")
    # formed a gram and rendered as a topic.
    heads = (_heads("Acme Shares Drop Ahead of Earnings", 2,
                    source="A")
             + _heads("Beta Stock Slides Ahead of Earnings", 1,
                      source="B"))
    terms = {t["term"] for t in _mine_phrases(heads, [])}
    assert not any("Ahead" in t for t in terms)


def test_shape_gate_rejects_closed_class_fragments():
    # 2026-11-03: "What Does" rendered live as a topic — a
    # gram of two function words (wh-word + auxiliary) whose
    # tokens sat in none of the gate's enumerative lists, so
    # each read as noun-like. Closed-class tokens can no
    # longer be a phrase's content head.
    from toolbox.DataZoo.market.news_cloud import _shape_ok
    assert not _shape_ok(("what", "does"))
    assert not _shape_ok(("how", "does"))
    assert not _shape_ok(("who", "will"))
    # The window fold's adjudicated question phrase still
    # passes the shared gate on its content head.
    assert _shape_ok(("what", "is", "behind", "the", "move"))


def test_mining_rejects_question_fragments():
    from toolbox.DataZoo.market.news_cloud import _mine_phrases
    heads = []
    for src, tail in (("A", "jobs report"), ("B", "inflation data"),
                      ("C", "tariff ruling"), ("A", "dollar slide")):
        heads.append({"title": f"What does the {tail} mean for the Fed",
                      "source": src, "link": "", "published": None})
    for src, tail in (("A", "on export hopes"),
                      ("B", "as demand grows"),
                      ("C", "after earnings")):
        heads.append({"title": "Semiconductor shares climb " + tail,
                      "source": src, "link": "", "published": None})
    mined = _mine_phrases(heads, [], top_n=10, min_weight=3)
    terms = [t["term"] for t in mined]
    assert "What Does" not in terms
    assert not any(t.startswith("What") for t in terms)
    assert any(t.startswith("Semiconductor Shares") for t in terms)


def _hybrid_pool():
    """An entity theme neither the lexicon nor the n-gram
    miner can produce: a person named across stories with
    varying verbs (no shared bigram), plus filler."""
    heads = []
    for src, title in (
            ("Wire A", "Zelensky addresses parliament in Kyiv"),
            ("Wire B", "Zelensky meets European leaders in Berlin"),
            ("Wire A", "Zelensky signs new defense decree"),
            ("Wire B", "Zelensky warns of a hard winter ahead"),
            ("Wire A", "Stocks drift as traders await data"),
            ("Wire B", "Bonds steady ahead of the auction")):
        heads.append({"title": title, "source": src,
                      "link": "", "published": None})
    return heads


def _block_spacy_module():
    import sys
    saved = sys.modules.get(
        "toolbox.DataZoo.market.news_cloud_spacy")
    sys.modules["toolbox.DataZoo.market.news_cloud_spacy"] = None
    return saved


def _restore_spacy_module(saved):
    import sys
    if saved is None:
        del sys.modules["toolbox.DataZoo.market.news_cloud_spacy"]
    else:
        sys.modules["toolbox.DataZoo.market.news_cloud_spacy"] = saved


def test_spacy_layer_failsoft_without_module():
    # With the spaCy module unimportable, the layer reports
    # None — the signal the production entry falls back on.
    import toolbox.DataZoo.market.news_cloud as nc
    saved = _block_spacy_module()
    try:
        assert nc._combined_layer(_hybrid_pool()) is None
    finally:
        _restore_spacy_module(saved)


def test_challenger_entry_combined_extraction():
    import pytest
    pytest.importorskip("spacy")
    try:
        import spacy
        spacy.load("en_core_web_sm")
    except Exception:
        pytest.skip("en_core_web_sm not installed")
    from toolbox.DataZoo.market.news_cloud import extract_topics_challenger
    topics = extract_topics_challenger(_hybrid_pool(),
                                       max_topics=20,
                                       min_weight=3)
    # The combined extractor on an entity-only pool: exactly
    # the entity theme — no lexicon or mined topic fires.
    assert [t["term"] for t in topics] == ["Zelensky"]


def test_challenger_entry_falls_back_without_spacy():
    import toolbox.DataZoo.market.news_cloud as nc
    saved = _block_spacy_module()
    try:
        pool = _hybrid_pool()
        assert nc.extract_topics_challenger(
            pool, max_topics=20, min_weight=3) == nc.extract_topics(
            pool, max_topics=20, min_weight=3)
    finally:
        _restore_spacy_module(saved)


def test_member_sets_honor_carried_members():
    # A topic carrying its exact member set is matched by
    # construction, not by substring: "ai" must not match
    # the "ai" inside "said".
    from toolbox.DataZoo.market.news_cloud import topic_member_sets
    heads = [{"title": "He said the plan again", "source": "A",
              "link": "", "published": None},
             {"title": "AI spending rises", "source": "B",
              "link": "", "published": None}]
    topics = [{"term": "AI", "weight": 1, "group": "tech",
               "canon": ("ai",), "mined": True,
               "_members": {1}}]
    members = topic_member_sets(topics, heads)
    assert members["AI"] == {1}


def test_combined_merges_lexicon_and_spacy_sources():
    import pytest
    pytest.importorskip("spacy")
    try:
        import spacy
        spacy.load("en_core_web_sm")
    except Exception:
        pytest.skip("en_core_web_sm not installed")
    from toolbox.DataZoo.market.news_cloud import extract_topics_challenger
    heads = _hybrid_pool()
    for src, title in (
            ("Wire A", "Binance freezes withdrawals after hack"),
            ("Wire B", "Binance says funds are safe after breach"),
            ("Wire A", "Binance halts trading amid investigation")):
        heads.append({"title": title, "source": src,
                      "link": "", "published": None})
    topics = extract_topics_challenger(heads, max_topics=20,
                                       min_weight=3)
    terms = [t["term"] for t in topics]
    # One theme per source strength: the lexicon's Binance
    # and the spaCy layer's Zelensky both present, once each.
    assert "Binance" in terms and "Zelensky" in terms
    assert terms.count("Binance") == 1


def test_family_of_politics_and_short_cues():
    # The 2026-11-04 mono-color fix: politics has its own
    # family, and two-letter cues respect word boundaries.
    assert _family_of("trump") == "politics"
    assert _family_of("democrats sweep senate") == "politics"
    assert _family_of("iran israel tensions") == "politics"
    assert _family_of("midterms") == "politics"
    assert _family_of("trump tariffs") == "politics"
    assert _family_of("strait of hormuz") != "tech"
    assert _family_of("ai") == "tech"
    assert _family_of("ai stocks rally") == "tech"


def test_headline_tone_and_gradient():
    from toolbox.DataZoo.market.news_cloud import (
        headline_tone, tone_gradient, tone_colored)
    assert headline_tone("Stocks rally as profits surge") > 0
    assert headline_tone("Selloff deepens amid crisis fears") < 0
    assert headline_tone("Senate panel meets Tuesday") == 0
    assert headline_tone("Sell-off worsens") < 0  # hyphen folds
    light, deep = tone_gradient(0.0)
    assert deep == "#000000"
    assert tone_gradient(0.8)[1] == "#1A9E4A"
    assert tone_gradient(-0.8)[1] == "#D62820"
    assert tone_gradient(99.0) == tone_gradient(0.8)  # clamps
    import os
    os.environ["NEWS_CLOUD_TONE_MODEL"] = "off"
    heads = [{"title": "AI stocks rally", "link": ""},
             {"title": "AI rally continues", "link": ""}]
    topics = [{"term": "AI", "weight": 2, "group": "tech",
               "canon": "ai", "example_headline": heads[0]["title"],
               "link": ""}]
    out = tone_colored(topics, heads)
    assert out[0]["group"] == "tech"  # family stays metadata
    assert out[0]["tone"] > 0 and out[0]["tone_color"] != "#000000"


def test_tone_colored_prefers_the_model(monkeypatch):
    from toolbox.DataZoo.market.news_cloud import tone_gradient
    import toolbox.DataZoo.market.news_cloud_tone as tone_mod
    monkeypatch.setattr(
        tone_mod, "model_headline_tones",
        lambda titles: [-0.9] * len(titles))
    from toolbox.DataZoo.market.news_cloud import tone_colored
    heads = [{"title": "AI panel meets on Tuesday", "link": ""},
             {"title": "AI hearing scheduled", "link": ""}]
    topics = [{"term": "AI", "weight": 2, "group": "tech",
               "canon": "ai",
               "example_headline": "AI panel meets on Tuesday",
               "link": ""}]
    out = tone_colored(topics, heads)
    assert out[0]["tone"] == -0.9
    from toolbox.DataZoo.market.news_cloud import TONE_CAP_MODEL
    assert out[0]["tone_color"] == tone_gradient(
        -0.9, cap=TONE_CAP_MODEL)[1]
