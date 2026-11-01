"""Tests for the linked word-cloud rendering
(news_cloud_links). The data-layer helpers come from
news_cloud; the links machinery lives in the new module."""
from toolbox.DataZoo.market.news_cloud import extract_topics


def _heads(title, n):
    return [{"title": title, "source": "Test", "link": "",
             "published": None} for _ in range(n)]

def test_cooccurrence_edges_threshold():
    from toolbox.DataZoo.market.news_cloud_links import (
        cooccurrence_edges,
        topic_members,
    )
    heads = (
        _heads("Bitcoin rally lifts crypto sentiment", 2)
        + _heads("Bitcoin and gold both climb today", 1)
        + _heads("Gold steady in quiet trade", 2))
    topics = extract_topics(heads)
    members = topic_members(topics, heads)
    edges = {(a, b): s for a, b, s in cooccurrence_edges(topics, members)}
    # Bitcoin + Crypto share 2 headlines -> linked; Bitcoin + Gold
    # share only 1 -> below the threshold, not linked.
    assert edges.get(("Bitcoin", "Crypto")) == 2
    assert ("Bitcoin", "Gold") not in edges


def test_linked_cloud_html_renders_svg_network():
    from toolbox.DataZoo.market.news_cloud_links import linked_cloud_html
    topics = [{"term": "Midterms", "weight": 18, "group": "macro",
               "example_headline": "x"},
              {"term": "Intel", "weight": 11, "group": "tech",
               "example_headline": "y"},
              {"term": "AI spending", "weight": 10, "group": "tech",
               "example_headline": "z"}]
    edges = [{"a": "Intel", "b": "AI spending", "shared": 4}]
    html = linked_cloud_html(topics, edges, {"left": "As of test"})
    assert "<svg" in html and "<path" in html
    assert "Midterms" in html and "Lines link topics" in html
    # Deterministic: same input, same picture.
    assert linked_cloud_html(topics, edges, {"left": "As of test"}) == html


def test_linked_cloud_html_structure_and_mobile_style():
    from toolbox.DataZoo.market.news_cloud_links import linked_cloud_html
    topics = [{"term": "Midterms", "weight": 18, "group": "macro",
               "example_headline": "x"}]
    html = linked_cloud_html(topics, [], {"left": "As of test"})
    # Title and caption live OUTSIDE the frame: topbar first,
    # then the frame (words only), then the caption row. The
    # old in-frame head/foot divs are gone.
    assert html.index('class="nc-topbar"') < html.index('class="nc-frame"')
    assert html.index('class="nc-frame"') < html.index('class="nc-cap"')
    assert 'class="nc-head"' not in html
    assert 'class="nc-foot"' not in html
    # The phone treatment rides in the same style block, scoped
    # to narrow viewports; the desktop rules stay as-is.
    assert "@media (max-width: 640px)" in html
    assert ".nc-topbar .t { font-size:12px" in html
    assert ".nc-cap { font-size:10.5px" in html


def test_linked_cloud_html_topics_are_clickable():
    from toolbox.DataZoo.market.news_cloud_links import linked_cloud_html
    topics = [{"term": "Midterms", "weight": 18, "group": "macro",
               "example_headline": "Midterms headline",
               "link": "https://example.com/midterms"},
              {"term": "Intel", "weight": 11, "group": "tech",
               "example_headline": "Intel headline"}]
    html = linked_cloud_html(topics, [], {"left": "As of test"})
    assert ('<a href="https://example.com/midterms" '
            'target="_blank"') in html
    assert "<title>Midterms headline</title>" in html
    # No underline marker on the words (removed at the user's
    # direction 2026-11-01) — the link itself is the feature.
    assert 'class="nc-link-line"' not in html
    # A topic without a link stays plain text — no empty anchor.
    assert html.count("<a href=") == 1
    assert "Select a topic to open a source article" in html


def test_word_gradients_carry_the_family_colors():
    from toolbox.DataZoo.market.news_cloud_links import linked_cloud_html
    topics = [{"term": "Gold", "weight": 9, "group": "commodities",
               "example_headline": "x"},
              {"term": "Midterms", "weight": 18, "group": "macro",
               "example_headline": "y"}]
    html = linked_cloud_html(topics, [], caption=None)
    # The words keep their family colors (a gray substitution
    # for faint families was tried and reverted at the user's
    # direction 2026-11-01).
    assert 'stop-color="#D9A821"' in html  # commodities light
    assert 'stop-color="#7A5C00"' in html  # commodities deep
    assert 'stop-color="#173F6E"' in html  # macro deep


def test_cloud_caption_text_uses_the_footnote_gray():
    from toolbox.DataZoo.market.news_cloud_links import linked_cloud_html
    topics = [{"term": "Gold", "weight": 9, "group": "commodities",
               "example_headline": "x"}]
    html = linked_cloud_html(
        topics, [], {"left": 'As of <span class="nc-hl">now</span>'})
    # The muted text around the cloud (as-of line, window
    # note, footer caption) uses the dashboard's canonical
    # muted gray #9AA0A6 — the footnote gray the user picked
    # (2026-11-01) — not the old tan #6F6555.
    assert "#6F6555" not in html
    assert html.count("#9AA0A6") >= 3
