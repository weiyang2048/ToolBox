"""Tests for the linked word-cloud rendering
(news_cloud_links). The data-layer helpers come from
news_cloud; the links machinery lives in the new module."""
from toolbox.DataZoo.market.news_cloud import extract_topics


def _heads(title, n):
    return [{"title": title, "source": "Test", "link": "",
             "published": None} for _ in range(n)]

def test_cooccurrence_edges_threshold():
    from toolbox.DataZoo.market.news_cloud_links import (
        cooccurrence_edges, topic_members)
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
