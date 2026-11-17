"""Known-value tests for the spaCy topic layer
(news_cloud_spacy.spacy_topics / combined_topics).

Wave 2 of the known-value program: every expectation below
is derived by hand from the module's stated rules (see the
per-corpus derivations in the comments), on corpora whose
words were chosen so en_core_web_sm's parses are the
unambiguous ones — proper names in name positions, regular
plurals, clear nouns. The tests pin the FUNCTIONS' logic
(gates, dedup, clustering, labels, weights, ordering), not
the model's quirks: where a parse fact matters it is stated
in the derivation (e.g. "Federal Reserve" is an ORG span
and a PROPN noun chunk in all four titles).

Availability-gated like the sibling tests in
test_news_cloud.py: skipped when spaCy or en_core_web_sm
is absent.
"""
import json
import os
import subprocess
import sys

import pytest


def _require_spacy():
    pytest.importorskip("spacy")
    try:
        import spacy
        spacy.load("en_core_web_sm")
    except Exception:  # noqa: BLE001 - availability probe: any load failure means skip
        pytest.skip("en_core_web_sm not installed")


def _heads(pairs):
    return [{"title": title, "source": src, "link": "",
             "published": None} for src, title in pairs]


def _topic(term, *, weight, example, group, canon, surfaces,
           members):
    return {"term": term, "weight": weight,
            "example_headline": example, "link": "",
            "group": group, "mined": True, "canon": canon,
            "surfaces": surfaces, "_members": set(members)}


# S1 — two themes of equal weight. "Federal Reserve" is an
# ORG entity span and a PROPN noun chunk in all four of its
# titles, so the key ("federal", "reserve") pools 4 distinct
# stories over 2 outlets. Its unigram sub-spans die at the
# gate: ("reserve",) is never a full span (standing-alone
# rule) and ("federal",) likewise; ("rate",) from "rates"
# appears in one story and is a common noun besides.
# "Zelensky" is a PERSON span / PROPN chunk in its four
# titles: the unigram clears the name-ness bar (proper in
# every occurrence) and the standing-alone bar (it IS the
# full span in every story). Equal weights (4, 4) pin the
# (-weight, term) sort: "Federal Reserve" before "Zelensky".
# Groups come from _family_of: "federal reserve" hits the
# macro cue "fed"; "zelensky" hits no cue -> neutral. The
# representative is the earliest member headline (Wire A/B
# tie at link rank 1).
S1 = _heads([
    ("Wire A", "Federal Reserve holds rates steady"),
    ("Wire B", "Federal Reserve signals patience on inflation"),
    ("Wire A", "Federal Reserve warns of lingering risks"),
    ("Wire B", "Markets shrug as Federal Reserve stands pat"),
    ("Wire A", "Zelensky addresses parliament in Kyiv"),
    ("Wire B", "Zelensky meets European leaders in Berlin"),
    ("Wire A", "Zelensky signs new defense decree"),
    ("Wire B", "Zelensky warns of a hard winter ahead"),
])
S1_FED = _topic("Federal Reserve", weight=4,
                example="Federal Reserve holds rates steady",
                group="macro", canon=("federal", "reserve"),
                surfaces={"Federal Reserve": 4},
                members={0, 1, 2, 3})
S1_ZEL = _topic("Zelensky", weight=4,
                example="Zelensky addresses parliament in Kyiv",
                group="neutral", canon=("zelensky",),
                surfaces={"Zelensky": 4}, members={4, 5, 6, 7})

# S2 — the edge-strip rule applied to a name. EDGE_STRIP
# contains "new", and _candidate_spans strips the SPAN
# before generating sub-spans, so "New York City" (GPE span
# + PROPN chunk in all four titles) enters as the 2-token
# span [York, City]: the only surviving key is
# ("york", "city") with surface "York City". The unigrams
# are never full spans (the stripped span has 2 tokens), so
# the standing-alone rule kills them. One topic exactly.
S2 = _heads([
    ("Wire A", "New York City approves housing plan"),
    ("Wire B", "New York City expands bus lanes"),
    ("Wire A", "New York City reports record tourism"),
    ("Wire B", "New York City opens new shelters"),
])
S2_TOPIC = _topic("York City", weight=4,
                  example="New York City approves housing plan",
                  group="neutral", canon=("york", "city"),
                  surfaces={"York City": 4}, members={0, 1, 2, 3})

# S2B — containment dedup. "Bank of America" is an ORG span
# in all four titles (its noun chunks fragment into [Bank]
# and [America], but the entity span is whole). Candidate
# keys on the same 4 stories: the trigram
# ("bank", "of", "america"), bigrams ("bank", "of") and
# ("of", "america"), unigrams ("bank",) and ("america",)
# (both proper, both full-span in >= half their stories via
# the fragment chunks, so both clear the unigram gates).
# Dedup sorts by (-weight, -len, key): the trigram is kept
# first and every other key is a subset of it with 100% of
# its stories inside -> all absorbed. One topic exactly.
S2B = _heads([
    ("Wire A", "Bank of America lifts forecast"),
    ("Wire B", "Bank of America opens branches"),
    ("Wire A", "Bank of America reports earnings"),
    ("Wire B", "Bank of America warns investors"),
])
S2B_TOPIC = _topic("Bank of America", weight=4,
                   example="Bank of America lifts forecast",
                   group="neutral",
                   canon=("bank", "of", "america"),
                   surfaces={"Bank of America": 4},
                   members={0, 1, 2, 3})

# S3 — theme clustering. ("interest", "rate") pools 4
# stories, ("rate", "hike") pools 3, disjoint headline
# sets. Lemma overlap |{rate}| / min(2, 2) = 0.5 reaches
# MIN_KEY_OVERLAP and the weight ratio 4/3 <= 2.5
# (THEME_MAX_WEIGHT_RATIO), so the two merge into one
# theme. Neither key contains the other, so containment
# dedup leaves both. The unigrams all die on name-ness:
# ("rate",) has 7 stories but proper fraction 0; likewise
# ("interest",) and ("hike",). Label: members carrying
# >= 70% of the top weight (4) are both candidates; the
# shortest key wins, tie on length broken by term ->
# "Interest rates". Weight = 7 distinct titles over the
# union; surfaces pool across the merged candidates.
S3 = _heads([
    ("Wire A", "Interest rates hold steady"),
    ("Wire B", "Interest rates stay high"),
    ("Wire A", "Interest rates worry borrowers"),
    ("Wire B", "Interest rates shape mortgages"),
    ("Wire B", "Rate hikes loom over housing"),
    ("Wire A", "Rate hikes split economists"),
    ("Wire B", "Rate hikes cool demand"),
])
S3_TOPIC = _topic("Interest rates", weight=7,
                  example="Interest rates hold steady",
                  group="macro", canon=("interest", "rate"),
                  surfaces={"Interest rates": 4, "Rate hikes": 3},
                  members={0, 1, 2, 3, 4, 5, 6})

# C1 — the combined extractor. Two themes contribute
# disjoint topics, and each pins a different rule.
# Zelensky (spaCy source only — no LEXICON pattern matches
# any of the 11 titles; the miner is silent because no
# bi/trigram repeats in >= 3 titles): 5 member headlines,
# but headlines 3 and 4 are one story — token Jaccard 7/8
# >= 0.55 and their only differing token "today" has
# document frequency 5 > rare_df 2, so the rare-token veto
# does not fire and _story_clusters merges them. Combined
# weight counts CLUSTERS: 4, where spacy_topics counts
# distinct stories: 5.
# Bitcoin (lexicon + spaCy, exact-canon merge): the
# LEXICON pattern "bitcoin" hits titles 8-10; spaCy also
# generates ("bitcoin",) — PROPN chunk in all three (an
# unambiguous verb "surged" and object positions; the
# model's sentence-initial ADJ reading of "Bitcoin climbs"
# is why the titles avoid that shape), full span in all
# three, so the unigram clears both unigram bars. Equal
# canon keys union the candidates into one theme: the
# lexicon label wins, surfaces pool 3 + 3 = 6, members are
# the same {8, 9, 10} from both sources, weight = 3
# singleton clusters (pairwise Jaccard <= 0.15). Group is
# recomputed by _family_of -> "crypto" either way.
# Every other LEXICON pattern was checked against all 11
# titles by hand (no "volatility", no "inflation"; the
# "oil price" pattern is two words, so "Oil flat" is safe).
C1 = _heads([
    ("Wire A", "Zelensky addresses parliament in Kyiv"),
    ("Wire B", "Zelensky meets European leaders in Berlin"),
    ("Wire A", "Zelensky signs new defense decree"),
    ("Wire B", "Zelensky warns of a hard winter ahead"),
    ("Wire A", "Zelensky warns of a hard winter ahead today"),
    ("Wire B", "Stocks drift today as traders wait"),
    ("Wire A", "Bonds steady today before the auction"),
    ("Wire B", "Oil flat today in quiet trading"),
    ("Wire A", "Bitcoin surged past $90,000 in heavy trading"),
    ("Wire B", "Investors keep buying Bitcoin despite the dip"),
    ("Wire A", "Traders watch Bitcoin before the open"),
])
C1_ZEL_SPACY = _topic("Zelensky", weight=5,
                      example="Zelensky addresses parliament in Kyiv",
                      group="neutral", canon=("zelensky",),
                      surfaces={"Zelensky": 5},
                      members={0, 1, 2, 3, 4})
C1_ZEL_COMBINED = _topic("Zelensky", weight=4,
                         example="Zelensky addresses parliament in Kyiv",
                         group="neutral", canon=("zelensky",),
                         surfaces={"Zelensky": 5},
                         members={0, 1, 2, 3, 4})
C1_BTC_SPACY = _topic("Bitcoin", weight=3,
                      example="Bitcoin surged past $90,000 in heavy trading",
                      group="crypto", canon=("bitcoin",),
                      surfaces={"Bitcoin": 3}, members={8, 9, 10})
C1_BTC_COMBINED = _topic("Bitcoin", weight=3,
                         example="Bitcoin surged past $90,000 in heavy trading",
                         group="crypto", canon=("bitcoin",),
                         surfaces={"Bitcoin": 6}, members={8, 9, 10})

# Pools where nothing can survive: E2's only repeated
# chunk key is ("it",) in 2 stories (< MIN_STORIES 3);
# every other key has 1 story. In E3 the GPE key
# ("harbor", "city") reaches only 2 stories.
E2 = _heads([
    ("Wire A", "He said it again"),
    ("Wire B", "She did it yesterday"),
    ("Wire A", "They went there quietly"),
])
E3 = _heads([
    ("Wire A", "Harbor City approves budget"),
    ("Wire B", "Harbor City cuts ribbon"),
    ("Wire A", "Stocks drift as traders wait"),
])


def test_spacy_topics_two_themes_exact_values():
    _require_spacy()
    from toolbox.DataZoo.market.news_cloud_spacy import spacy_topics
    assert spacy_topics(S1) == [S1_FED, S1_ZEL]


def test_spacy_topics_edge_strip_shortens_name():
    _require_spacy()
    from toolbox.DataZoo.market.news_cloud_spacy import spacy_topics
    assert spacy_topics(S2) == [S2_TOPIC]


def test_spacy_topics_containment_dedup():
    _require_spacy()
    from toolbox.DataZoo.market.news_cloud_spacy import spacy_topics
    assert spacy_topics(S2B) == [S2B_TOPIC]


def test_spacy_topics_theme_clustering_merges_variants():
    _require_spacy()
    from toolbox.DataZoo.market.news_cloud_spacy import spacy_topics
    assert spacy_topics(S3) == [S3_TOPIC]


def test_spacy_topics_gates_and_parameters():
    _require_spacy()
    from toolbox.DataZoo.market.news_cloud_spacy import spacy_topics
    assert spacy_topics([]) == []
    assert spacy_topics(E2) == []
    assert spacy_topics(E3) == []
    # Both S1 themes sit at 4 stories: a floor of 5 empties
    # the result, and max_topics truncates after the sort.
    assert spacy_topics(S1, min_stories=5) == []
    assert spacy_topics(S1, max_topics=1) == [S1_FED]


def test_spacy_topics_on_combined_corpus():
    _require_spacy()
    from toolbox.DataZoo.market.news_cloud_spacy import spacy_topics
    assert spacy_topics(C1) == [C1_ZEL_SPACY, C1_BTC_SPACY]


def test_combined_topics_cluster_weights_and_merge():
    _require_spacy()
    from toolbox.DataZoo.market.news_cloud_spacy import (
        combined_topics,
    )
    assert combined_topics(C1) == [C1_ZEL_COMBINED,
                                   C1_BTC_COMBINED]


def test_combined_topics_min_weight_drops_lexicon_source():
    _require_spacy()
    from toolbox.DataZoo.market.news_cloud_spacy import (
        combined_topics,
    )
    # min_weight=4 gates the lexicon candidate out (3 hits);
    # the Bitcoin theme survives on its spaCy candidate
    # alone: same label (the capitalized display surface),
    # but surfaces no longer pool the lexicon's 3.
    bitcoin = _topic("Bitcoin", weight=3,
                     example="Bitcoin surged past $90,000 in heavy trading",
                     group="crypto", canon=("bitcoin",),
                     surfaces={"Bitcoin": 3}, members={8, 9, 10})
    assert combined_topics(C1, min_weight=4) == [
        C1_ZEL_COMBINED, bitcoin]


def test_combined_topics_edges():
    _require_spacy()
    from toolbox.DataZoo.market.news_cloud_spacy import (
        combined_topics,
    )
    assert combined_topics([]) == []
    # Two headlines: no generator can reach the
    # MIN_STORIES cluster floor.
    assert combined_topics(C1[:2]) == []
    assert combined_topics(C1, max_topics=1) == [C1_ZEL_COMBINED]


def _slim(topics):
    out = []
    for t in topics:
        d = dict(t)
        d["canon"] = list(d["canon"])
        d["_members"] = sorted(d["_members"])
        out.append(d)
    return out


def _dump():
    from toolbox.DataZoo.market.news_cloud_spacy import (
        combined_topics,
        spacy_topics,
    )
    return json.dumps({
        "s1": _slim(spacy_topics(S1)),
        "s3": _slim(spacy_topics(S3)),
        "c1_spacy": _slim(spacy_topics(C1)),
        "c1_combined": _slim(combined_topics(C1)),
    }, sort_keys=True)


_CHILD_DUMP = (
    "import importlib.util, sys\n"
    "spec = importlib.util.spec_from_file_location("
    "'spacy_test_mod', sys.argv[1])\n"
    "mod = importlib.util.module_from_spec(spec)\n"
    "spec.loader.exec_module(mod)\n"
    "print(mod._dump())\n"
)


def test_extraction_deterministic_across_processes():
    # Set/dict iteration order must not leak into the
    # output: the same extraction in fresh processes under
    # different hash seeds is byte-identical to the
    # in-process result (and to an in-process repeat).
    _require_spacy()
    from toolbox.DataZoo.market.news_cloud_spacy import spacy_topics
    assert spacy_topics(S1) == spacy_topics(S1)
    expected = _dump()
    root = os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))
    for seed in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        out = subprocess.run(
            [sys.executable, "-c", _CHILD_DUMP,
             os.path.abspath(__file__)],
            capture_output=True, text=True, env=env, cwd=root,
            check=False)
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == expected
