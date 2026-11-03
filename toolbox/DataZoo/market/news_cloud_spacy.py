"""The word cloud's extractor: POS-pattern generation with
spaCy (built as the from-scratch challenger, 2026-11-03;
adopted standalone as the production extractor the same
day, superseding the hybrid union built an hour earlier).

Why a separate module: Streamlit Cloud caches imported
modules across deploys, so new names in a long-imported
module can arrive as ImportError in production (the
2026-10-30 lesson recorded in news_cloud_links) — and spaCy
is an optional dependency. news_cloud imports this module
lazily, inside a fail-soft wrapper: where spaCy or its model
is not installed (or a pool yields nothing), the cloud
falls back to the incumbent pipeline (lexicon + mined
phrases), so the panel never goes blank.

What this layer adds (the head-to-head finding, demos/
news-cloud-sota): the incumbent's miner sees only 2-3 grams
and its unigrams come only from the curated lexicon, so the
pool's dominant ENTITY themes (a person, a country, an
institution mentioned across dozens of stories) are
invisible to it. spaCy noun chunks + named-entity spans are
the literature's candidate front end (TextRank / KPMiner /
TopicRank all generate by POS pattern); this module
generates those candidates, canonicalizes by lemma, pools
sub-spans, and adjudicates them with its own containment
dedup and theme clustering. Adopted as-is: no lexicon
dedupe and no incumbent gate is applied to its candidates —
its weights are distinct-story counts, exactly as built.

Generation rules (ported from the challenger that won the
coverage axis of the 2026-11-03 head-to-head):
  * candidates are noun chunks and entity spans, plus every
    contiguous sub-span (a name inside a longer chunk pools
    its stories instead of splitting below the threshold);
  * a candidate needs >= 3 distinct stories across >= 2
    outlets (vacuous outlet rule in a one-outlet pool);
  * unigrams survive only as names: entity / proper /
    initialism in at least half their occurrences, standing
    alone (not as a longer phrase's tail) in at least half
    their stories, and not in the generic-furniture set —
    the same stance the incumbent's stopwords take toward
    "stock" / "market" / "us";
  * candidates are deduped among themselves by containment
    and by lemma-overlap theme clustering (surface variants
    of one theme carry disjoint story sets, so only lemma
    overlap can merge them).
"""
from collections import Counter, defaultdict

from toolbox.DataZoo.market.news_cloud import (
    _family_of,
    _representative,
)

GENERIC_UNIGRAMS = {"stock", "stocks", "market", "markets",
                    "share", "shares", "today", "news", "day",
                    "week", "live", "update", "updates", "us"}
EDGE_STRIP = {"the", "a", "an", "its", "his", "her", "their",
              "this", "that", "these", "those", "'s", "new"}
MIN_STORIES = 3
MAX_TOPICS = 20

_NLP = None
_NLP_FAILED = False


def _load_nlp():
    """The spaCy model, loaded once per process. Raises on
    any failure — the caller (news_cloud's wrapper) treats
    that as 'layer unavailable'."""
    global _NLP, _NLP_FAILED
    if _NLP is not None:
        return _NLP
    if _NLP_FAILED:
        raise ImportError("spaCy unavailable in this process")
    try:
        import spacy
        _NLP = spacy.load("en_core_web_sm")
    except Exception:
        _NLP_FAILED = True
        raise
    return _NLP


def _norm_lemma(tok):
    """Lemma key form: punctuation inside a token is a
    spelling variant, not a different word ("U.S." and "US"
    are one candidate)."""
    return tok.lemma_.lower().replace(".", "").replace("’", "'")


def _candidate_spans(doc):
    """(lemma_key, surface, is_proper, is_full_span) tuples
    from one headline: noun chunks and named entities,
    edge-stripped, plus every contiguous sub-span."""
    spans = []
    for chunk in doc.noun_chunks:
        spans.append((chunk.start, chunk.end, False))
    for ent in doc.ents:
        if ent.label_ in ("PERSON", "ORG", "GPE", "LOC",
                          "EVENT", "PRODUCT", "NORP"):
            spans.append((ent.start, ent.end, True))
    out = []
    for start, end, is_ent in spans:
        toks = [t for t in doc[start:end]
                if t.text.strip(".,:;!?\"'")]
        while toks and _norm_lemma(toks[0]) in EDGE_STRIP:
            toks = toks[1:]
        while toks and _norm_lemma(toks[-1]) in EDGE_STRIP:
            toks = toks[:-1]
        if not toks or len(toks) > 4:
            continue
        for i in range(len(toks)):
            for j in range(i + 1, len(toks) + 1):
                sub = toks[i:j]
                if _norm_lemma(sub[0]) in EDGE_STRIP \
                        or _norm_lemma(sub[-1]) in EDGE_STRIP:
                    continue
                if any(not any(c.isalnum() for c in t.text)
                       for t in sub):
                    continue
                if not any(t.pos_ in ("NOUN", "PROPN")
                           for t in sub):
                    continue
                key = tuple(_norm_lemma(t) for t in sub)
                if key in (("i",),):
                    continue
                proper = is_ent or any(
                    t.pos_ == "PROPN" for t in sub) or any(
                    t.text.isupper() and len(t.text) > 1
                    for t in sub)
                out.append((key, " ".join(t.text for t in sub),
                            proper, i == 0 and j == len(toks)))
    return out


def spacy_topics(headlines, min_stories=MIN_STORIES,
                 max_topics=MAX_TOPICS):
    """Topic dicts in the cloud's topic shape (term, weight,
    group, canon, surfaces, example/link, _members) for the
    spaCy-generated candidates that clear the bars. Raises
    if spaCy is unavailable — the caller's wrapper converts
    that to the incumbent fallback."""
    nlp = _load_nlp()
    # Work per distinct title (wire duplicates are one
    # story); members are recorded against the FULL headline
    # list, whose indexes the pipeline's recount expects.
    by_title = {}
    for idx, h in enumerate(headlines):
        by_title.setdefault(h["title"], []).append(idx)
    titles = list(by_title)
    docs = list(nlp.pipe(titles))
    cand_members = defaultdict(set)     # key -> headline idxs
    cand_outlets = defaultdict(set)
    cand_surfaces = defaultdict(Counter)
    cand_proper = defaultdict(lambda: [0, 0])
    cand_full = defaultdict(set)        # key -> story titles
    for title, doc in zip(titles, docs):
        per_story = {}
        for key, surface, proper, full in _candidate_spans(doc):
            per_story[key] = surface
            cand_proper[key][1] += 1
            cand_proper[key][0] += 1 if proper else 0
            if full:
                cand_full[key].add(title)
        for key in per_story:
            cand_members[key].update(by_title[title])
            cand_outlets[key].add(
                headlines[by_title[title][0]].get("source", ""))
            cand_surfaces[key][per_story[key]] += 1
    pool_outlets = len({h.get("source", "") for h in headlines})
    cands = []
    for key, members in cand_members.items():
        stories = {headlines[i]["title"] for i in members}
        if len(stories) < min_stories:
            continue
        if pool_outlets >= 2 and len(cand_outlets[key]) < 2:
            continue
        if len(key) == 1:
            if key[0] in GENERIC_UNIGRAMS:
                continue
            proper_n, total_n = cand_proper[key]
            if total_n == 0 or proper_n < 0.5 * total_n:
                continue
            if len(cand_full[key]) < 0.5 * len(stories):
                continue
        surface = cand_surfaces[key].most_common(1)[0][0]
        cands.append({"key": key, "term": surface,
                      "members": set(members),
                      "stories": stories,
                      "weight": len(stories),
                      "surfaces": dict(cand_surfaces[key])})
    # Containment dedup among the candidates themselves.
    cands.sort(key=lambda c: (-c["weight"], -len(c["key"]),
                              c["key"]))
    kept = []
    for c in cands:
        absorbed = False
        for k in kept:
            if set(c["key"]) <= set(k["key"]):
                inside = len(c["stories"] & k["stories"])
                if inside >= 0.8 * len(c["stories"]):
                    absorbed = True
                    break
        if not absorbed:
            kept.append(c)
    # Theme clustering: lemma-overlap >= 0.5 with a weight
    # ratio <= 2.5 merges surface variants of one theme;
    # the label is the shortest surface among the members
    # carrying >= 70% of the top member's weight, members
    # and surfaces pool.
    parent = list(range(len(kept)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(kept)):
        for j in range(i + 1, len(kept)):
            a, b = set(kept[i]["key"]), set(kept[j]["key"])
            overlap = len(a & b) / max(1, min(len(a), len(b)))
            ratio = max(kept[i]["weight"], kept[j]["weight"]) \
                / max(1, min(kept[i]["weight"],
                             kept[j]["weight"]))
            if overlap >= 0.5 and ratio <= 2.5:
                pi, pj = find(i), find(j)
                if pi != pj:
                    parent[pi] = pj
    themes = {}
    for i, c in enumerate(kept):
        themes.setdefault(find(i), []).append(c)
    topics = []
    for members_ in themes.values():
        top = max(m["weight"] for m in members_)
        label = min((m for m in members_
                     if m["weight"] >= 0.7 * top),
                    key=lambda m: (len(m["key"]), m["term"]))
        hits_idx = set().union(*(m["members"]
                                 for m in members_))
        surfaces = {}
        for m in members_:
            for s, cnt in m["surfaces"].items():
                surfaces[s] = surfaces.get(s, 0) + cnt
        hits = [headlines[i] for i in sorted(hits_idx)]
        rep = _representative(hits)
        topics.append({
            "term": label["term"],
            "weight": len({headlines[i]["title"]
                           for i in hits_idx}),
            "example_headline": rep["title"],
            "link": rep.get("link", ""),
            "group": _family_of(label["term"].lower()),
            "mined": True,
            "canon": label["key"],
            "surfaces": surfaces,
            "_members": hits_idx,
        })
    topics.sort(key=lambda t: (-t["weight"], t["term"]))
    return topics[:max_topics]
