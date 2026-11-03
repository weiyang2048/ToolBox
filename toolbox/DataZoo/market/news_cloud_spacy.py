"""The word cloud's extractor: POS-pattern generation with
spaCy (built as the from-scratch challenger, 2026-11-03;
adopted standalone as the production extractor the same
day, superseding the hybrid union built an hour earlier;
superseded in turn by the combined extractor below —
challenger-3, adopted 2026-11-03 after the all-builders
revisit — which keeps this module's generation as one of
its three sources).

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
    LEXICON,
    _canon_of_term,
    _family_of,
    _matches,
    _mine_phrases,
    _representative,
    _story_clusters,
    topic_member_sets,
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


# ------------------------------------------------------------------
# The combined extractor (challenger-3, adopted 2026-11-03)
# ------------------------------------------------------------------
def _display_surface(surfaces):
    """A theme's display label from its pooled surfaces: the
    most frequent capitalized variant when it carries >= 30%
    of the top surface's count; an all-lowercase winner is a
    mid-sentence form and takes title case (the incumbent's
    mined-label house style)."""
    if not surfaces:
        return None
    items = sorted(surfaces.items(), key=lambda kv: (-kv[1], kv[0]))
    top_surface, top_count = items[0]
    caps = [(s, c) for s, c in items if s[:1].isupper()]
    if caps and caps[0][1] >= 0.3 * top_count:
        return caps[0][0]
    if not any(c.isupper() for c in top_surface):
        return top_surface.title()
    return top_surface


def _gather_candidates(headlines, min_weight):
    """Candidate records {key, term, source, surfaces,
    members} from all three generators: the curated lexicon
    (pattern matches), the n-gram miner, and this module's
    spaCy generation (as feedstock, capped higher)."""
    cands = []
    lexicon_terms = []
    for term, patterns, _group in LEXICON:
        hits = {i for i, h in enumerate(headlines)
                if any(_matches(p, h["title"].lower())
                       for p in patterns)}
        if len(hits) < min_weight:
            continue
        lexicon_terms.append(term)
        cands.append({"key": tuple(_canon_of_term(term)),
                      "term": term, "source": "lexicon",
                      "surfaces": {term: len(hits)},
                      "members": hits})
    mined = _mine_phrases(headlines, lexicon_terms)
    mined_members = topic_member_sets(mined, headlines)
    for t in mined:
        cands.append({"key": tuple(t["canon"]),
                      "term": t["term"], "source": "mined",
                      "surfaces": t.get("surfaces") or {},
                      "members": mined_members[t["term"]]})
    for t in spacy_topics(headlines, max_topics=40):
        cands.append({"key": tuple(t["canon"]),
                      "term": t["term"], "source": "spacy",
                      "surfaces": t.get("surfaces") or {},
                      "members": set(t["_members"])})
    return cands


def _cluster_themes(cands):
    """Union-find across sources over three merge rules:
    exact canon; lemma overlap >= 0.5 with a weight ratio
    <= 3.0 (merges disjoint surface variants of one theme);
    and containment (canon subset with >= 80% of the
    smaller candidate's members inside the larger's)."""
    parent = list(range(len(cands)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    weights = [len(c["members"]) for c in cands]
    for i, a in enumerate(cands):
        for j, b in enumerate(cands[i + 1:], i + 1):
            sa, sb = set(a["key"]), set(b["key"])
            if sa == sb:
                union(i, j)
                continue
            overlap = len(sa & sb) / max(1, min(len(sa), len(sb)))
            ratio = max(weights[i], weights[j]) / max(
                1, min(weights[i], weights[j]))
            if overlap >= 0.5 and ratio <= 3.0:
                union(i, j)
                continue
            if sa < sb or sb < sa:
                small, big = (i, j) if sa < sb else (j, i)
                ms = cands[small]["members"]
                mb = cands[big]["members"]
                if ms and len(ms & mb) >= 0.8 * len(ms):
                    union(i, j)
    themes = {}
    for i, c in enumerate(cands):
        themes.setdefault(find(i), []).append(c)
    out = []
    for members_ in themes.values():
        hits = set().union(*(c["members"] for c in members_))
        surfaces = {}
        for c in members_:
            for s, cnt in c["surfaces"].items():
                surfaces[s] = surfaces.get(s, 0) + cnt
        lex = [c for c in members_ if c["source"] == "lexicon"]
        if lex:
            label, label_key = lex[0]["term"], lex[0]["key"]
        else:
            best = max(members_, key=lambda c: len(c["members"]))
            label = _display_surface(best["surfaces"]) \
                or best["term"]
            label_key = best["key"]
        out.append({"term": label, "canon": label_key,
                    "surfaces": surfaces, "members": hits,
                    "mined": any(c["source"] != "lexicon"
                                 for c in members_)})
    return out


def combined_topics(headlines, max_topics=20, min_weight=2):
    """The production extraction (challenger-3, adopted
    2026-11-03): three generators unioned, one cross-source
    theme clustering, weights = the incumbent's story-cluster
    counts over each theme's union members, a lexicon label
    winning its theme, capitalized display surfaces. Raises
    if spaCy is unavailable — the caller's wrapper converts
    that to the incumbent fallback."""
    clusters = _story_clusters(headlines)
    topics = []
    for t in _cluster_themes(_gather_candidates(headlines,
                                                min_weight)):
        weight = len({clusters[i] for i in t["members"]})
        if weight < MIN_STORIES:
            continue
        hits = [headlines[i] for i in sorted(t["members"])]
        rep = _representative(hits)
        topics.append({
            "term": t["term"], "weight": weight,
            "example_headline": rep["title"],
            "link": rep.get("link", ""),
            "group": _family_of(t["term"].lower()),
            "mined": t["mined"], "canon": t["canon"],
            "surfaces": t["surfaces"],
            "_members": t["members"],
        })
    topics.sort(key=lambda t: (-t["weight"], t["term"]))
    return topics[:max_topics]
