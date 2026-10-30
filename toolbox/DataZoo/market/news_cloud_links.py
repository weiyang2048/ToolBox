"""The linked rendering of the market-news word cloud: topics as
positioned words, curved lines between topics that share
headlines. The data layer (fetch, extraction, sizes, styles)
stays canonical in news_cloud; this module holds only the
co-occurrence machinery and the SVG network rendering, and
imports long-standing names from news_cloud.

Why a separate module (live lesson, 2026-10-30): Streamlit
Cloud re-reads page scripts on every run but caches imported
modules in the running process across deploys, so new names
added to a long-imported module can arrive as ImportError in
production while everything works locally (it bit the app's
own config module in the morning and news_cloud in the
afternoon). A brand-new module has no stale cache entry and
imports fresh. New shared helpers go in new module files.
"""
import math
from datetime import datetime, timezone

from toolbox.DataZoo.market.news_cloud import (
    FAMILY_GRADIENTS,
    LEXICON,
    _FRAME_CSS,
    _matches,
    _sizes,
    extract_topics,
    fetch_headlines,
)


# ------------------------------------------------------------------
# Links: co-occurrence edges + the linked (SVG) rendering
# ------------------------------------------------------------------
LINK_MIN_SHARED = 2
LINK_MAX_EDGES = 22
LINK_PANEL_W, LINK_PANEL_H = 1124, 500


def topic_members(topics, headlines):
    """term -> set of headline indexes mentioning it. Lexicon
    topics match by their patterns; mined phrases by their words
    appearing, in order, in the headline."""
    lex = {term: patterns for term, patterns, _ in LEXICON}
    members = {}
    for t in topics:
        term = t["term"]
        hits = set()
        if term in lex:
            pats = lex[term]
            for i, h in enumerate(headlines):
                if any(_matches(p, h["title"].lower()) for p in pats):
                    hits.add(i)
        else:
            gram = term.lower().replace("-", " ")
            for i, h in enumerate(headlines):
                title = " ".join(h["title"].lower().replace("-", " ").split())
                if gram in title:
                    hits.add(i)
        members[term] = hits
    return members


def cooccurrence_edges(topics, members, min_shared=LINK_MIN_SHARED,
                       max_edges=LINK_MAX_EDGES):
    """(term_a, term_b, shared) triples: pairs of topics sharing
    at least `min_shared` distinct headlines — one shared headline
    is usually a roundup name-checking everything, so it counts
    as noise. Strongest first, capped at `max_edges`."""
    terms = [t["term"] for t in topics]
    edges = []
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            shared = len(members[terms[i]] & members[terms[j]])
            if shared >= min_shared:
                edges.append((terms[i], terms[j], shared))
    edges.sort(key=lambda e: (-e[2], e[0], e[1]))
    return edges[:max_edges]


def _linked_layout(topics, edges, sizes):
    """Deterministic spread layout for the linked cloud: greedy
    max-min placement over the whole panel, then ellipse-aware
    repulsion (a word clears its full box plus a margin, not just
    its center) with weak edge springs and weak gravity, and a
    padded de-overlap pass. Tuned through four demo rounds with
    the user on 2026-10-30: spiral start clustered (v1), box
    repulsion spread it (v2), wider still (v3, wired first),
    then tightened back at his word — shorter panel, smaller
    margin, stronger springs (v4, the settled form)."""
    margin = 20.0
    nodes = {}
    placed = []
    grid = [(gx, gy) for gx in range(40, LINK_PANEL_W - 39, 40)
            for gy in range(30, LINK_PANEL_H - 29, 30)]
    for t in topics:
        term = t["term"]
        px = sizes[term]
        w, h = 0.60 * px * len(term) + 14, px * 1.10 + 8
        best, best_score = None, -1.0
        for gx, gy in grid:
            if gx < w / 2 or gx > LINK_PANEL_W - w / 2 or \
                    gy < h / 2 or gy > LINK_PANEL_H - h / 2:
                continue
            score = min((math.hypot(gx - qx, gy - qy)
                         for qx, qy in placed), default=1e9)
            score -= 0.12 * math.hypot(gx - LINK_PANEL_W / 2,
                                       gy - LINK_PANEL_H / 2) \
                * (1 if not placed else 0)
            if score > best_score:
                best, best_score = (gx, gy), score
        x, y = best if best else (LINK_PANEL_W / 2, LINK_PANEL_H / 2)
        placed.append((x, y))
        nodes[term] = {"x": float(x), "y": float(y), "vx": 0.0,
                       "vy": 0.0, "w": w, "h": h}
    strength = {}
    for a, b, s in edges:
        strength[(a, b)] = strength[(b, a)] = s
    terms = list(nodes)
    for _ in range(500):
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                a, b = nodes[terms[i]], nodes[terms[j]]
                dx, dy = b["x"] - a["x"], b["y"] - a["y"]
                rx = (a["w"] + b["w"]) / 2 + margin
                ry = (a["h"] + b["h"]) / 2 + margin
                rho = math.hypot(dx / rx, dy / ry) or 0.01
                if rho < 1.7:
                    push = (1.7 - rho) * 3.2
                    ux, uy = dx / (rho * rx), dy / (rho * ry)
                    a["vx"] -= push * ux; a["vy"] -= push * uy
                    b["vx"] += push * ux; b["vy"] += push * uy
                s = strength.get((terms[i], terms[j]))
                if s:
                    d = math.hypot(dx, dy) or 1.0
                    pull = 0.0032 * s * (d - rx * 1.5)
                    fx, fy = pull * dx / d, pull * dy / d
                    a["vx"] += fx; a["vy"] += fy
                    b["vx"] -= fx; b["vy"] -= fy
        for t in terms:
            n = nodes[t]
            n["vx"] += (LINK_PANEL_W / 2 - n["x"]) * 0.003
            n["vy"] += (LINK_PANEL_H / 2 - n["y"]) * 0.0045
            n["vx"] *= 0.84; n["vy"] *= 0.84
            n["x"] += n["vx"]; n["y"] += n["vy"]
            n["x"] = min(max(n["x"], n["w"] / 2), LINK_PANEL_W - n["w"] / 2)
            n["y"] = min(max(n["y"], n["h"] / 2), LINK_PANEL_H - n["h"] / 2)
    for _ in range(80):
        moved = False
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                a, b = nodes[terms[i]], nodes[terms[j]]
                ox = (a["w"] + b["w"]) / 2 + 12 - abs(b["x"] - a["x"])
                oy = (a["h"] + b["h"]) / 2 + 8 - abs(b["y"] - a["y"])
                if ox > 0 and oy > 0:
                    moved = True
                    if ox < oy:
                        push = ox / 2 * (1 if b["x"] >= a["x"] else -1)
                        a["x"] -= push; b["x"] += push
                    else:
                        push = oy / 2 * (1 if b["y"] >= a["y"] else -1)
                        a["y"] -= push; b["y"] += push
                    for n in (a, b):
                        n["x"] = min(max(n["x"], n["w"] / 2),
                                     LINK_PANEL_W - n["w"] / 2)
                        n["y"] = min(max(n["y"], n["h"] / 2),
                                     LINK_PANEL_H - n["h"] / 2)
        if not moved:
            break
    return nodes


def linked_cloud_html(topics, edges, caption=None):
    """The linked cloud as embeddable HTML: the fancy frame with
    an SVG panel — words at computed positions, curved lines
    between topics that share headlines. `edges` are (a, b,
    shared) triples or build_cloud's edge dicts. The frame's
    right footer line is set to the links explanation."""
    import html as html_mod
    triples = [(e["a"], e["b"], e["shared"]) if isinstance(e, dict)
               else tuple(e) for e in edges]
    sizes = _sizes(topics, max_px=72)
    nodes = _linked_layout(topics, triples, sizes)
    by_term = {t["term"]: t for t in topics}
    smax = max((s for _, _, s in triples), default=1)
    grads = []
    for fam, (light, deep) in FAMILY_GRADIENTS.items():
        grads.append(
            '<linearGradient id="ncg-%s" x1="0" y1="0" x2="0" y2="1">'
            '<stop offset="15%%" stop-color="%s"/>'
            '<stop offset="85%%" stop-color="%s"/></linearGradient>'
            % (fam, light, deep))
    paths = []
    for k, (a, b, s) in enumerate(triples):
        if a not in nodes or b not in nodes:
            continue
        na, nb = nodes[a], nodes[b]
        mx, my = (na["x"] + nb["x"]) / 2, (na["y"] + nb["y"]) / 2
        dx, dy = nb["x"] - na["x"], nb["y"] - na["y"]
        d = math.hypot(dx, dy) or 1.0
        bend = 0.10 * d * (1 if k % 2 == 0 else -1)
        cx, cy = mx - dy / d * bend, my + dx / d * bend
        same = (by_term[a]["group"] == by_term[b]["group"]
                and by_term[a]["group"] != "neutral")
        color = (FAMILY_GRADIENTS[by_term[a]["group"]][1]
                 if same else "#9A8A70")
        paths.append(
            '<path d="M %.1f %.1f Q %.1f %.1f %.1f %.1f" fill="none" '
            'stroke="%s" stroke-opacity="%.2f" stroke-width="%.1f" '
            'stroke-linecap="round"/>'
            % (na["x"], na["y"], cx, cy, nb["x"], nb["y"], color,
               0.16 + 0.30 * (s / smax), 1.0 + 2.4 * (s / smax)))
    texts = []
    for t in topics:
        n = nodes[t["term"]]
        texts.append(
            '<text x="%.1f" y="%.1f" text-anchor="middle" '
            'dominant-baseline="central" font-size="%d" '
            'font-weight="%s" fill="url(#ncg-%s)" '
            'style="paint-order:stroke;stroke:#FBF7EE;stroke-width:5px">'
            '%s</text>'
            % (n["x"], n["y"], sizes[t["term"]],
               700 if t["weight"] >= 8 else 600, t["group"],
               html_mod.escape(t["term"])))
    svg = ('<svg viewBox="0 0 %d %d" style="width:100%%;height:auto;'
           'display:block" xmlns="http://www.w3.org/2000/svg">'
           '<defs>%s</defs>%s%s</svg>'
           % (LINK_PANEL_W, LINK_PANEL_H, "".join(grads),
              "".join(paths), "".join(texts)))
    caption = dict(caption or {})
    caption["right"] = ("Lines link topics that share headlines · "
                        "Size = distinct headlines mentioning the topic")
    return ('<div class="nc-frame" style="width:100%%;">'
            "<style>%s"
            "svg text { font-family:Georgia,'Times New Roman',"
            "'DejaVu Serif',serif; }</style>"
            '<div class="nc-head"><div class="nc-title">What the '
            'market is talking about</div><div class="nc-window">'
            'past 24 hours</div></div>'
            '<div style="padding:6px 0 2px;">%s</div>'
            '<div class="nc-foot"><div>%s</div><div>%s</div></div>'
            "</div>"
            % (_FRAME_CSS, svg, caption.get("left", ""),
               caption.get("right", "")))


def build_linked_cloud_html(hours=24):
    """Fetch -> extract -> linked frame HTML. Returns "" when
    the fetch produced no topics (the caller renders nothing
    rather than a broken panel). The footer caption matches
    build_cloud's left line; linked_cloud_html sets the right
    line to the links explanation."""
    now = datetime.now(timezone.utc)
    headlines, source_counts = fetch_headlines(hours=hours, now=now)
    topics = extract_topics(headlines)
    if not topics:
        return ""
    try:
        from zoneinfo import ZoneInfo
        asof = now.astimezone(ZoneInfo("America/Chicago")).strftime(
            "%a %b %-d, %-I:%M %p CT")
    except Exception:
        asof = now.strftime("%Y-%m-%d %H:%M UTC")
    caption = {
        "left": 'As of <span class="nc-hl">%s</span> · '
                '<span class="nc-hl">%d</span> distinct headlines, '
                '<span class="nc-hl">%d</span> sources'
                % (asof, len(headlines), len(source_counts)),
    }
    edges = cooccurrence_edges(topics, topic_members(topics, headlines))
    return linked_cloud_html(topics, edges, caption)
