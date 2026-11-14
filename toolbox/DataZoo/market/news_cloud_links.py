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
    _FRAME_CSS,
    FAMILY_GRADIENTS,
    FEEDS,
    GOOGLE_SECTIONS,
    _sizes,
    extract_topics_challenger,
    fetch_headlines,
    tone_colored,
    topic_member_sets,
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
    appearing, in order, in the headline — canonically: a mined
    topic matches on every surface form pooled into it, so a
    merged topic's member set is the full union its weight
    counts. The implementation is canonical in news_cloud
    (topic_member_sets), where extraction itself — grouping
    and the containment fold — runs on the same sets."""
    return topic_member_sets(topics, headlines)


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


def _place_nodes(topics, sizes):
    """Greedy max-min placement over the whole panel:
    each topic, in order, takes the grid point farthest
    from those already placed."""
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
    return nodes


def _force_relax(nodes, edges):
    """Ellipse-aware repulsion (a word clears its full
    box plus a margin, not just its center) with weak edge
    springs and weak gravity, 500 iterations. Mutates
    ``nodes`` in place."""
    margin = 20.0
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


def _deoverlap(nodes):
    """Padded de-overlap pass: push apart any pair whose
    boxes (plus padding) still intersect, up to 80 rounds,
    early exit when nothing moves. Mutates ``nodes``."""
    terms = list(nodes)
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
    nodes = _place_nodes(topics, sizes)
    _force_relax(nodes, edges)
    _deoverlap(nodes)
    return nodes


def linked_cloud_html(topics, edges, caption=None):
    """The linked cloud as embeddable HTML: an SVG panel of
    words at computed positions, curved lines between topics
    that share headlines, inside the fancy frame — with the
    title and footer lines OUTSIDE the frame (user-approved
    2026-10-30): the title above it as a small section head,
    the caption below it as a quiet row, the frame itself
    holding only the words. The caption's left line (the as-of
    stamp with the headline/source counts) renders as its own
    row between the title and the frame, not in the footer
    (user request 2026-10-30). `edges` are (a, b, shared) triples
    or dicts; the caption's right line is set to the links
    explanation."""
    import html as html_mod
    triples = [(e["a"], e["b"], e["shared"]) if isinstance(e, dict)
               else tuple(e) for e in edges]
    sizes = _sizes(topics, max_px=64)
    nodes = _linked_layout(topics, triples, sizes)
    by_term = {t["term"]: t for t in topics}
    smax = max((s for _, _, s in triples), default=1)
    grads = []
    for fam, (light, deep) in FAMILY_GRADIENTS.items():
        grads.append(
            f'<linearGradient id="ncg-{fam}" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="15%" stop-color="{light}"/>'
            f'<stop offset="85%" stop-color="{deep}"/></linearGradient>')
    # Tone gradients: one per topic — the tone scale is
    # continuous, so no shared family def can carry it.
    for i, t in enumerate(topics):
        if "tone_color" in t:
            grads.append(
                '<linearGradient id="ncg-tone-{}" x1="0" y1="0" '
                'x2="0" y2="1"><stop offset="15%" '
                'stop-color="{}"/><stop offset="85%" '
                'stop-color="{}"/></linearGradient>'.format(
                    i, t["tone_light"], t["tone_color"]))
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
        if "tone_color" in by_term[a] \
                and "tone_color" in by_term[b]:
            # The link takes the average of its endpoints'
            # tone colors.
            ca = [int(by_term[a]["tone_color"][j:j + 2], 16)
                  for j in (1, 3, 5)]
            cb = [int(by_term[b]["tone_color"][j:j + 2], 16)
                  for j in (1, 3, 5)]
            color = "#{:02X}{:02X}{:02X}".format(
                *[round((x + y) / 2) for x, y in zip(ca, cb)])
        else:
            same = (by_term[a]["group"] == by_term[b]["group"]
                    and by_term[a]["group"] != "neutral")
            color = (FAMILY_GRADIENTS[by_term[a]["group"]][1]
                     if same else "#9A8A70")
        paths.append(
            '<path d="M {:.1f} {:.1f} Q {:.1f} {:.1f} {:.1f} {:.1f}" fill="none" '
            'stroke="{}" stroke-opacity="{:.2f}" stroke-width="{:.1f}" '
            'stroke-linecap="round"/>'.format(na["x"], na["y"], cx, cy, nb["x"], nb["y"], color,
               0.16 + 0.30 * (s / smax), 1.0 + 2.4 * (s / smax)))
    texts = []
    for i, t in enumerate(topics):
        n = nodes[t["term"]]
        fill_id = (f"tone-{i}" if "tone_color" in t
                   else t["group"])
        text_el = (
            '<text x="{:.1f}" y="{:.1f}" text-anchor="middle" '
            'dominant-baseline="central" font-size="{}" '
            'font-weight="{}" fill="url(#ncg-{})" '
            'style="paint-order:stroke;stroke:#FBF7EE;stroke-width:5px">'
            '{}</text>'
            .format(n["x"], n["y"], sizes[t["term"]],
                    700 if t["weight"] >= 8 else 600, fill_id,
                    html_mod.escape(t["term"])))
        link = t.get("link") or ""
        if link:
            # Clickable topic (user request 2026-11-01): the
            # word links to its representative source article;
            # the SVG <title> gives the headline as a tooltip,
            # and the caption line says topics open a source
            # article. No visual marker on the words themselves:
            # an underline shipped earlier the same day was
            # removed at the user's direction ("I dont want to
            # see underlines in the word cloud").
            text_el = ('<a href="{}" target="_blank" '
                       'rel="noopener noreferrer"><title>{}</title>'
                       '{}</a>'.format(html_mod.escape(link, quote=True),
                          html_mod.escape(
                              t.get("example_headline", "")),
                          text_el))
        texts.append(text_el)
    svg = ('<svg viewBox="0 0 {} {}" style="width:100%;height:auto;'
           'display:block" xmlns="http://www.w3.org/2000/svg">'
           '<defs>{}</defs>{}{}</svg>'
           .format(LINK_PANEL_W, LINK_PANEL_H, "".join(grads),
                   "".join(paths), "".join(texts)))
    caption = dict(caption or {})
    caption["right"] = ("Lines link topics that share headlines · "
                        "Size = distinct stories mentioning the topic"
                        " · Color = headline tone (green positive,"
                        " black neutral, red negative)"
                        " · The headlines below link to the source "
                        "articles")
    asof_row = ('<div class="nc-asof">{}</div>'.format(caption.get("left", ""))) if caption.get("left") else ""
    return ('<div style="width:100%;font-family:Georgia,'
            "'Times New Roman','DejaVu Serif',serif;\">"
            "<style>{}"
            "svg text {{ font-family:Georgia,'Times New Roman',"
            "'DejaVu Serif',serif; }}"
            "svg a {{ cursor:pointer; }}"
            "svg a:hover {{ opacity:.68; }}"
            ".nc-topbar {{ display:flex;"
            " justify-content:space-between; align-items:baseline;"
            " padding:0 2px 6px; margin-bottom:10px; }}"
            # The topbar sits OUTSIDE the white frame, on the
            # host page's dark background (user screenshot
            # 2026-11-05: the title was "not very visible") —
            # the on-white navy #1D4E89 and mid gray #9AA0A6
            # from the framed variant do not read there, so
            # the bar carries its own light colors.
            ".nc-topbar .t {{ font-size:15px; letter-spacing:2.6px;"
            " color:#EAF2FB; font-variant:small-caps;"
            " font-weight:700; }}"
            ".nc-topbar .w {{ font-size:12px; color:#C9D4DC;"
            " font-style:italic; }}"
            ".nc-asof {{ padding:0 2px 8px; font-size:12px;"
            " color:#9AA0A6; }}"
            ".nc-asof .nc-hl {{ color:#FFB066; font-weight:700; }}"
            ".nc-cap {{ display:flex;"
            " justify-content:space-between; gap:16px;"
            " padding:7px 2px 0; font-size:12px; color:#9AA0A6; }}"
            # Compact text on phones (user-approved 2026-10-30):
            # smaller head/caption type and tighter frame
            # padding at <=640px; desktop untouched.
            "@media (max-width: 640px) {{"
            " .nc-frame {{ padding:10px 12px 8px;"
            " border-radius:12px; }}"
            " .nc-topbar .t {{ font-size:12px;"
            " letter-spacing:.14em; }}"
            " .nc-topbar .w {{ font-size:10.5px; }}"
            " .nc-asof {{ font-size:10.5px; }}"
            " .nc-cap {{ font-size:10.5px; line-height:1.5; }}"
            " }}</style>"
            '<div class="nc-topbar"><div class="t">What the '
            'market is talking about</div><div class="w">'
            'past 24 hours</div></div>'
            '{}'
            '<div class="nc-frame" '
            'style="width:100%;padding:14px 18px;">'
            '<div style="padding:2px 0;">{}</div></div>'
            '<div class="nc-cap"><div></div><div>{}</div></div>'
            "</div>".format(_FRAME_CSS, asof_row, svg,
               caption.get("right", "")))


def build_linked_cloud_html(hours=24, progress=None):
    """Fetch -> extract -> linked frame HTML. Returns "" when
    the fetch produced no topics (the caller renders nothing
    rather than a broken panel). The footer caption matches
    build_cloud's left line; linked_cloud_html sets the right
    line to the links explanation.

    ``progress`` (optional) reports the WHOLE build, stage
    by stage (user direction 2026-11-04: everything between
    the wait starting and the results showing): per feed
    from fetch_headlines, a stage label at the full feed
    count before topic extraction (which names the headline
    count it works on), per-batch headline counts from the
    tone scorer — re-labelled "scoring tone (FinBERT)"
    unless the scorer named the step itself (the model
    load) — and a closing stage label for the layout. The
    default ``None`` reports nothing."""
    now = datetime.now(timezone.utc)
    headlines, source_counts = fetch_headlines(
        hours=hours, now=now, progress=progress)
    n_feeds = len(FEEDS) + len(GOOGLE_SECTIONS)
    if progress is not None:
        progress(n_feeds, n_feeds,
                 f"extracting topics from {len(headlines)} "
                 "headlines")
    topics = extract_topics_challenger(headlines)
    if topics:
        def _tone_progress(done, total, label):
            progress(done, total,
                     label or "scoring tone (FinBERT)")

        topics = tone_colored(
            topics, headlines,
            progress=_tone_progress if progress is not None
            else None)
    if not topics:
        return ""
    if progress is not None:
        progress(n_feeds, n_feeds, "laying out the cloud")
    try:
        from toolbox.DataZoo.market.eastern import asof_stamp
        asof = asof_stamp(now)
    except Exception:  # noqa: BLE001 - stamp fallback must never break the build (import can fail non-ImportError, e.g. missing tzdata)
        asof = now.strftime("%Y-%m-%d %H:%M UTC")
    caption = {
        "left": f'As of <span class="nc-hl">{asof}</span> · '
                f'<span class="nc-hl">{len(headlines)}</span> distinct headlines, '
                f'<span class="nc-hl">{len(source_counts)}</span> sources',
    }
    edges = cooccurrence_edges(topics, topic_members(topics, headlines))
    return linked_cloud_html(topics, edges, caption)
