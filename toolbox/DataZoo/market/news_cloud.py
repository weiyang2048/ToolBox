"""Market-news word cloud for the dashboard Futures page.

Builds a word cloud of what the market news is talking about over
the past ~24 hours: the topics moving the board, sized by how many
distinct headlines carry them. Self-regenerating — the module
fetches live public RSS (no API keys, no accounts), extracts and
weights topics, and renders the PNG end to end:

    python3 news_cloud.py            # writes cloud PNG + terms JSON

On the Futures page the intended use is a cached call, e.g.
st.cache_data(ttl=900) around build_cloud(), so the cloud refreshes
at most every ~15 minutes while the board itself stays live.

The sample is NEUTRAL (user requirement 2026-10-30): it pools the
section front pages / section feeds that major outlets assemble
themselves — no topic-targeted search queries anywhere, since a
query manufactures the ranking it claims to measure. The pool is
a GLOBAL MIX (user direction 2026-10-30): the Google News
Business section in six English regional editions, US outlet
fronts, Nikkei Asia and Livemint Markets. Topic
extraction is two-layered and co-equal: a curated market lexicon
(the Fed, earnings names, oil, gold, bitcoin, yields, ...) matched
against every headline, plus data-driven bigram/trigram mining
over the whole sample for whatever the lexicon has no name for.
A topic's weight is the number of *distinct* headlines mentioning
it, so wire syndication across outlets counts toward a topic once
per distinct story.

Rendering is the fancy HTML frame (user request 2026-10-30):
serif display type, per-family gradient fills, a paper-tinted
framed panel with a title line and an as-of footer. The Futures
page embeds cloud_html() directly — the visitor's browser renders
the gradients, so the live site needs no Chromium and no image.
render_html() screenshots the same frame to PNG with headless
Chromium for demos/offline use, and the `wordcloud` package
renderer is the plain fallback if Chromium is unavailable.
Colors stay categorical either way (gold for commodities and
energy, deep blue for rates and macro, red only for down-moves,
matching the board's own day-change colors).
"""

import json
import math
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PNG = os.path.join(HERE, "wordcloud-demo.png")
DEFAULT_JSON = os.path.join(HERE, "wordcloud_terms.json")
CHROMIUM = "/home/hatch/.cache/ms-playwright/chromium-1194/chrome-linux/chrome"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) news_cloud/1.0"

# ------------------------------------------------------------------
# Sources: public RSS, no credentials. NEUTRAL SAMPLE BY DESIGN
# (user requirement 2026-10-30): every source is a section front
# page or section feed an outlet assembles itself — its markets /
# business desk's own running coverage. There are NO topic-targeted
# searches anywhere in the sample: a query for "Binance" or "gold"
# would manufacture the very ranking the cloud claims to measure.
# Weights therefore read as "share of the pooled market-news front
# pages", and the terms JSON records each source's contribution so
# the mix is auditable. Feeds that 404 or return nothing parseable
# are skipped at fetch time (and show as absent in the JSON).
# ------------------------------------------------------------------
FEEDS = [
    ("CNBC Top News", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    ("CNBC Markets", "https://www.cnbc.com/id/20910258/device/rss/rss.html"),
    ("MarketWatch", "https://feeds.content.dowjones.io/public/rss/mw_topstories"),
    ("WSJ Markets", "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"),
    ("Bloomberg Markets", "https://feeds.bloomberg.com/markets/news.rss"),
    ("FT Markets", "https://www.ft.com/markets?format=rss"),
    ("Investing.com", "https://www.investing.com/rss/news.rss"),
    ("Yahoo ^GSPC", "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EGSPC&region=US&lang=en-US"),
    ("Yahoo ^IXIC", "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EIXIC&region=US&lang=en-US"),
    # Regional section fronts (global-mix widening, 2026-10-30).
    ("Nikkei Asia", "https://asia.nikkei.com/rss/feed/nar"),
    ("Livemint Markets", "https://www.livemint.com/rss/markets"),
    # SCMP's RSS answers with a bot-wall HTML page (checked
    # 2026-10-30) — omitted rather than carried as a dead entry.
]
# Google News *section* feeds (the Business desk's top stories as
# Google assembles them) — section fronts, not searches. Widened to
# a GLOBAL MIX at the user's direction 2026-10-30: the same Business
# section in six English-language regional editions. Cross-edition
# duplication is naturally suppressed — the same wire story carries
# the same headline in every edition and dedupes to one — while
# edition-specific coverage (an ASX story leading in Sydney, a
# Budget story in London) adds the breadth.
_BIZ_TOPIC = ("https://news.google.com/rss/topics/"
              "CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx1YlY4U0FtVnVHZ0pWVXlnQVAB")
GOOGLE_SECTIONS = [
    ("Google News Business US", _BIZ_TOPIC + "?hl=en-US&gl=US&ceid=US:en"),
    ("Google News Business UK", _BIZ_TOPIC + "?hl=en-GB&gl=GB&ceid=GB:en"),
    ("Google News Business IN", _BIZ_TOPIC + "?hl=en-IN&gl=IN&ceid=IN:en"),
    ("Google News Business AU", _BIZ_TOPIC + "?hl=en-AU&gl=AU&ceid=AU:en"),
    ("Google News Business SG", _BIZ_TOPIC + "?hl=en-SG&gl=SG&ceid=SG:en"),
    ("Google News Business CA", _BIZ_TOPIC + "?hl=en-CA&gl=CA&ceid=CA:en"),
]

# ------------------------------------------------------------------
# Curated market lexicon: (display term, [match patterns], family).
# Patterns are case-insensitive substrings unless wrapped in /regex/.
# Topics are colored BY FAMILY (user request 2026-10-30): similar
# topics get similar colors — one hue family per theme, in a few
# close shades so family members read as kin without being clones.
# Families: tech (AI, chips, platforms), crypto, macro (Fed, rates,
# data, policy), commodities (energy, metals), market (direction,
# volatility), neutral (unclassified mined phrases).
# ------------------------------------------------------------------
LEXICON = [
    ("AI spending", ["ai spending", "ai capex", "ai ambitions",
                     "spending scare", "ai buildout", "ai trade"], "tech"),
    ("Amazon earnings", ["amazon"], "tech"),
    ("AWS", ["aws", "amazon web services"], "tech"),
    ("Alphabet · Google", ["alphabet", "google"], "tech"),
    ("Microsoft · Azure", ["microsoft", "azure"], "tech"),
    ("Intel", ["intel"], "tech"),
    ("Nvidia", ["nvidia"], "tech"),
    ("Meta", ["/\\bmeta\\b/"], "tech"),
    ("Apple", ["apple"], "tech"),
    ("Rate hikes", ["december hike", "hike odds", "rate hike",
                    "fedwatch", "hike bets"], "macro"),
    ("Federal Reserve", ["federal reserve", "/\\bfed\\b/", "fomc",
                         "powell", "warsh"], "macro"),
    ("PCE inflation", ["pce", "inflation"], "macro"),
    ("Treasury yields", ["treasury yield", "10-year", "10 year",
                         "bond yield", "yields"], "macro"),
    ("GDP", ["gdp", "economic growth"], "macro"),
    ("Oil prices", ["oil price", "crude", "brent", "wti", "/\\bopec\\b/"], "commodities"),
    ("Strait of Hormuz", ["hormuz", "houthi", "tanker"], "commodities"),
    ("Gold", ["gold", "bullion"], "commodities"),
    ("Silver", ["silver"], "commodities"),
    ("Bitcoin", ["bitcoin", "/\\bbtc\\b/"], "crypto"),
    ("Binance", ["binance"], "crypto"),
    ("Crypto", ["crypto", "ether", "solana", "/\\bxrp\\b/"], "crypto"),
    ("Tech selloff", ["selloff", "sell-off", "rout", "futures fall",
                      "futures drop", "stocks fall", "stocks drop",
                      "stocks slide", "nasdaq fall", "nasdaq drop"], "market"),
    ("Free cash flow", ["free cash flow", "cash flow"], "tech"),
    ("Memory chips", ["memory chip", "memory prices", "pc market",
                      "pc slump", "pc slowdown"], "tech"),
    ("Chip stocks", ["semiconductor", "chipmaker", "chip stocks"], "tech"),
    ("Exxon · Chevron", ["exxon", "chevron"], "commodities"),
    ("Tariffs", ["tariff", "trade war"], "macro"),
    ("Anthropic", ["anthropic"], "tech"),
    ("OpenAI", ["openai"], "tech"),
    ("US dollar", ["dollar"], "macro"),
    ("Jobs data", ["jobs report", "unemployment", "payrolls",
                   "labor market"], "macro"),
    ("VIX", ["vix", "volatility"], "market"),
    ("Midterms", ["midterm"], "macro"),
    ("Reddit", ["reddit"], "tech"),
]
FAMILY_SHADES = {
    "tech": ["#0E7C7B", "#0F766E", "#2A9D8F"],
    "crypto": ["#7C3AED", "#6D28D9", "#8B5CF6"],
    "macro": ["#1D4E89", "#2F6690", "#3E7CB1"],
    "commodities": ["#A67C00", "#B8860B", "#8A6D00"],
    "market": ["#B03030", "#C0392B", "#9E2B25"],
    "neutral": ["#26343F", "#3D5666", "#546A7A"],
}
# Keyword cues that place a mined phrase in a family; first match wins.
FAMILY_CUES = [
    ("crypto", ["bitcoin", "crypto", "binance", "ether", "solana", "xrp"]),
    ("commodities", ["oil", "gold", "silver", "crude", "brent", "opec",
                     "copper", "gas prices"]),
    ("macro", ["fed", "rate", "inflation", "treasury", "yield", "gdp",
               "jobs", "tariff", "dollar", "powell", "economy"]),
    ("tech", ["ai", "nvidia", "chip", "amazon", "apple", "microsoft",
              "google", "alphabet", "meta", "openai", "anthropic",
              "intel", "semiconductor", "software"]),
    ("market", ["wall street", "s&p", "nasdaq", "dow", "stocks",
                "shares", "selloff", "rally", "futures", "vix"]),
]


# Gradient pairs (light -> deep) per family for the fancy frame.
FAMILY_GRADIENTS = {
    "tech": ("#35B5A8", "#0B6463"),
    "crypto": ("#A78BFA", "#5B21B6"),
    "macro": ("#5B8DC9", "#173F6E"),
    "commodities": ("#D9A821", "#7A5C00"),
    "market": ("#D95550", "#8E2320"),
    "neutral": ("#5B7A8C", "#1E2C38"),
}


def term_colors(topics):
    """{term: hex} — within each family, topics take the family's
    shades in weight order (largest topic gets the base shade), so
    similar topics land in visibly similar colors."""
    by_family = {}
    for t in topics:
        by_family.setdefault(t["group"], []).append(t)
    colors = {}
    for family, members in by_family.items():
        shades = FAMILY_SHADES.get(family, FAMILY_SHADES["neutral"])
        members.sort(key=lambda t: -t["weight"])
        for i, t in enumerate(members):
            colors[t["term"]] = shades[i % len(shades)]
    return colors
STOPWORDS = set("""the a an and or of to in on for as at by with from after
before over under amid among its it is are was were be been has have had
not no new now says say said will would can could may might than then
more most other into out up down off per via s u.s. us stocks stock
market markets today friday thursday wednesday week daily live""".split())


# ------------------------------------------------------------------
# Fetching
# ------------------------------------------------------------------
def _fetch(url):
    """GET url -> bytes. urllib first (honors the proxy env), curl as
    fallback — one of the two reaches every feed we use."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            return resp.read()
    except Exception:
        out = subprocess.run(
            ["curl", "-sL", "--max-time", "25", "-A", USER_AGENT, url],
            capture_output=True)
        return out.stdout


def _parse_date(text):
    if not text:
        return None
    text = text.strip()
    try:
        dt = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        dt = None
    if dt is None:
        # RSS 1.0 feeds (Nikkei) carry ISO dc:date instead of RFC 822.
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:
                dt = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def fetch_headlines(hours=24, now=None):
    """All distinct headlines from the last `hours` hours.

    Returns (headlines, source_counts): headlines are dicts with
    title / source / published; source_counts says how many usable
    in-window items each feed contributed."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=hours)
    urls = list(FEEDS) + list(GOOGLE_SECTIONS)
    seen, headlines, source_counts = set(), [], Counter()
    for name, url in urls:
        try:
            root = ET.fromstring(_fetch(url))
        except Exception:
            continue
        # Namespace-agnostic item walk: RSS 2.0 uses bare <item>,
        # RSS 1.0 (Nikkei Asia) namespaced <item> with <dc:date>.
        items = [el for el in root.iter()
                 if el.tag == "item" or el.tag.endswith("}item")]
        for item in items:
            def _field(name):
                for child in item:
                    if child.tag == name or child.tag.endswith("}" + name):
                        return child.text
                return None
            title = (_field("title") or "").strip()
            if not title:
                continue
            # Google News appends " - Publisher" to every title
            # (some edition suffixes run long — "Australian
            # Broadcasting Corporation"); strip twice for safety.
            for _ in range(2):
                title = re.sub(r"\s+-\s+[^-]{2,64}$", "", title)
            published = _parse_date(_field("pubDate") or _field("date"))
            if published is not None and published < cutoff:
                continue
            key = re.sub(r"\W+", " ", title.lower()).strip()
            if key in seen:
                continue
            seen.add(key)
            headlines.append({"title": title, "source": name,
                              "published": published})
            source_counts[name] += 1
    return headlines, source_counts


# ------------------------------------------------------------------
# Topic extraction
# ------------------------------------------------------------------
def _matches(pattern, text):
    if pattern.startswith("/") and pattern.endswith("/"):
        return re.search(pattern[1:-1], text) is not None
    return pattern in text


def _family_of(phrase):
    """Place a mined phrase in a color family by keyword cue."""
    for family, cues in FAMILY_CUES:
        if any(cue in phrase for cue in cues):
            return family
    return "neutral"


_TOKEN_RE = re.compile(r"[a-z0-9$%.'&-]+")


def _tokens(text):
    """Headline -> normalized word tokens for phrase mining.

    One canonical tokenizer for the cloud. The character class
    keeps intra-word punctuation that carries meaning — the &
    of S&P / AT&T / M&A, the $ and % of figures, apostrophes,
    hyphens — so entity names survive tokenization whole
    instead of shattering into fragments (the miner once
    surfaced "P 500" out of "S&P 500": & split the token and
    the bigram "p 500" outlived the pieces). Edge punctuation
    is stripped, a trailing period goes unless the token is an
    abbreviation ("u.s." keeps its dots, matching STOPWORDS),
    possessives fold to the bare noun ("china's" -> "china"),
    and a spaced ampersand stays as its own glue token so
    "Procter & Gamble" can form as a trigram.
    """
    out = []
    for raw in _TOKEN_RE.findall(text.lower()):
        if raw == "&":
            out.append("&")
            continue
        tok = raw.strip("&'-")
        if tok.endswith(".") and tok.count(".") == 1:
            tok = tok[:-1]
        if tok.endswith("'s") and len(tok) > 2:
            tok = tok[:-2]
        if tok and re.search(r"[a-z0-9]", tok):
            out.append(tok)
    return out


def _display(tokens):
    """Display form for a mined gram: Title Case per word, except
    ampersand tokens — their letter segments are initialisms
    and take full caps ("s&p" -> "S&P", "at&t" -> "AT&T",
    "m&a" -> "M&A")."""
    words = []
    for tok in tokens:
        if "&" in tok:
            words.append("&".join(
                seg.upper() if len(seg) <= 2 else seg.capitalize()
                for seg in tok.split("&")))
        elif tok == "&":
            words.append("&")
        else:
            words.append(tok.title())
    return " ".join(words)


def _mine_phrases(headlines, known_terms, top_n=10, min_weight=4):
    """Frequent bigrams/trigrams across ALL headlines — the neutral
    discovery layer, co-equal with the lexicon: it runs on the whole
    sample (not just lexicon leftovers), so a story the lexicon has
    no entry for can still top the cloud, and one it has an entry
    for can't hide a second theme. Phrases overlapping a lexicon
    term (either direction) are dropped as duplicates of it."""
    counts, example = Counter(), {}
    for h in headlines:
        words = _tokens(h["title"])
        grams = set()
        for n in (2, 3):
            for i in range(len(words) - n + 1):
                gram = words[i:i + n]
                if gram[0] in STOPWORDS or gram[-1] in STOPWORDS:
                    continue
                # The ampersand is glue, never an edge; and no
                # single-letter token may sit in a gram — a lone
                # letter is a fragment of something the tokenizer
                # split (the "p" of the old "P 500"), not a topic.
                if gram[0] == "&" or gram[-1] == "&":
                    continue
                if any(len(w) == 1 and w.isalpha() for w in gram):
                    continue
                grams.add(" ".join(gram))
        for gram in grams:
            counts[gram] += 1
            example.setdefault(gram, h["title"])
    # Template/boilerplate phrases are not topics: regional
    # editions carry formulaic headlines ("Q2 results 2026: X share
    # price ...") whose scaffolding would otherwise top the cloud.
    boilerplate = re.compile(
        r"\bresults?\b|share price|stock price|\bq[1-4]\b|"
        r"\bnews\b|broadcasting|corporation|experts'? views")
    mined = []
    # Deterministic candidate order: Counter.most_common ties
    # follow set-iteration order (hash-randomized per process),
    # which made overlapping variants (bigram vs trigram of the
    # same phrase) win at random. Sort by weight, then prefer
    # the shorter, cleaner gram.
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], len(kv[0]), kv[0]))
    for gram, weight in ranked[:60]:
        if weight < min_weight or len(mined) >= top_n:
            break
        if boilerplate.search(gram):
            continue
        # Overlap dedupe on a hyphen/space-insensitive key, so a
        # mined "tech sell-off" cannot duplicate the lexicon's
        # "Tech selloff" (seen live 2026-10-30: both rendered).
        key = gram.replace("-", "").replace(" ", "")

        def _overlaps(term):
            k = term.lower().replace("-", "").replace(" ", "")
            return key in k or k in key

        if any(_overlaps(t) for t in known_terms):
            continue
        if any(_overlaps(t["term"]) for t in mined):
            continue
        mined.append({"term": _display(gram.split(" ")),
                      "weight": weight,
                      "example_headline": example[gram],
                      "group": _family_of(gram), "mined": True})
    return mined


def extract_topics(headlines, max_topics=30, min_weight=2):
    """Weight lexicon topics by distinct-headline mentions, merged
    with phrases mined from the whole sample on equal footing.
    Returns topic dicts sorted by weight desc."""
    topics = []
    for term, patterns, group in LEXICON:
        hits = [h for h in headlines
                if any(_matches(p, h["title"].lower()) for p in patterns)]
        if len(hits) < min_weight:
            continue
        topics.append({"term": term, "weight": len(hits),
                       "example_headline": hits[0]["title"],
                       "group": group})
    topics += _mine_phrases(headlines, [t["term"] for t in topics])
    topics.sort(key=lambda t: (-t["weight"], t["term"]))
    return topics[:max_topics]


# ------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------
def _sizes(topics, max_px=52):
    """Font sizes on a POWER curve between a 16px floor and
    max_px. Calibrated by the user's three calls on 2026-10-30:
    sqrt (0.5) was too flat, linear (1.0) not dramatic enough,
    exponent 1.8 too much — 1.35 is the settled middle: leaders
    clearly dominate, the tail stays legible. The floor was
    raised 9px -> 16px on 2026-10-31 (the smallest words were
    too small to read on the live page)."""
    wmax = max(t["weight"] for t in topics)
    return {t["term"]: round(16 + (max_px - 16) * (t["weight"] / wmax) ** 1.35)
            for t in topics}


def render_wordcloud(topics, png_path):
    """Render via the wordcloud package; returns False if it is not
    importable (caller falls back to HTML + Chromium)."""
    try:
        from wordcloud import WordCloud
    except ImportError:
        return False
    import matplotlib
    font = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data",
                        "fonts", "ttf", "DejaVuSans-Bold.ttf")
    colors = term_colors(topics)

    def color_func(word, **_kwargs):
        return colors.get(word, FAMILY_SHADES["neutral"][0])

    cloud = WordCloud(
        width=1200, height=600, background_color="white",
        font_path=font if os.path.exists(font) else None,
        color_func=color_func, max_font_size=118, min_font_size=11,
        relative_scaling=0.38, prefer_horizontal=0.92,
        margin=5, collocations=False,
    ).generate_from_frequencies({t["term"]: t["weight"] for t in topics})
    cloud.to_file(png_path)
    return True


_FRAME_CSS = """
.nc-frame { padding:24px 38px 18px; box-sizing:border-box;
  font-family:Georgia,'Times New Roman','DejaVu Serif',serif;
  background:radial-gradient(120% 130% at 50% 38%,#FFFFFF 55%,#F6F1E7 100%);
  border:1px solid #D8CBB6; border-radius:14px; display:flex;
  flex-direction:column; }
.nc-head { display:flex; justify-content:space-between; align-items:baseline;
  border-bottom:2px solid #A67C00; padding-bottom:8px; }
.nc-title { font-size:19px; letter-spacing:3.5px; color:#1D4E89;
  font-variant:small-caps; font-weight:700; }
.nc-window { font-size:14px; color:#6F6555; font-style:italic; }
.nc-cloud { flex:1; display:flex; flex-wrap:wrap; align-items:center;
  align-content:center; justify-content:center; column-gap:26px; row-gap:8px;
  padding:10px 4px; }
.nc-w { white-space:nowrap; line-height:1.08; display:inline-block;
  -webkit-background-clip:text; background-clip:text; color:transparent;
  filter:drop-shadow(0 1px 1px rgba(35,34,31,.22)); }
.nc-tilt { transform:rotate(-5deg); }
.nc-foot { border-top:1px solid #D8CBB6; padding-top:8px; font-size:13.5px;
  color:#6F6555; display:flex; justify-content:space-between; gap:16px; }
.nc-hl { color:#FFB066; font-weight:700; }
"""


def _frame_html(topics, caption=None, max_px=60):
    """The fancy frame's inner markup + scoped CSS (nc- prefixed, so
    it can be embedded in a host page without style collisions)."""
    import html as html_mod
    sizes = _sizes(topics, max_px=max_px)
    order = [t["term"] for t in topics]
    # Flow order: interleave large and small so the biggest terms land
    # mid-cloud instead of stacking in the first row.
    order = order[::2][::-1] + order[1::2]
    by_term = {t["term"]: t for t in topics}
    spans = []
    for pos, term in enumerate(order):
        t = by_term[term]
        light, deep = FAMILY_GRADIENTS.get(t["group"],
                                           FAMILY_GRADIENTS["neutral"])
        cls = "nc-w"
        # A slight tilt on a few mid terms — texture, never the leaders.
        # (Vertical terms were tried and dropped on demo day: in a
        # flow layout they collide with their neighbours.)
        if t["weight"] <= 5 and pos % 7 == 4:
            cls = "nc-w nc-tilt"
        spans.append(
            '<span class="%s" style="font-size:%dpx;font-weight:%s;'
            'background-image:linear-gradient(178deg,%s 15%%,%s 85%%)">'
            '%s</span>' % (cls, sizes[term],
                           700 if t["weight"] >= 8 else 600,
                           light, deep, html_mod.escape(term)))
    caption = caption or {}
    # Caption lines are trusted HTML (built by build_cloud from a
    # format string and integer counts — the date and counts carry
    # .nc-hl accent spans), so they are inserted unescaped.
    return ("<style>%s</style>"
            '<div class="nc-head"><div class="nc-title">What the market '
            'is talking about</div><div class="nc-window">past 24 hours'
            '</div></div><div class="nc-cloud">%s</div>'
            '<div class="nc-foot"><div>%s</div><div>%s</div></div>'
            % (_FRAME_CSS, "\n".join(spans),
               caption.get("left", ""), caption.get("right", "")))


def cloud_html(topics, caption=None):
    """The fancy frame as embeddable HTML for the Futures page:
    responsive width, content height. The host page renders this
    directly (the visitor's browser does the gradient text), so the
    live site gets the exact fancy design with no Chromium and no
    image round-trip. `topics` are extract_topics() dicts (term,
    weight, group); `caption` carries the left/right footer lines."""
    return ('<div class="nc-frame" style="width:100%%;min-height:330px;">'
            + _frame_html(topics, caption, max_px=60) + "</div>")


def render_html(topics, png_path, caption=None):
    """The fancy renderer as a PNG: the same frame at a fixed
    1200x600, screenshotted with headless Chromium (demo / offline
    use; the live page embeds cloud_html() instead)."""
    page = """<!doctype html><html><head><meta charset="utf-8"><style>
* { margin:0; padding:0; box-sizing:border-box; }
body { background:#FFFFFF; }
</style></head><body><div class="nc-frame"
style="width:1200px;height:600px;">%s</div></body></html>""" % (
        _frame_html(topics, caption, max_px=76))
    if not os.path.exists(CHROMIUM):
        return False
    with tempfile.NamedTemporaryFile("w", suffix=".html",
                                     delete=False) as fh:
        fh.write(page)
        tmp = fh.name
    try:
        # Window deliberately larger than the 1200x600 frame: at an
        # exact-fit window, headless Chromium clips the frame's
        # bottom edge (footer) out of the shot. Shoot with headroom,
        # then trim the blank margin back off with PIL.
        subprocess.run([CHROMIUM, "--headless", "--disable-gpu", "--no-sandbox",
                        "--hide-scrollbars", "--window-size=1240,700",
                        "--screenshot=" + png_path, "file://" + tmp],
                       capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    finally:
        os.unlink(tmp)
    if not os.path.exists(png_path):
        return False
    try:
        from PIL import Image
        im = Image.open(png_path).convert("L")
        import numpy as np
        rows = np.where(np.asarray(im).min(axis=1) < 250)[0]
        cols = np.where(np.asarray(im).min(axis=0) < 250)[0]
        if len(rows) and len(cols):
            box = (max(int(cols.min()) - 4, 0), max(int(rows.min()) - 4, 0),
                   int(cols.max()) + 5, int(rows.max()) + 5)
            Image.open(png_path).crop(box).save(png_path)
    except Exception:
        pass
    return True


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------
def build_cloud(png_path=DEFAULT_PNG, json_path=DEFAULT_JSON, hours=24):
    """Fetch -> extract -> render -> write terms JSON. Returns the
    summary dict that is also written to json_path."""
    now = datetime.now(timezone.utc)
    headlines, source_counts = fetch_headlines(hours=hours, now=now)
    topics = extract_topics(headlines)
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
        "right": "Size = distinct headlines mentioning the topic",
    }
    # Rendering the PNG is best-effort: embedders (the Futures
    # page) consume topics + caption and render HTML themselves,
    # and a renderer failure must never cost them the data. (Live
    # lesson 2026-10-30: on Streamlit Cloud there is no Chromium,
    # and the uncaught spawn error silently emptied the panel.)
    try:
        renderer = ("html-fancy" if render_html(topics, png_path, caption)
                    else "wordcloud")
        if renderer != "html-fancy":
            render_wordcloud(topics, png_path)
    except Exception:
        renderer = "none"
    summary = {
        "window": "past %d hours ending %s" % (hours, now.isoformat()),
        "headlines": len(headlines),
        "sources": dict(source_counts),
        "renderer": renderer,
        "caption": caption,
        "topics": [{"term": t["term"], "weight": t["weight"],
                    "group": t["group"],
                    "example_headline": t["example_headline"]}
                   for t in topics],
    }
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)
    return summary


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    png = args[0] if len(args) > 0 else DEFAULT_PNG
    js = args[1] if len(args) > 1 else DEFAULT_JSON
    summary = build_cloud(png, js)
    print("headlines: %d from %d sources | renderer: %s"
          % (summary["headlines"], len(summary["sources"]),
             summary["renderer"]))
    for t in summary["topics"][:12]:
        print("  %3d  %s" % (t["weight"], t["term"]))
    print("wrote", png, "and", js)
    return 0


if __name__ == "__main__":
    sys.exit(main())
