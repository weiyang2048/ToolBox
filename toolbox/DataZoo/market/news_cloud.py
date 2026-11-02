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
fronts, the BBC and Guardian business desks, and Nikkei Asia.
Topic
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
    # UK business desks (added 2026-11-01, live-tested the same day).
    ("BBC Business", "https://feeds.bbci.co.uk/news/business/rss.xml"),
    ("Guardian Business", "https://www.theguardian.com/uk/business/rss"),
    ("Investing.com", "https://www.investing.com/rss/news.rss"),
    ("Yahoo ^GSPC", "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EGSPC&region=US&lang=en-US"),
    ("Yahoo ^IXIC", "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EIXIC&region=US&lang=en-US"),
    # Regional section fronts (global-mix widening, 2026-10-30).
    ("Nikkei Asia", "https://asia.nikkei.com/rss/feed/nar"),
    # Livemint Markets was removed at the user's direction
    # 2026-11-01.
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
    ("AWS", ["/\\baws\\b/", "amazon web services"], "tech"),
    ("Alphabet · Google", ["alphabet", "google"], "tech"),
    ("Microsoft · Azure", ["microsoft", "azure"], "tech"),
    ("Intel", ["/\\bintel\\b/"], "tech"),
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
    ("Crypto", ["crypto", "/\\bether(eum)?\\b/", "solana", "/\\bxrp\\b/"], "crypto"),
    ("Tech selloff", ["selloff", "sell-off", "/\\brout\\b/", "futures fall",
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
market markets today friday thursday wednesday week daily live
ahead""".split())


# ------------------------------------------------------------------
# Robustness gates for mined topics (user-approved 2026-11-02,
# demoed on the live pool first: demos/news-cloud-fixes). The
# mining layer's known junk classes each get the test designed
# for them — a shape gate (fragments and bare quantities), an
# outlet gate (single-outlet house style), template demotion
# (rotating-entity headline skeletons) — and topic weights count
# story clusters, not raw headlines.
# ------------------------------------------------------------------
# Possessive / demonstrative edges can never head a topic
# ("Their Own" rendered as a topic on 2026-11-02). Question and
# preposition edges are deliberately NOT here: the window fold
# owns those ("What Is Behind The Move" is its adjudicated
# output, pinned by its tests).
_EDGE_JUNK = {"their", "his", "her", "our", "your", "my",
              "this", "that", "these", "those"}
# Common headline verbs / adjectives: a gram built only from
# these has no noun head ("Make Final" — verb + adjective).
_HEAD_VERBS = set("""make makes made get gets take takes see sees hit hits
rise rises fall falls cut cuts raise raises drop drops gain gains lose
loses surge surges plunge plunges jump jumps climb climbs slip slips
soar soars tumble tumbles set sets put puts turn turns face faces show
shows give gives lead leads hold holds keep keeps add adds end ends
begin begins post posts beat beats miss misses warn warns""".split())
_HEAD_ADJS = {"final", "own", "big", "large", "small", "high", "low",
              "top", "first", "last", "next", "major", "key", "main",
              "strong", "weak", "hot", "prime"}
_NOMINAL_SUFFIXES = ("tion", "sion", "ness", "ment", "ity", "ism",
                     "ist", "ship", "ance", "ence")
_QUANTITY_WORDS = {"billion", "million", "trillion", "percent", "%"}
# Fragment shapes caught live 2026-11-02, after the first gates
# shipped ("Least 24", "Some Facts", "30 Years" all rendered):
# determiner edges that head no noun phrase, quantity adjectives
# that cannot serve as a noun head before a number tail, and
# bare durations (a number + a time unit is a span, not a topic).
_DETERMINER_EDGES = {"some", "any", "all", "each", "every",
                      "another"}
_QUANTITY_ADJS = {"least", "most", "more", "less", "fewer",
                  "few", "many", "much"}
_TIME_UNITS = {"year", "month", "week", "day", "hour", "minute",
               "decade"}


def _noun_like(tok):
    """A token that can head a noun phrase: protected tokens
    (& $ % digits — S&P, 500), nominal-suffix words, and anything
    that is not a known verb / adjective / function word."""
    if any(c in tok for c in "&$%") or any(c.isdigit() for c in tok):
        return True
    if tok.endswith(_NOMINAL_SUFFIXES):
        return True
    return (tok not in _HEAD_VERBS and tok not in _HEAD_ADJS
            and tok not in STOPWORDS and tok not in _EDGE_JUNK)


def _shape_ok(canon):
    """The shape gate on a candidate's canonical tokens: no
    possessive/demonstrative edge, not a bare quantity ("$20
    Billion" is a figure, not a topic), and at least one
    noun-like token as the phrase's head."""
    if canon[0] in _EDGE_JUNK or canon[-1] in _EDGE_JUNK:
        return False
    if all(re.fullmatch(r"\$?[\d,.]+", t) or t in _QUANTITY_WORDS
           for t in canon):
        return False
    # Determiner-initial fragments ("Some Facts"): a determiner
    # edge heads no topic, like the possessive edges above.
    if canon[0] in _DETERMINER_EDGES:
        return False
    # Bare duration ("30 Years"): number + time unit is a span.
    if (len(canon) == 2 and canon[1] in _TIME_UNITS
            and re.fullmatch(r"\$?[\d,.]+", canon[0])):
        return False
    # Number tail ("Least 24"): a phrase ending in a bare number
    # needs a genuine noun head before the tail — a token that
    # is noun-like on its own letters, not via its digits, and
    # not a quantity adjective. ("S&P 500" keeps its head: the
    # & token is noun-like without any digit.)
    if re.fullmatch(r"\$?[\d,.]+", canon[-1]):
        return any(_noun_like(t) and not any(c.isdigit()
                                             for c in t)
                   and t not in _QUANTITY_ADJS
                   for t in canon[:-1])
    return any(_noun_like(t) for t in canon)


def _skeleton(title):
    """A headline's template skeleton: figures and currency
    words -> <N>, capitalized tokens -> <E>, placeholder runs
    collapsed (entity names vary in token length, prices carry
    units — one template must give one skeleton)."""
    out = []
    for tok in title.replace(",", " ").split():
        low = tok.lower().strip(".:;!?")
        if re.fullmatch(r"\$?[\d][\d,.]*", tok) or low in {
                "usd", "eur", "gbp", "jpy", "dollars", "dollar"}:
            out.append("<N>")
        elif tok[:1].isupper():
            out.append("<E>")
        else:
            out.append(low)
    collapsed = []
    for tok in out:
        if (not collapsed or tok != collapsed[-1]
                or tok not in ("<E>", "<N>")):
            collapsed.append(tok)
    return " ".join(collapsed)


def _template_bound(titles):
    """True when a topic's headlines live inside ONE headline
    skeleton with rotating entities — a wire/data template
    ("<E> stock hits 52-week low at <N>" x 4 companies), not a
    story. The rotating-entity count is the guard: a genuinely
    repeated story keeps the same entities and is not demoted."""
    if len(titles) < 3:
        return False
    skels = Counter(_skeleton(t) for t in titles)
    top_skel, cnt = skels.most_common(1)[0]
    if cnt < 0.8 * len(titles):
        return False
    fillers = {" ".join(tok for tok in t.replace(",", " ").split()
                        if tok[:1].isupper())
               for t in titles if _skeleton(t) == top_skel}
    return len(fillers) >= 3


# Two headlines tell the same story when their token sets are
# close (the published headline story-linking operating point,
# token Jaccard >= 0.55) AND they differ in no rare token: near
# duplicates of one story differ in boilerplate, while template
# siblings ("What Is Behind The Move In X Stock?" for two
# different X) differ in exactly one rare entity token — the
# rare-token veto keeps those apart.
STORY_JACCARD = 0.55


def _story_clusters(headlines):
    """Union-find story clusters over the pool; returns a
    cluster id per headline index."""
    sets = [set(_tokens(h["title"])) for h in headlines]
    df = Counter()
    for s in sets:
        df.update(s)
    rare_df = max(2, int(0.01 * len(headlines)))
    parent = list(range(len(headlines)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(headlines)):
        for j in range(i + 1, len(headlines)):
            a, b = sets[i], sets[j]
            if not a or not b:
                continue
            if len(a & b) / len(a | b) < STORY_JACCARD:
                continue
            if any(df[t] <= rare_df for t in a ^ b):
                continue
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
    return [find(i) for i in range(len(headlines))]


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
                              "link": (_field("link") or "").strip(),
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


# ------------------------------------------------------------------
# Canonical keys: lemmatization-lite for headline phrases. A
# topic's canonical key is the tuple of its singularized
# lowercase tokens; surface variants of one phrase ("senate
# race" / "senate races", "midterm" / "midterm elections")
# share a key family, so counting and merging can pool them.
# Keys are merge-only — display labels are always real surface
# forms, so an imperfect key (endorses -> endors) can at worst
# fail to merge, never print a mangled word.
# ------------------------------------------------------------------
NEVER_SINGULARIZE = {
    # Not plurals despite the trailing s.
    "news", "jones", "reuters", "houthis", "wales", "andes",
    "series", "species", "politics", "economics", "athletics",
    "business",  # ends in ss anyway; listed for clarity
}


def singularize(tok):
    """One token -> its canonical (singular) form. Protections:
    tokens carrying meaning-bearing punctuation (& $ % .) or
    digits are never touched (S&P, AT&T, M&A, 500); a small
    never-list of proper nouns / mass nouns; ss/us/is endings
    are not plurals."""
    if (any(c in tok for c in "&$%.") or any(c.isdigit() for c in tok)
            or tok in NEVER_SINGULARIZE or len(tok) <= 3):
        return tok
    if tok.endswith(("ss", "us", "is")):
        return tok
    if tok.endswith("ies") and len(tok) > 4:
        return tok[:-3] + "y"
    if tok.endswith(("ses", "xes", "zes", "ches", "shes")):
        return tok[:-2]
    if tok.endswith("s"):
        return tok[:-1]
    return tok


def _canon_tokens(tokens):
    return tuple(singularize(t) for t in tokens)


def _canon_of_term(term):
    """Canonical key of a topic's display term."""
    toks = term.lower().replace("-", " ").replace("·", " ").split()
    return _canon_tokens(toks)


def _mine_phrases(headlines, known_terms, top_n=10, min_weight=3):
    """Frequent bigrams/trigrams across ALL headlines — the neutral
    discovery layer, co-equal with the lexicon: it runs on the whole
    sample (not just lexicon leftovers), so a story the lexicon has
    no name for can still top the cloud, and one it has an entry
    for can't hide a second theme. Phrases overlapping a lexicon
    term (either direction) are dropped as duplicates of it.

    Grams are counted by CANONICAL KEY, so surface variants of
    one phrase pool their headlines before the min_weight cut
    and the ranking; the display label is the key's most
    frequent surface form, and the topic records every pooled
    surface (the links layer matches on all of them)."""
    counts = Counter()
    surf_counts = {}
    key_sources = {}
    pool_outlets = len({h.get("source", "") for h in headlines})
    surf_example, surf_example_link = {}, {}
    surf_free, surf_free_link = {}, {}
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
        per_key = {}
        for gram in grams:
            key = _canon_tokens(gram.split(" "))
            per_key.setdefault(key, []).append(gram)
        for key, surfaces in per_key.items():
            counts[key] += 1
            key_sources.setdefault(key, set()).add(
                h.get("source", ""))
            sc = surf_counts.setdefault(key, Counter())
            for gram in surfaces:
                sc[gram] += 1
                surf_example.setdefault(gram, h["title"])
                surf_example_link.setdefault(gram, h.get("link", ""))
                # The first NON-paywalled example per surface is
                # tracked alongside: a mined topic's link prefers
                # it (the representative rule in _representative,
                # applied here at collection time).
                if not _paywalled(h.get("source", "")):
                    surf_free.setdefault(gram, h["title"])
                    surf_free_link.setdefault(gram,
                                               h.get("link", ""))
    # Template/boilerplate phrases are not topics: regional
    # editions carry formulaic headlines ("Q2 results 2026: X share
    # price ...") whose scaffolding would otherwise top the cloud.
    boilerplate = re.compile(
        r"\bresults?\b|share price|stock price|\bq[1-4]\b|"
        r"\bnews\b|broadcasting|corporation|experts'? views")
    # Fragment suppression: a gram that occurs only inside a
    # longer gram is a fragment of it, not its own topic (seen
    # live 2026-11-01: "protests turn" and "turn violent" both
    # rendered as topics, linked to each other, while every one
    # of their headlines was the single phrase "protests turn
    # violent" — at equal weight the shorter grams rank first
    # below, get accepted, and the dedupe then rejects the
    # whole phrase as their duplicate). A gram survives only
    # if its count outside every longer container still clears
    # min_weight on its own.
    def _contained(short, long_):
        return any(long_[i:i + len(short)] == short
                   for i in range(len(long_) - len(short) + 1))

    containers = [k for k, w in counts.items()
                  if w >= 2 and not boilerplate.search(" ".join(k))]

    def _independent(key, weight):
        best = 0
        for other in containers:
            if len(other) > len(key) and _contained(key, other):
                best = max(best, counts[other])
        return weight - best

    mined = []
    # Deterministic candidate order: Counter.most_common ties
    # follow set-iteration order (hash-randomized per process),
    # which made overlapping variants (bigram vs trigram of the
    # same phrase) win at random. Sort by weight, then prefer
    # the shorter, cleaner gram.
    ranked = sorted(counts.items(),
                    key=lambda kv: (-kv[1], len(kv[0]), kv[0]))
    for key, weight in ranked[:60]:
        if weight < min_weight or len(mined) >= top_n:
            break
        if boilerplate.search(" ".join(key)):
            continue
        if _independent(key, weight) < min_weight:
            continue
        # Shape gate: fragments and bare quantities are not
        # topics, however often they repeat.
        if not _shape_ok(key):
            continue
        # Outlet gate: in a multi-outlet pool, a phrase carried
        # by a single outlet is that outlet's house style, not a
        # market topic (the 2026-11-02 "Hits 52-Week Low" /
        # "Hong Kong" cases, both one outlet). Vacuous in a
        # single-outlet pool, where the concept has no meaning.
        if pool_outlets >= 2 and len(key_sources[key]) < 2:
            continue
        # Display surface: the key's most frequent surface form
        # (ties: shorter, then A-Z — deterministic).
        surface = sorted(surf_counts[key].items(),
                         key=lambda kv: (-kv[1], len(kv[0]), kv[0]))[0][0]
        # Overlap dedupe on a hyphen/space-insensitive key, so a
        # mined "tech sell-off" cannot duplicate the lexicon's
        # "Tech selloff" (seen live 2026-10-30: both rendered).
        dkey = surface.replace("-", "").replace(" ", "")

        def _overlaps(term, _dkey=dkey):
            k = term.lower().replace("-", "").replace(" ", "")
            return _dkey in k or k in _dkey

        if any(_overlaps(t) for t in known_terms):
            continue
        if any(_overlaps(t["term"]) for t in mined):
            continue
        mined.append({"term": _display(surface.split(" ")),
                      "weight": weight,
                      "example_headline": surf_free.get(
                          surface, surf_example[surface]),
                      "link": surf_free_link.get(
                          surface, surf_example_link.get(surface, "")),
                      "group": _family_of(surface), "mined": True,
                      "canon": key,
                      "surfaces": dict(surf_counts[key])})
    folded = _fold_windows(mined, headlines, known_terms)
    # Template demotion: a mined phrase whose headlines live
    # inside one rotating-entity skeleton is a data template,
    # not a story. Window-fold products are exempt — the fold's
    # own guards already adjudicated them (the Yahoo explainer
    # case is pinned as a kept topic by its test).
    if folded:
        members = topic_member_sets(folded, headlines)
        folded = [t for t in folded
                  if "merged_from" in t or not _template_bound(
                      [headlines[i]["title"] for i in
                       members.get(t["term"], set())])]
    return folded


def _join_windows(a, b):
    """Join two canonical token tuples into one phrase when a
    suffix of one is a prefix of the other (overlap >= 1 token)
    and the join is strictly longer than both — i.e. the two
    grams are overlapping windows of a single longer phrase.
    Largest overlap wins; returns the joined tuple or None.
    Pure containment (one gram inside the other) is not a
    window join — the containment fold owns that case."""
    for k in range(min(len(a), len(b)), 0, -1):
        if a[len(a) - k:] == b[:k]:
            joined = a + b[k:]
            if len(joined) > max(len(a), len(b)):
                return joined
        if b[len(b) - k:] == a[:k]:
            joined = b + a[k:]
            if len(joined) > max(len(a), len(b)):
                return joined
    return None


# Stopwords that may sit INSIDE a genuine phrase — determiners
# and copula/auxiliaries ("what IS behind", "behind THE move").
# Any other stopword inside a gram ("violent IN france",
# "falls AS megacap") heads an adjunct: the gram crosses a
# phrase boundary, and window-joining through it would extend
# a topic into a trailing clause.
_WINDOW_OK_INTERNAL = {"the", "a", "an", "is", "are", "was",
                       "were", "be", "been"}


def _fold_windows(mined, headlines, known_terms):
    """Window fold: two mined grams that are overlapping windows
    of one recurring headline phrase render as two topics for a
    single story template (seen live 2026-11-01: Yahoo's
    explainer template "What Is Behind The Move In X Stock?"
    mined as both "What Is Behind" and "Behind The Move" —
    identical headline sets, neither gram containing the
    other's tokens, so the containment fold could not see
    them). The evidence bar is deliberately strict — a miss
    leaves the status quo, an over-merge invents a phrase no
    headline contains. A pair folds into the joined phrase
    iff ALL of: the grams have equal canonical length (two
    windows of one stem, not a topic and its tail); their
    member sets are IDENTICAL (co-extensive — one window
    living inside a larger topic's headlines is that topic's
    business, not a window); the joined phrase occurs
    contiguously in >= 80% of the shared headlines (the
    containment fold's guard constant); and neither gram
    carries an adjunct-heading stopword inside it (see
    _WINDOW_OK_INTERNAL — "protests turn violent" must never
    extend into "violent in France"). The survivor's label
    is the joined phrase itself — a real headline substring,
    taken from an actual headline's token span; its weight
    is the member set's size, and its surfaces pool both
    grams' surfaces plus the joined one, so member matching
    keeps counting every form. Two more exclusions: only a
    gram's INTERNAL tokens are judged for the stopword rule
    (edges are content by mining construction, so a bigram
    has no internal token to judge); and a gram carrying an
    ampersand is an entity name the tokenizer deliberately
    kept whole ("Procter & Gamble", "S&P 500") — entities
    are complete topics, never template windows, and never
    fold."""
    if len(mined) < 2:
        return mined
    titles = [" ".join(h["title"].lower().replace("-", " ").split())
              for h in headlines]
    streams = [_canon_tokens(_tokens(h["title"])) for h in headlines]
    raw_streams = [_tokens(h["title"]) for h in headlines]

    def _members(topic):
        hits = set()
        grams = topic.get("surfaces") or {topic["term"].lower(): 1}
        for gram in grams:
            g = gram.lower().replace("-", " ")
            for i, title in enumerate(titles):
                if g in title:
                    hits.add(i)
        return hits

    def _contains(stream, seq):
        stream, seq = list(stream), list(seq)
        return any(stream[i:i + len(seq)] == seq
                   for i in range(len(stream) - len(seq) + 1))

    def _overlaps(term, others):
        dk = term.lower().replace("-", "").replace(" ", "")
        for other in others:
            k = other.lower().replace("-", "").replace(" ", "")
            if dk in k or k in dk:
                return True
        return False

    topics = [dict(t) for t in mined]
    changed = True
    while changed:
        changed = False
        members = {t["term"]: _members(t) for t in topics}
        topics.sort(key=lambda t: (t["weight"], t["term"]))
        for i, x in enumerate(topics):
            if changed:
                break
            for y in topics[i + 1:]:
                if len(x["canon"]) != len(y["canon"]):
                    continue
                if any("&" in tok
                       for tok in x["canon"] + y["canon"]):
                    continue
                if any(tok in STOPWORDS
                       and tok not in _WINDOW_OK_INTERNAL
                       for tok in x["canon"][1:-1]
                       + y["canon"][1:-1]):
                    continue
                joined = _join_windows(x["canon"], y["canon"])
                if joined is None:
                    continue
                mx, my = members[x["term"]], members[y["term"]]
                if not mx or mx != my:
                    continue
                contig = [j for j in sorted(mx)
                          if _contains(streams[j], joined)]
                if not contig or len(contig) < \
                        FOLD_MIN_CONTAINMENT * len(mx):
                    continue
                # The joined surface, verbatim from the first
                # headline whose token stream carries it.
                j0 = contig[0]
                stream = list(streams[j0])
                pos = next(p for p in range(
                    len(stream) - len(joined) + 1)
                    if stream[p:p + len(joined)] == list(joined))
                surface = " ".join(raw_streams[j0][pos:pos + len(joined)])
                surfaces = dict(x.get("surfaces") or {})
                for s, c in (y.get("surfaces") or {}).items():
                    surfaces[s] = surfaces.get(s, 0) + c
                surfaces[surface] = len(contig)
                union = mx | my
                hits = [headlines[j] for j in sorted(union)]
                rep = _representative(hits)
                merged = {
                    "term": _display(surface.split(" ")),
                    "weight": len(union),
                    "example_headline": rep["title"],
                    "link": rep.get("link", ""),
                    "group": (x if x["weight"] >= y["weight"]
                              else y)["group"],
                    "mined": True, "canon": joined,
                    "surfaces": surfaces,
                    "merged_from":
                        (x.get("merged_from") or [x["term"]]) +
                        (y.get("merged_from") or [y["term"]])}
                others = [t["term"] for t in topics
                          if t is not x and t is not y]
                if _overlaps(merged["term"], known_terms):
                    # The joined phrase is a lexicon topic's
                    # territory: both windows are fragments of
                    # it — consume them, keep no mined copy.
                    topics = [t for t in topics
                              if t is not x and t is not y]
                elif _overlaps(merged["term"], others):
                    continue
                else:
                    topics = [t for t in topics
                              if t is not x and t is not y]
                    topics.append(merged)
                changed = True
                break
    topics.sort(key=lambda t: (-t["weight"], t["term"]))
    return topics


# ------------------------------------------------------------------
# Representative link choice
# ------------------------------------------------------------------
# Outlets whose articles sit behind a paywall. A topic's link
# prefers a non-paywalled source (user request 2026-11-01) and
# falls back to a paywalled one only when no free source covers
# the topic. Matched against the headline's source name — the
# feed's name for direct feeds, the publisher for Google News.
_PAYWALLED_PHRASES = ("wall street journal", "bloomberg",
                      "financial times", "nikkei", "marketwatch",
                      "barron", "seeking alpha", "economist",
                      "new york times", "washington post",
                      "telegraph", "fortune")
_PAYWALLED_TOKENS = {"wsj", "ft"}


def _paywalled(source):
    s = (source or "").lower()
    if any(p in s for p in _PAYWALLED_PHRASES):
        return True
    return any(tok in _PAYWALLED_TOKENS
               for tok in re.split(r"[^a-z]+", s))


def _representative(hits):
    """The headline a topic links to: the first non-paywalled
    hit, else the first hit."""
    for h in hits:
        if not _paywalled(h.get("source", "")):
            return h
    return hits[0]


def topic_member_sets(topics, headlines):
    """term -> set of headline indexes mentioning it, with
    canonical matching: lexicon topics match by their patterns
    (plus the patterns of any lexicon topic merged into them,
    recorded in merged_from); a mined topic matches by EVERY
    pooled surface form, not just its display form — so a
    merged topic's member set is the full union its weight
    counts. (news_cloud_links.topic_members is this function;
    member matching lives here because extraction itself —
    grouping and the containment fold — runs on member sets.)"""
    lex = {term: patterns for term, patterns, _ in LEXICON}
    titles = [" ".join(h["title"].lower().replace("-", " ").split())
              for h in headlines]
    members = {}
    for t in topics:
        term = t["term"]
        hits = set()
        pat_terms = [term] + [m for m in (t.get("merged_from") or [])
                              if m != term]
        for pt in pat_terms:
            for p in lex.get(pt, ()):
                for i, h in enumerate(headlines):
                    if _matches(p, h["title"].lower()):
                        hits.add(i)
        surfaces = t.get("surfaces")
        if surfaces:
            grams = set(surfaces)
        elif term not in lex:
            grams = {term}
        else:
            grams = set()
        for gram in grams:
            g = gram.lower().replace("-", " ")
            for i, title in enumerate(titles):
                if g in title:
                    hits.add(i)
        members[term] = hits
    return members


def _group_exact(topics, headlines):
    """Merge topics sharing an exact canonical key: weight =
    size of the UNION of their member sets (never a sum — a
    headline carrying both variants counts once); label = the
    highest-weight constituent's term (a lexicon constituent
    wins ties over a mined one)."""
    members = topic_member_sets(topics, headlines)
    by_key = {}
    for t in topics:
        by_key.setdefault(t["canon"], []).append(t)
    out = []
    for key, group in by_key.items():
        if len(group) == 1:
            out.append(group[0])
            continue
        union = set()
        surfaces = {}
        for t in group:
            union |= members[t["term"]]
            for s, c in (t.get("surfaces") or {}).items():
                surfaces[s] = surfaces.get(s, 0) + c
        label_t = sorted(group, key=lambda t: (
            -t["weight"], bool(t.get("mined")), t["term"]))[0]
        hits = [headlines[i] for i in sorted(union)]
        rep = _representative(hits)
        merged = {"term": label_t["term"], "weight": len(union),
                  "example_headline": rep["title"],
                  "link": rep.get("link", ""),
                  "group": label_t["group"],
                  "mined": any(t.get("mined") for t in group),
                  "canon": key,
                  "merged_from": [t["term"] for t in group]}
        if surfaces:
            merged["surfaces"] = surfaces
        out.append(merged)
    return out


# The containment fold fires only when the headline evidence
# backs the token evidence: at least this share of the SMALLER
# topic's headlines must sit inside the larger's.
FOLD_MIN_CONTAINMENT = 0.80


def _fold_containment(topics, headlines, members):
    """Containment fold: topic X folds into topic Y iff X's
    canonical token set is a strict subset of Y's AND >= 80% of
    the smaller topic's headlines are contained in the larger's
    (the guard — token-set inclusion alone would one day
    swallow a general topic into a narrow phrase that merely
    shares its words; the live case, the mined "Midterm
    Elections" fragment of the lexicon's "Midterms", passes at
    100%). The survivor keeps Y's canonical key, takes the
    higher-weight constituent's term as its label, and its
    weight becomes the size of the union of the two member
    sets. Folds iterate to a fixpoint, smallest topics first."""
    members = dict(members)
    topics = [dict(t) for t in topics]
    changed = True
    while changed:
        changed = False
        topics.sort(key=lambda t: (len(set(t["canon"])), t["weight"]))
        for x in list(topics):
            xs = set(x["canon"])
            supersets = [y for y in topics if y is not x
                         and xs < set(y["canon"])]
            if not supersets:
                continue
            y = sorted(supersets, key=lambda t: (
                -t["weight"], len(set(t["canon"])), t["term"]))[0]
            mx, my = members[x["term"]], members[y["term"]]
            smaller = min(len(mx), len(my))
            if (smaller == 0
                    or len(mx & my) < FOLD_MIN_CONTAINMENT * smaller):
                continue
            union = mx | my
            own = [(y["term"], y["weight"]), (x["term"], x["weight"])]
            label_t = sorted(own, key=lambda tw: (-tw[1], tw[0]))[0][0]
            label_group = y["group"] if label_t == y["term"] \
                else x["group"]
            hits = [headlines[i] for i in sorted(union)]
            rep = _representative(hits)
            surfaces = dict(y.get("surfaces") or {})
            for s, c in (x.get("surfaces") or {}).items():
                surfaces[s] = surfaces.get(s, 0) + c
            y.update({"weight": len(union),
                      "example_headline": rep["title"],
                      "link": rep.get("link", ""),
                      "merged_from":
                          (y.get("merged_from") or [y["term"]]) +
                          (x.get("merged_from") or [x["term"]])})
            if surfaces:
                y["surfaces"] = surfaces
            members[label_t] = union
            if label_t != y["term"]:
                y["term"] = label_t
                y["group"] = label_group
            topics.remove(x)
            changed = True
            break
    topics.sort(key=lambda t: (-t["weight"], t["term"]))
    return topics


def extract_topics(headlines, max_topics=30, min_weight=2):
    """Weight lexicon topics by distinct-headline mentions, merged
    with phrases mined from the whole sample on equal footing.
    Mined surface variants pool by canonical key, exact-key
    duplicates merge on headline unions, and a guarded
    containment fold absorbs a topic into a strict token
    superset when their headline sets agree. Returns topic
    dicts sorted by weight desc."""
    topics = []
    for term, patterns, group in LEXICON:
        hits = [h for h in headlines
                if any(_matches(p, h["title"].lower()) for p in patterns)]
        if len(hits) < min_weight:
            continue
        rep = _representative(hits)
        topics.append({"term": term, "weight": len(hits),
                       "example_headline": rep["title"],
                       "link": rep.get("link", ""),
                       "group": group,
                       "canon": _canon_of_term(term)})
    topics += _mine_phrases(headlines, [t["term"] for t in topics])
    topics = _group_exact(topics, headlines)
    members = topic_member_sets(topics, headlines)
    topics = _fold_containment(topics, headlines, members)
    # Story-cluster counting (2026-11-02): a topic's weight is
    # the number of distinct STORIES carrying it, not the number
    # of headlines — near-duplicate coverage of one story counts
    # once. (The fold above weights by headline union; this
    # recount is the final word.)
    members = topic_member_sets(topics, headlines)
    clusters = _story_clusters(headlines)
    for t in topics:
        t["weight"] = len({clusters[i]
                           for i in members.get(t["term"], set())})
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
.nc-window { font-size:14px; color:#9AA0A6; font-style:italic; }
.nc-cloud { flex:1; display:flex; flex-wrap:wrap; align-items:center;
  align-content:center; justify-content:center; column-gap:26px; row-gap:8px;
  padding:10px 4px; }
.nc-w { white-space:nowrap; line-height:1.08; display:inline-block;
  -webkit-background-clip:text; background-clip:text; color:transparent;
  filter:drop-shadow(0 1px 1px rgba(35,34,31,.22)); }
.nc-tilt { transform:rotate(-5deg); }
.nc-foot { border-top:1px solid #D8CBB6; padding-top:8px; font-size:13.5px;
  color:#9AA0A6; display:flex; justify-content:space-between; gap:16px; }
.nc-hl { color:#FFB066; font-weight:700; }
.nc-cloud a.nc-w { text-decoration:none; cursor:pointer; }
.nc-cloud a.nc-w:hover { opacity:.72; }
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
        style = ('font-size:{}px;font-weight:{};'
                 'background-image:linear-gradient(178deg,{} 15%,{} 85%)'
                 .format(sizes[term], 700 if t["weight"] >= 8 else 600,
                         light, deep))
        label = html_mod.escape(term)
        link = t.get("link") or ""
        if link:
            # Clickable topic (user request 2026-11-01): the
            # word links to its representative source article.
            spans.append(
                '<a class="{}" style="{}" href="{}" target="_blank" '
                'rel="noopener noreferrer" title="{}">{}</a>'.format(cls, style, html_mod.escape(link, quote=True),
                   html_mod.escape(t.get("example_headline", ""),
                                   quote=True),
                   label))
        else:
            spans.append('<span class="{}" style="{}">{}</span>'.format(cls, style, label))
    caption = caption or {}
    # Caption lines are trusted HTML (built by build_cloud from a
    # format string and integer counts — the date and counts carry
    # .nc-hl accent spans), so they are inserted unescaped.
    return ("<style>{}</style>"
            '<div class="nc-head"><div class="nc-title">What the market '
            'is talking about</div><div class="nc-window">past 24 hours'
            '</div></div><div class="nc-cloud">{}</div>'
            '<div class="nc-foot"><div>{}</div><div>{}</div></div>'.format(_FRAME_CSS, "\n".join(spans),
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
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#FFFFFF; }}
</style></head><body><div class="nc-frame"
style="width:1200px;height:600px;">{}</div></body></html>""".format(_frame_html(topics, caption, max_px=76))
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
        from toolbox.DataZoo.market.eastern import asof_stamp
        asof = asof_stamp(now)
    except Exception:
        asof = now.strftime("%Y-%m-%d %H:%M UTC")
    caption = {
        "left": 'As of <span class="nc-hl">{}</span> · '
                '<span class="nc-hl">{}</span> distinct headlines, '
                '<span class="nc-hl">{}</span> sources'
                .format(asof, len(headlines), len(source_counts)),
        "right": "Size = distinct stories mentioning the topic",
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
        "window": "past {} hours ending {}".format(hours, now.isoformat()),
        "headlines": len(headlines),
        "sources": dict(source_counts),
        "renderer": renderer,
        "caption": caption,
        "topics": [{"term": t["term"], "weight": t["weight"],
                    "group": t["group"],
                    "example_headline": t["example_headline"],
                    "link": t.get("link", "")}
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
    print("headlines: {} from {} sources | renderer: {}"
          .format(summary["headlines"], len(summary["sources"]),
                  summary["renderer"]))
    for t in summary["topics"][:12]:
        print("  {:3d}  {}".format(t["weight"], t["term"]))
    print("wrote", png, "and", js)
    return 0


if __name__ == "__main__":
    sys.exit(main())
