"""Prediction markets — the data behind the
Investments app's Prediction markets page (under Data;
user request 2026-11-10, demoed with live data and
approved the same day).

Source: Polymarket's public APIs, no key — the Gamma
API for events (gamma-api.polymarket.com) and the CLOB
API for price history (clob.polymarket.com). Prices
are market-implied probabilities (the Yes price); a
multi-outcome event's "leading outcome" is its
highest-priced one.

Two standing filters, applied to every board: sports
markets are excluded by tag (the user's no-sports
rule — raw volume order is otherwise all NBA games
and esports), and entertainment by tag (streamers,
pop culture, gaming — a GTA-marathon market reached
the demo's most-traded board untagged as sport).

The featured section is the NEXT Fed decision,
selected dynamically, never a hardcoded meeting
(the user's question at approval): among the active
"Fed Decision in..." events it takes the nearest
resolution date, so the section rolls December ->
January -> March on its own and is still the next
meeting in 2028. If no Fed event is live, the section
is simply absent. Every fetch degrades independently
(a dead category yields no board, a dead history
yields the outcomes table without its chart); the
payload as a whole never raises.

Public surface:

    fetch_payload()   {fed, top, cats} from the live
                      APIs (each piece optional)
    chart_blocks(p)   the page as ("html", str) /
                      ("fig", figure) blocks in order
"""

import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone

_GAMMA = "https://gamma-api.polymarket.com"
_CLOB = "https://clob.polymarket.com"
_UA = {"User-Agent": "Mozilla/5.0 (compatible; "
                     "Investments dashboard)"}

# Tag slugs that keep a market off every board: sports
# and entertainment (see the module docstring).
EXCLUDE_TAGS = frozenset({
    "sports", "nba", "nfl", "mlb", "nhl", "soccer",
    "epl", "tennis", "golf", "mma", "ufc", "esports",
    "cs2", "league-of-legends", "dota-2", "valorant",
    "f1", "racing", "boxing", "cricket", "rugby",
    "basketball", "football", "baseball", "hockey",
    "gaming", "pop-culture", "streamer", "live-streams",
    "entertainment", "celebrities", "movies", "tv",
    "music",
})

# (Gamma tag slug, board heading) in page order.
CATEGORIES = (
    ("politics", "Politics"),
    ("economy", "Economy & the Fed"),
    ("crypto", "Crypto"),
    ("geopolitics", "Geopolitics"),
    ("ai", "AI & tech"),
)

_INK = "#E8ECEC"
_MUTE = "#9AA0A6"
_BLUE = "#5B8DEF"


def _get(url):
    """Parsed JSON from a public API, or None on any
    failure (the module degrades; it never raises)."""
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except (OSError, ValueError):
        return None


def _events(tag=None, limit=40):
    q = {"active": "true", "closed": "false",
         "order": "volume24hr", "ascending": "false",
         "limit": str(limit)}
    if tag:
        q["tag_slug"] = tag
    out = _get(_GAMMA + "/events?"
               + urllib.parse.urlencode(q))
    return out if isinstance(out, list) else []


def _excluded(event):
    tags = {t.get("slug", "") for t in event.get("tags", [])}
    return bool(tags & EXCLUDE_TAGS)


def _yes_price(market):
    try:
        return float(json.loads(market["outcomePrices"])[0])
    except (ValueError, TypeError, KeyError, IndexError):
        return None


def _leading(event):
    """(label, prob, 24h change) of the event's
    highest-priced outcome, or None when no market in
    it has a parseable price. The label is the market's
    group item title ("25 bps increase", a candidate's
    name), or "Yes" for a plain binary market."""
    best = None
    for m in event.get("markets", []):
        p = _yes_price(m)
        if p is None:
            continue
        if best is None or p > best[1]:
            best = (m.get("groupItemTitle") or "Yes", p,
                    m.get("oneDayPriceChange") or 0.0)
    return best


def _row(event):
    if _excluded(event):
        return None
    lead = _leading(event)
    if lead is None:
        return None
    return {"title": event.get("title", ""),
            "lead": lead[0], "prob": lead[1],
            "chg": lead[2],
            "vol24": event.get("volume24hr", 0) or 0}


def _rows(events, n):
    out = []
    for e in events:
        r = _row(e)
        if r is not None:
            out.append(r)
            if len(out) >= n:
                break
    return out


def _parse_end(text):
    try:
        return datetime.fromisoformat(
            text.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _fed_feature(events, now=None):
    """The next Fed decision event from a Gamma event
    list: title "Fed Decision in...", a resolution date
    not yet past, the nearest such date (see the module
    docstring). Returns {heading, outcomes, vol24} with
    outcomes [(label, prob, chg, token)] sorted by prob
    desc, or None when no live Fed event qualifies."""
    now = now or datetime.now(timezone.utc)
    cands = []
    for e in events:
        if not e.get("title", "").startswith(
                "Fed Decision in"):
            continue
        end = _parse_end(e.get("endDate") or "")
        if end is None or end < now:
            continue
        cands.append((end, e))
    if not cands:
        return None
    end, ev = min(cands, key=lambda c: c[0])
    outcomes = []
    for m in ev.get("markets", []):
        p = _yes_price(m)
        if p is None:
            continue
        try:
            token = json.loads(m["clobTokenIds"])[0]
        except (ValueError, TypeError, KeyError, IndexError):
            token = None
        outcomes.append(
            (m.get("groupItemTitle") or m.get("question", ""),
             p, m.get("oneDayPriceChange") or 0.0, token))
    outcomes.sort(key=lambda o: -o[1])
    if not outcomes:
        return None
    return {"heading": "Fed decision — "
            + end.strftime("%B %Y") + " FOMC",
            "outcomes": outcomes,
            "vol24": ev.get("volume24hr", 0) or 0}


def _history(token):
    """Daily price history [(datetime, prob)] for one
    outcome token from the CLOB API; [] on failure."""
    if not token:
        return []
    out = _get(f"{_CLOB}/prices-history?market={token}"
               "&interval=1m&fidelity=1440")
    if not isinstance(out, dict):
        return []
    return [(datetime.fromtimestamp(p["t"],
                                    tz=timezone.utc),
             float(p["p"]))
            for p in out.get("history", [])
            if "t" in p and "p" in p]


def fetch_payload():
    """The live page payload: the featured next Fed
    decision (with price history for its top three
    outcomes), the most-traded board, and one board per
    category. Every piece degrades on its own."""
    top = _rows(_events(limit=60), 10)
    cats = {}
    fed = None
    for tag, name in CATEGORIES:
        evs = _events(tag=tag, limit=30)
        cats[name] = _rows(evs, 6)
        if tag == "economy":
            fed = _fed_feature(evs)
    if fed is not None:
        fed["hist"] = {
            label: _history(token)
            for label, _, _, token in fed["outcomes"][:3]}
    return {"fed": fed, "top": top, "cats": cats}


def _money(v):
    if v >= 1e6:
        return f"${v / 1e6:.1f}M"
    if v >= 1e3:
        return f"${v / 1e3:.0f}K"
    return f"${v:.0f}"


def _pct(p):
    return f"{p * 100:.1f}%" if p < 0.995 else f"{p * 100:.0f}%"


def _chg(c):
    pts = c * 100
    col = ("#7FB069" if pts > 0.05 else
           "#E4572E" if pts < -0.05 else _MUTE)
    return f'<span style="color:{col}">{pts:+.1f}</span>'


def _bar(prob, color=_BLUE):
    w = max(prob * 100, 1)
    return ('<div style="height:5px;background:#22303e;'
            'border-radius:3px;margin-top:4px">'
            f'<div style="height:5px;border-radius:3px;'
            f'background:{color};width:{w:.0f}%">'
            '</div></div>')


_TH = ('style="text-align:left;color:' + _MUTE + ';'
       'font-weight:600;font-size:11px;'
       'text-transform:uppercase;letter-spacing:.04em;'
       'padding:5px 8px;border-bottom:1px solid #26313d"')
_THN = _TH.replace("text-align:left", "text-align:right")
_TD = ('style="padding:6px 8px;'
       'border-bottom:1px solid #1b2632"')
_TDN = _TD[:-1] + ';text-align:right"'
_H2 = ('style="font-size:17px;font-weight:700;'
       'color:#FAFAFA;margin:22px 0 6px"')


def _board(rows, heading):
    trs = "".join(
        f'<tr><td {_TD}><div style="font-weight:600">'
        f'{r["title"]}</div>'
        f'<div style="color:{_MUTE};font-size:12px">'
        f'Leading: {r["lead"]}</div>{_bar(r["prob"])}</td>'
        f'<td {_TDN}><span style="font-weight:700;'
        f'font-size:14.5px">{_pct(r["prob"])}</span></td>'
        f'<td {_TDN}>{_chg(r["chg"])}</td>'
        f'<td {_TDN}>{_money(r["vol24"])}</td></tr>'
        for r in rows)
    return (f'<div {_H2}>{heading}</div>'
            '<table style="width:100%;'
            'border-collapse:collapse;font-size:13px">'
            f'<tr><th {_TH}>Market</th>'
            f'<th {_THN}>Prob</th>'
            f'<th {_THN}>24h Δ (pts)</th>'
            f'<th {_THN}>24h vol</th></tr>'
            f'{trs}</table>')


def _fed_table(fed):
    trs = "".join(
        f'<tr><td {_TD}><div style="font-weight:600">'
        f'{label}</div>{_bar(prob, "#E4572E")}</td>'
        f'<td {_TDN}><span style="font-weight:700;'
        f'font-size:14.5px">{_pct(prob)}</span></td>'
        f'<td {_TDN}>{_chg(chg)}</td></tr>'
        for label, prob, chg, _ in fed["outcomes"])
    return (f'<div {_H2}>{fed["heading"]}</div>'
            '<table style="width:100%;'
            'border-collapse:collapse;font-size:13px">'
            f'<tr><th {_TH}>Outcome</th>'
            f'<th {_THN}>Prob</th>'
            f'<th {_THN}>24h Δ (pts)</th></tr>'
            f'{trs}</table>')


_OUTCOME_COLORS = (("increase", "#E4572E"),
                   ("decrease", "#7FB069"),
                   ("No change", _BLUE))


def fed_fig(fed):
    """The featured Fed event's price-history figure:
    the top three outcomes' daily implied probabilities
    (house Plotly style — the shared plotly_panels
    layout, the % scale inside on chips). None when no
    outcome has any history."""
    import plotly.graph_objects as go

    from toolbox.DataZoo.market.plotly_panels import (
        add_y_chips,
        panel_layout,
        y_axis_dict,
    )

    series = [(label, fed["hist"][label])
              for label, *_ in fed["outcomes"][:3]
              if fed.get("hist", {}).get(label)]
    if not series:
        return None
    fig = go.Figure()
    for rank, (label, pts) in enumerate(series):
        color = next(
            (c for key, c in _OUTCOME_COLORS if key in label),
            ("#E3B341", "#B678D8", "#5B8DEF")[rank % 3])
        fig.add_trace(go.Scatter(
            x=[d for d, _ in pts], y=[p * 100 for _, p in pts],
            mode="lines", name=label,
            line=dict(color=color, width=2),
            hovertemplate="%{x|%b %d}: %{y:.0f}%<extra>"
            + label + "</extra>"))
    x0 = min(d for _, pts in series for d, _ in pts)
    xaxis = dict(showgrid=False, automargin=False,
                 tickfont=dict(color=_MUTE, size=10))
    layout = panel_layout(300, xaxis)
    layout["yaxis"] = y_axis_dict([0, 50, 100], (0, 100))
    fig.update_layout(**layout)
    add_y_chips(fig, [0, 50, 100], lambda v: f"{v:g}%",
                x_at=x0)
    return fig


def chart_blocks(payload):
    """The page as render blocks — ("html", str) and
    ("fig", figure) in page order: the lede, the
    featured Fed section (its outcomes table, then the
    history figure when history exists), the
    most-traded board, one board per non-empty
    category, and the source note. Each html block
    carries the centered 1120px measure."""
    def wrap(inner):
        return ('<div style="max-width:1120px;'
                'margin:0 auto">' + inner + "</div>")

    lede = ('<div style="font-size:12.5px;color:' + _MUTE
            + ';margin:2px 0 4px">Market-implied '
            'probabilities from Polymarket, on a '
            '15-minute cache. Sports and entertainment '
            'markets are excluded.</div>')
    blocks = [("html", wrap(lede))]
    fed = payload.get("fed")
    if fed:
        blocks.append(("html", wrap(_fed_table(fed))))
        fig = fed_fig(fed)
        if fig is not None:
            blocks.append(("fig", fig))
    if payload.get("top"):
        blocks.append(("html", wrap(
            _board(payload["top"],
                   "Most traded — ex-sports"))))
    for _, name in CATEGORIES:
        rows = payload.get("cats", {}).get(name) or []
        if rows:
            blocks.append(("html", wrap(_board(rows, name))))
    note = ('<div style="font-size:11.5px;color:#8A8378;'
            'margin-top:20px;line-height:1.5">Definitions: '
            'probabilities are Polymarket prices — the Yes '
            'price of each event\'s leading outcome '
            '(highest-priced of its outcomes). Δ is that '
            'outcome\'s 24-hour price change in percentage '
            'points; volume is the event\'s trailing-24h '
            'traded volume. The featured Fed section '
            'selects the next FOMC decision event by '
            'nearest resolution date and rolls forward as '
            'meetings resolve; its chart is the CLOB daily '
            'price history of the top three outcomes. '
            'Prediction-market prices are traded prices, '
            'not forecasts — they can be thin or wrong.'
            '</div>')
    blocks.append(("html", wrap(note)))
    return blocks
