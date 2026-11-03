"""Freshness stamps for the Markets page's secondary tabs.

The boards stamp their pull time; a secondary tab (US vs ex-US,
The dollar, Bitcoin power law) must instead be dated by its own
last real measurement — and when that date trails the board
date, the stamp says so (user-approved demo 2026-11-02, wired
the same day). This module is the one home of that arithmetic:
trading-session lag (weekdays between the data date and the
board date) and the lag pill's HTML, shared by the ToolBox
renderers and the dashboard page so the two can never drift.

A new module on purpose (the codebase's module-cache lesson):
the hosted app caches imported modules across deploys, and a
fresh module imports clean.
"""
import datetime

LAG_BG, LAG_FG = "#7C2D12", "#FFD9B3"
OK_BG, OK_FG = "#14532D", "#A7F3C7"


def parse_payload_date(value):
    """A payload's last date as a datetime.date.

    Accepts a date/datetime, an ISO date string ("2026-10-30"),
    and the renderers' display format ("Fri Oct 30, 2026").
    Returns None when nothing parses."""
    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%a %b %d, %Y", "%b %d, %Y"):
        try:
            return datetime.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def sessions_behind(board_date, data_date):
    """Trading sessions (Mon-Fri) in (data_date, board_date].

    Mon Nov 2 board vs Fri Oct 30 data = 1; vs Thu Oct 29 = 2;
    vs Tue Oct 27 = 4. Zero when the data date is the board
    date (or later). Market holidays are not modelled — the
    pill counts weekday sessions, which is the granularity the
    page's dates carry."""
    board_date = parse_payload_date(board_date)
    data_date = parse_payload_date(data_date)
    if board_date is None or data_date is None:
        return None
    n, d = 0, data_date
    while d < board_date:
        d += datetime.timedelta(days=1)
        if d.weekday() < 5:
            n += 1
    return n


def lag_pill_html(board_date, data_date):
    """The stamp's pill: green "current" when the tab's date is
    the board date, burnt-orange "N session(s) behind the
    board" when it trails. "" when either date is unknown."""
    n = sessions_behind(board_date, data_date)
    if n is None:
        return ""
    if n == 0:
        bg, fg, text = OK_BG, OK_FG, "current"
    else:
        bg, fg = LAG_BG, LAG_FG
        text = f"{n} session{'s' if n != 1 else ''} behind the board"
    return ('<span style="display:inline-block;'
            f'background:{bg};color:{fg};font-size:11.5px;'
            'font-weight:700;border-radius:999px;'
            'padding:2px 9px;margin-left:8px;'
            f'vertical-align:1px">{text}</span>')
