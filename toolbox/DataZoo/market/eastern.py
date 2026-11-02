"""The one Eastern-time convention for the dashboard's market data.

User direction 2026-11-01: everything in the Investments app reads
in Eastern time. Every clock stamp these modules produce is rendered
in America/New_York with the zone's own abbreviation (EST/EDT, via
``%Z`` — never a hard-coded literal), and every calendar date
attached to a data point is the America/New_York calendar date of
that measurement.

This module is the single home of that convention: the board, the
sentiment block, the power law and the news cloud all stamp through
:func:`asof_stamp` instead of keeping a per-module zone constant
(the arrangement that left the app reading "CT" while the user
reads Eastern), and instant-stamped daily series (Yahoo's
UTC-midnight crypto bars, alternative.me's 00:00 UTC values) are
re-dated through :func:`utc_index_to_et_dates`.

A new module on purpose (the codebase's module-cache lesson): the
consumers import these names from here, and a fresh module has no
stale cache entry in the hosted app's running process.
"""

import datetime
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")

#: The Markets page's one as-of format ("Sun Nov 1, 2:36 PM EST").
ASOF_FORMAT = "%a %b %-d, %-I:%M %p %Z"


def now_et():
    """The current instant in America/New_York."""
    return datetime.datetime.now(ET)


def asof_stamp(when=None):
    """Format ``when`` (default: now) as the shared as-of stamp in
    Eastern time, e.g. "Sun Nov 1, 2:36 PM EST" — the abbreviation
    comes from the zone, so it flips between EST and EDT on its own.
    An aware datetime is converted to Eastern; a naive one is taken
    to already be Eastern wall time (the board's injectable clock).
    """
    if when is None:
        when = now_et()
    elif when.tzinfo is None:
        when = when.replace(tzinfo=ET)
    else:
        when = when.astimezone(ET)
    return when.strftime(ASOF_FORMAT)


def utc_index_to_et_dates(idx):
    """Re-date a tz-naive index of UTC instants to Eastern calendar
    dates (midnight-normalized, tz-naive).

    For daily series whose bars are stamped at 00:00 UTC (Yahoo
    BTC-USD, alternative.me): the instant falls on the prior
    Eastern evening (19:00 EST / 20:00 EDT), so the bar belongs to
    that Eastern day — the 2026-11-02 00:00 UTC bar reads
    2026-11-01. Values are untouched; only the dates move.
    """
    idx = pd.DatetimeIndex(idx)
    return idx.tz_localize("UTC").tz_convert(ET).normalize() \
        .tz_localize(None)
