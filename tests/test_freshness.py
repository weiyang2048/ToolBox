"""Tests for toolbox.DataZoo.market.freshness (network-free)."""
import datetime

from toolbox.DataZoo.market.freshness import (
    lag_pill_html,
    parse_payload_date,
    sessions_behind,
)

BOARD = datetime.date(2026, 11, 2)  # Monday


def test_parse_payload_date_formats():
    assert parse_payload_date("2026-10-30") == datetime.date(2026, 10, 30)
    assert parse_payload_date("Fri Oct 30, 2026") == datetime.date(2026, 10, 30)
    assert parse_payload_date(datetime.date(2026, 10, 30)) == datetime.date(2026, 10, 30)
    assert parse_payload_date(None) is None
    assert parse_payload_date("n/a") is None


def test_sessions_behind_counts_weekdays():
    # The live 2026-11-02 audit's three lags, board = Mon Nov 2.
    assert sessions_behind(BOARD, datetime.date(2026, 11, 2)) == 0
    assert sessions_behind(BOARD, datetime.date(2026, 10, 30)) == 1
    assert sessions_behind(BOARD, datetime.date(2026, 10, 29)) == 2
    assert sessions_behind(BOARD, datetime.date(2026, 10, 27)) == 4
    # A weekend data date counts the sessions after it.
    assert sessions_behind(BOARD, datetime.date(2026, 10, 31)) == 1
    assert sessions_behind(BOARD, None) is None


def test_lag_pill_html():
    cur = lag_pill_html(BOARD, datetime.date(2026, 11, 2))
    assert "current" in cur and "#14532D" in cur and "#A7F3C7" in cur
    lag = lag_pill_html(BOARD, datetime.date(2026, 10, 30))
    assert "1 session behind the board" in lag
    assert "#7C2D12" in lag and "#FFD9B3" in lag
    lag2 = lag_pill_html(BOARD, "2026-10-29")
    assert "2 sessions behind the board" in lag2
    assert lag_pill_html(BOARD, None) == ""
