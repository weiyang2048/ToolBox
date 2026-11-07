"""The DataZoo.market package loads its re-exports
lazily (PEP 562, user direction 2026-11-06): importing
the package must not import any submodule, so the
package import lock is held only momentarily. Runs in
a fresh interpreter — the suite's own process has the
submodules loaded already."""
import subprocess
import sys

_PROBE = """
import sys

import toolbox.DataZoo.market as m

heavy = [k for k in sys.modules
         if k.startswith("toolbox.DataZoo.market.")]
assert not heavy, heavy

assert m.FearGreed.__module__ == \\
    "toolbox.DataZoo.market.fear_greed"
assert "toolbox.DataZoo.market.fear_greed" in sys.modules
assert "toolbox.DataZoo.market.zillow_data" \\
    not in sys.modules

from toolbox.DataZoo.market import (  # noqa: F401
    CNN_SERIES_LABELS, StockIndices, ZillowZipData,
    board_scales, download_close_prices, fetch_board,
    fetch_json, normalize_close_prices, shade_zero,
    yahoo_daily)
from toolbox.DataZoo.market import fear_greed  # noqa: F401
assert m.futures_board.fetch_board is fetch_board
try:
    m.no_such_name
except AttributeError:
    pass
else:
    raise AssertionError("missing name must raise")
print("lazy-ok")
"""


def test_market_package_imports_lazily():
    out = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True, text=True, timeout=120, check=False)
    assert out.returncode == 0, out.stderr
    assert "lazy-ok" in out.stdout
