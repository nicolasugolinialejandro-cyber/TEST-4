"""Central configuration: asset universe, stress windows, and data provenance.

Keeping these constants in one module means the app, the tests and the
verification script always describe the same portfolio problem.
"""

from dataclasses import dataclass

# --- Asset universe -------------------------------------------------------
# Four liquid, low-cost ETFs that a high-net-worth investor can actually hold.
TICKERS = ["SPY", "TLT", "GLD", "DBMF"]

ASSET_LABELS = {
    "SPY": "US Equity (SPY)",
    "TLT": "Long Treasuries (TLT)",
    "GLD": "Gold (GLD)",
    "DBMF": "Managed Futures (DBMF)",
}

BENCHMARK = "SPY"  # equity sleeve the investor is trying to keep up with

# DBMF launched 2019-05-08. To stress-test the strategy through 2008 we splice
# in RYMFX (Rydex/Guggenheim Managed Futures Strategy, live since 2007-02-22),
# a documented managed-futures mutual fund tracking a trend-following index.
MANAGED_FUTURES_PROXY = "RYMFX"
DBMF_INCEPTION = "2019-05-08"

# Analysis window starts when *all* four series (incl. the proxy) exist.
DEFAULT_START = "2007-02-22"

TRADING_DAYS = 252

# --- Historical stress episodes ------------------------------------------
# Peak-to-trough equity drawdown episodes, dated on SPY closes.
STRESS_WINDOWS = {
    "GFC 2007-2009": ("2007-10-09", "2009-03-09"),
    "Euro crisis 2011": ("2011-04-29", "2011-10-03"),
    "China/oil selloff 2015-16": ("2015-05-21", "2016-02-11"),
    "Q4 2018 tightening": ("2018-09-20", "2018-12-24"),
    "COVID crash 2020": ("2020-02-19", "2020-03-23"),
    "2022 inflation shock": ("2022-01-03", "2022-10-12"),
}


@dataclass(frozen=True)
class Provenance:
    """Where the data came from - shown in the app and the README."""

    provider: str = "Yahoo Finance via the yfinance Python package"
    field: str = "Adjusted close (auto_adjust=True: splits and dividends reinvested)"
    frequency: str = "Daily"
    instruments: str = ", ".join(TICKERS) + f" (+ {MANAGED_FUTURES_PROXY} pre-{DBMF_INCEPTION})"


PROVENANCE = Provenance()
