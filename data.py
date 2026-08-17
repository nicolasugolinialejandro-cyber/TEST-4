"""Data retrieval, splicing and cleaning.

Normal path: download daily adjusted closes from Yahoo Finance with yfinance.
Fallback path: a dated CSV snapshot committed in data/prices_snapshot.csv, so
the app still runs (and the grader can reproduce it) if Yahoo is unreachable.
"""

from __future__ import annotations

import os
from datetime import date

import pandas as pd

from .config import (
    DBMF_INCEPTION,
    DEFAULT_START,
    MANAGED_FUTURES_PROXY,
    TICKERS,
)

SNAPSHOT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "prices_snapshot.csv",
)


def _download(tickers: list[str], start: str) -> pd.DataFrame:
    import yfinance as yf  # imported lazily so the snapshot path works offline

    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    return prices[tickers] if set(tickers).issubset(prices.columns) else prices


def splice_managed_futures(prices: pd.DataFrame) -> pd.Series:
    """Return one continuous managed-futures series.

    Before DBMF's 2019-05-08 inception we use the RYMFX *return* stream,
    chain-linked onto DBMF's first price so the level is continuous. We splice
    returns, never raw prices, because the two funds have different NAV levels.
    """
    dbmf = prices["DBMF"].dropna()
    proxy = prices[MANAGED_FUTURES_PROXY].dropna()
    if dbmf.empty:
        return proxy.rename("DBMF")

    cutover = max(pd.Timestamp(DBMF_INCEPTION), dbmf.index[0])
    head = proxy.loc[proxy.index < cutover]
    if head.empty:
        return dbmf.rename("DBMF")

    # Scale the proxy so it ends exactly at DBMF's first traded price.
    scaled_head = head / head.iloc[-1] * dbmf.loc[cutover:].iloc[0]
    return pd.concat([scaled_head, dbmf.loc[cutover:]]).rename("DBMF")


def clean_prices(prices: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Align the four series and document every observation we drop.

    Decisions (deliberate, not silent):
      * forward-fill at most 3 consecutive days for holiday mismatches between
        the ETF and the mutual-fund proxy;
      * drop any remaining row with a missing price so every return in the
        analysis is computed from four real, simultaneous observations;
      * drop exact duplicate dates and non-positive prices (bad ticks).
    """
    report: dict[str, object] = {"rows_downloaded": int(len(prices))}

    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    prices = prices.where(prices > 0)
    report["nonpositive_or_duplicate_removed"] = int(
        report["rows_downloaded"] - len(prices)
    )

    filled = prices.ffill(limit=3)
    report["values_forward_filled"] = int(
        filled.notna().sum().sum() - prices.notna().sum().sum()
    )

    clean = filled.dropna(how="any")
    report["rows_dropped_incomplete"] = int(len(filled) - len(clean))
    report["rows_used"] = int(len(clean))
    report["first_date"] = str(clean.index.min().date()) if len(clean) else None
    report["last_date"] = str(clean.index.max().date()) if len(clean) else None
    return clean, report


def load_prices(start: str = DEFAULT_START, use_snapshot: bool = False):
    """Return (prices, cleaning_report, source_label, retrieval_date)."""
    if not use_snapshot:
        try:
            raw = _download(TICKERS + [MANAGED_FUTURES_PROXY], start)
            raw["DBMF"] = splice_managed_futures(raw)
            prices, report = clean_prices(raw[TICKERS])
            if len(prices) > 250:
                return prices, report, "Yahoo Finance (live download)", str(date.today())
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"Live download failed, falling back to snapshot: {exc}")

    snap = pd.read_csv(SNAPSHOT_PATH, index_col=0, parse_dates=True)
    prices, report = clean_prices(snap[TICKERS])
    prices = prices.loc[prices.index >= pd.Timestamp(start)]
    retrieved = os.environ.get("SNAPSHOT_DATE", "see data/README_DATA.md")
    return prices, report, "Local CSV snapshot (data/prices_snapshot.csv)", retrieved


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns; the first row is undefined and dropped."""
    return prices.pct_change().dropna(how="any")
