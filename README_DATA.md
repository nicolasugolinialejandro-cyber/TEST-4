# Data provenance

| Field | Value |
|---|---|
| Provider | Yahoo Finance, retrieved with the `yfinance` Python package |
| Instruments | SPY, TLT, GLD, DBMF (spliced with RYMFX before 2019-05-08) |
| Field | Adjusted close (`auto_adjust=True`: splits and dividends reinvested) |
| Frequency | Daily |
| Date range | 2007-02-22 to 2026-08-17 |
| Retrieval date | 2026-08-17 |
| Rows | 4902 |

`prices_snapshot.csv` is a frozen copy of exactly this download. The app tries a
live `yfinance` download first and falls back to this file if Yahoo Finance is
unreachable, so the grader can always run the app and always reproduce the
numbers in the presentation.

## Managed-futures splice

DBMF (iMGP DBi Managed Futures Strategy ETF) only launched on 2019-05-08, which
would exclude the 2008 crisis - the single most important stress test for this
investor. Before that date we use the *return stream* of RYMFX (Guggenheim
Managed Futures Strategy Fund, live since 2007-02-22), chain-linked onto DBMF's
first traded price. Both track diversified trend-following managed futures.
The splice is disclosed in the app and the pre-2019 sleeve should be read as
"a managed-futures allocation", not "DBMF itself".
