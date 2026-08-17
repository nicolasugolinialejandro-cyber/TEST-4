# 🛡️ Crisis Tail-Risk & Volatility Shield

**Deployed app:** https://<your-app>.streamlit.app  ← *paste your Streamlit Community Cloud URL here and in the presentation PDF*

## The user and the decision

**User:** a high-net-worth individual with a **$1M+ liquid growth portfolio** who is
willing to keep most of the equity upside but cannot tolerate another 50% drawdown.

**Decision:** *what mix of SPY, TLT, GLD and managed futures (DBMF) minimises the
peak-to-trough loss in a severe market crash, subject to an agreed tracking-error
budget versus the equity market?*

The app answers it: it enumerates 12,341 allocations on a 2.5% weight grid, replays
each one through six historical crises, discards those that drift too far from SPY,
and returns the allocation with the shallowest worst-case crisis drawdown - next to
the mandated **equal-weight (25/25/25/25)** comparison and a 100% SPY baseline.

## Asset universe

| Ticker | Role |
|---|---|
| SPY | US equity growth engine and tracking benchmark |
| TLT | Long Treasuries - flight-to-quality hedge (failed in 2022) |
| GLD | Gold - inflation and currency-shock hedge |
| DBMF | Managed futures / trend following - crisis-alpha sleeve |

## Data provenance

| Field | Value |
|---|---|
| Provider | Yahoo Finance via `yfinance` |
| Field | Adjusted close (`auto_adjust=True` → total returns) |
| Frequency | Daily |
| Range | 2007-02-22 → retrieval date |
| Fallback | `data/prices_snapshot.csv` (frozen copy, used if Yahoo is unreachable) |

DBMF launched 2019-05-08, so before that date the managed-futures sleeve uses the
chain-linked **returns** of RYMFX (Guggenheim Managed Futures Strategy Fund, live
since 2007). Full detail: [`data/README_DATA.md`](data/README_DATA.md).

## Method

1. **Clean** - drop duplicate/non-positive prices, forward-fill at most 3 days,
   drop any date without all four prices (`src/data.py`).
2. **Measure** - CAGR, volatility, max drawdown, Sharpe, Sortino, VaR/CVaR 95%,
   Ulcer index, beta and tracking error vs SPY (`src/metrics.py`).
3. **Optimise** - max drawdown is path-dependent and non-convex, so instead of a
   gradient solver we enumerate the whole weight simplex on a 2.5% grid and pick
   the global best within the tracking-error budget and the policy weight limits
   (`src/optimizer.py`, vectorised in `src/fastcalc.py`).
4. **Compare** - always against equal weight and 100% SPY, in every crisis.

## Interaction (each control changes the answer)

Tracking-error budget · equity floor/ceiling · concentration cap per diversifier ·
worst-case vs average-crisis objective · rebalancing frequency (monthly → buy & hold) ·
which crises the shield must survive · portfolio size in dollars · live data vs offline
snapshot · a manual "build your own mix" comparator.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Verification

```bash
python verify.py     # 6 independent checks → VERIFICATION_OUTPUT.md
pytest -q            # 8 unit tests on the calculation engines
```

See [`VERIFICATION_OUTPUT.md`](VERIFICATION_OUTPUT.md) for the last run and
[`AI_USE_DISCLOSURE.md`](AI_USE_DISCLOSURE.md).

## Limitations

* Six crisis episodes is a small sample; the weights are a robust *region*, not a point.
* Pre-2019 managed futures is a proxy fund, not DBMF.
* No fees, spreads, taxes, slippage or market impact.
* Bond behaviour changed between 2008 and 2022; past drawdowns do not cap future ones.
* Educational project, not investment advice.

## Repository layout

```
app.py                  Streamlit app (UI, narrative, charts)
src/config.py           Universe, crisis windows, provenance
src/data.py             Download, splice, clean
src/metrics.py          Readable one-portfolio metrics
src/fastcalc.py         Vectorised engine for the grid search
src/optimizer.py        Simplex grid search under a TE budget
verify.py               Independent verification script
tests/test_core.py      Unit tests
data/                   Frozen price snapshot + data documentation
```
