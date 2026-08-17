# Verification output

- Data source: Yahoo Finance (live download)
- Retrieval date: 2026-08-17
- Sample: 2007-02-22 to 2026-08-17 (4902 daily observations)
- Crisis windows used: GFC 2007-2009, Euro crisis 2011, China/oil selloff 2015-16, Q4 2018 tightening, COVID crash 2020, 2022 inflation shock

## Checks

| Check | Result | Detail |
|---|---|---|
| Buy-and-hold total return from raw prices | PASS | prices 325.720260% vs engine 325.720260% |
| Max drawdown vs brute-force double loop | PASS | brute -13.8240% vs engine -13.8371% (5-day sampled path) |
| fastcalc vs metrics max drawdown | PASS | fast -0.1383714483 vs metrics -0.1383714483 |
| Tracking error: manual vs metrics vs fastcalc | PASS | 0.17056884 / 0.17056884 / 0.17056884 |
| 100% SPY portfolio reproduces SPY exactly | PASS | max abs deviation 2.22e-16 |
| Edge case: crisis window with no data | PASS | optimiser degrades gracefully with no usable crisis data |
| Edge case: impossible tracking-error budget | PASS | flagged as infeasible instead of returning a silently wrong answer |

## Headline result reproduced by this script

- Shield weights: {'SPY': 0.55, 'TLT': 0.1, 'GLD': 0.1, 'DBMF': 0.25}
- Worst crisis drawdown: -30.13%
- Tracking error vs SPY: 9.99%
- Equal-weight max drawdown (full sample): -13.84%
- SPY max drawdown (full sample): -55.19%

## What would make the conclusion unreliable

- Only six crisis episodes exist in the sample; the optimiser can fit them.
- Pre-2019 managed futures is RYMFX, not DBMF (see data/README_DATA.md).
- Bond/gold behaviour in 2022 differed from 2008; future crises may differ again.
- Costs, taxes, bid-ask spreads and slippage are not modelled.
