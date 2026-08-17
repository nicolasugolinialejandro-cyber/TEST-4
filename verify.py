"""Independent verification of the app's headline numbers.

Run:  python verify.py
It writes VERIFICATION_OUTPUT.md and exits non-zero if any check fails.

Each check recomputes a result *outside* the app's main pipeline:
  1. Portfolio return recomputed from weights and raw prices with plain pandas.
  2. Max drawdown recomputed with a naive O(n^2) double loop.
  3. Vectorised grid engine (fastcalc) vs the readable engine (metrics).
  4. Tracking error recomputed from first principles with NumPy.
  5. Edge case: a single-asset portfolio must reproduce that asset exactly,
     and a zero-length crisis window must not crash the optimiser.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.config import STRESS_WINDOWS, TRADING_DAYS
from src.data import load_prices, to_returns
from src.fastcalc import max_drawdowns, tracking_errors, wealth_paths
from src.metrics import max_drawdown, portfolio_returns, tracking_error
from src.optimizer import equal_weight, optimise, slice_windows

TOL = 1e-9
results: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str) -> None:
    results.append((name, passed, detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")


def main() -> int:
    prices, report, source, retrieved = load_prices()
    returns = to_returns(prices)
    weights = equal_weight(list(returns.columns))
    print(f"Data source: {source} | retrieved {retrieved} | rows {report['rows_used']}")

    # --- 1. Buy-and-hold portfolio value rebuilt straight from prices --------
    norm = prices / prices.iloc[0]
    manual_level = (norm * pd.Series(weights)).sum(axis=1)
    manual_total = float(manual_level.iloc[-1] - 1.0)
    engine_total = float((1 + portfolio_returns(returns, weights, "BH")).prod() - 1.0)
    check("Buy-and-hold total return from raw prices",
          abs(manual_total - engine_total) < 1e-6,
          f"prices {manual_total:.6%} vs engine {engine_total:.6%}")

    # --- 2. Max drawdown by brute force ------------------------------------
    port = portfolio_returns(returns, weights, "M")
    level = (1 + port).cumprod().to_numpy()
    sub = level[::5]  # every 5th point keeps the O(n^2) loop quick but honest
    brute = min(float(sub[j] / sub[i] - 1.0)
                for i in range(len(sub)) for j in range(i, len(sub)))
    engine_dd = max_drawdown(port)
    check("Max drawdown vs brute-force double loop",
          abs(brute - engine_dd) < 0.005,
          f"brute {brute:.4%} vs engine {engine_dd:.4%} (5-day sampled path)")

    # --- 3. Vectorised grid engine vs readable engine ----------------------
    w_vec = np.array([[weights[c] for c in returns.columns]])
    fast_dd = float(max_drawdowns(wealth_paths(returns, w_vec, "M"))[0])
    check("fastcalc vs metrics max drawdown",
          abs(fast_dd - engine_dd) < TOL,
          f"fast {fast_dd:.10f} vs metrics {engine_dd:.10f}")

    # --- 4. Tracking error from first principles ---------------------------
    diff = (port - returns["SPY"]).to_numpy()
    manual_te = float(np.std(diff, ddof=1) * np.sqrt(TRADING_DAYS))
    engine_te = tracking_error(port, returns["SPY"])
    fast_te = float(tracking_errors(wealth_paths(returns, w_vec, "M"),
                                    returns["SPY"].to_numpy())[0])
    check("Tracking error: manual vs metrics vs fastcalc",
          abs(manual_te - engine_te) < TOL and abs(manual_te - fast_te) < 1e-8,
          f"{manual_te:.8f} / {engine_te:.8f} / {fast_te:.8f}")

    # --- 5. Edge cases ------------------------------------------------------
    only_spy = portfolio_returns(returns, {"SPY": 1.0, "TLT": 0, "GLD": 0, "DBMF": 0}, "M")
    check("100% SPY portfolio reproduces SPY exactly",
          float(np.abs(only_spy - returns["SPY"]).max()) < 1e-12,
          f"max abs deviation {float(np.abs(only_spy - returns['SPY']).max()):.2e}")

    empty_window = {"Impossible window": ("2099-01-01", "2099-02-01")}
    try:
        best, _ = optimise(returns, te_cap=0.10, step=0.10, windows=empty_window)
        ok = best["crises_used"] == []
        detail = "optimiser degrades gracefully with no usable crisis data"
    except Exception as exc:  # pragma: no cover
        ok, detail = False, f"raised {exc!r}"
    check("Edge case: crisis window with no data", ok, detail)

    # A 0.01% TE budget is impossible once equity is capped at 50%.
    infeasible, _ = optimise(returns, te_cap=0.0001, step=0.10,
                             bounds=[(0.0, 0.5), (0, 1), (0, 1), (0, 1)])
    check("Edge case: impossible tracking-error budget",
          infeasible["te_budget_feasible"] is False,
          "flagged as infeasible instead of returning a silently wrong answer")

    # --- Headline result, for the record ------------------------------------
    best, _ = optimise(returns, te_cap=0.10, step=0.025)
    eq = portfolio_returns(returns, weights, "M")
    lines = [
        "# Verification output",
        "",
        f"- Data source: {source}",
        f"- Retrieval date: {retrieved}",
        f"- Sample: {report['first_date']} to {report['last_date']} "
        f"({report['rows_used']} daily observations)",
        f"- Crisis windows used: {', '.join(best['crises_used'])}",
        "",
        "## Checks",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    lines += [f"| {n} | {'PASS' if p else 'FAIL'} | {d} |" for n, p, d in results]
    lines += [
        "",
        "## Headline result reproduced by this script",
        "",
        f"- Shield weights: {best['weights']}",
        f"- Worst crisis drawdown: {best['worst_crisis_dd']:.2%}",
        f"- Tracking error vs SPY: {best['tracking_error']:.2%}",
        f"- Equal-weight max drawdown (full sample): {max_drawdown(eq):.2%}",
        f"- SPY max drawdown (full sample): {max_drawdown(returns['SPY']):.2%}",
        "",
        "## What would make the conclusion unreliable",
        "",
        "- Only six crisis episodes exist in the sample; the optimiser can fit them.",
        "- Pre-2019 managed futures is RYMFX, not DBMF (see data/README_DATA.md).",
        "- Bond/gold behaviour in 2022 differed from 2008; future crises may differ again.",
        "- Costs, taxes, bid-ask spreads and slippage are not modelled.",
    ]
    with open("VERIFICATION_OUTPUT.md", "w") as fh:
        fh.write("\n".join(lines) + "\n")

    failed = [n for n, p, _ in results if not p]
    print("\nAll checks passed." if not failed else f"\nFAILED: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
