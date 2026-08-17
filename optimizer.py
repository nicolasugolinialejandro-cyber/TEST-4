"""Portfolio construction: minimise crisis drawdown subject to a tracking-error budget.

Why a grid search and not a solver?
    Maximum drawdown is a path-dependent, non-convex, non-differentiable
    function of the weights, so gradient methods are unreliable and their
    output is hard to defend in a viva. With four assets we can enumerate the
    whole simplex on a 2.5% grid (12,341 candidate portfolios) in under a
    second, which gives a globally optimal answer *on that grid* and a full
    trade-off surface for free.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from .config import BENCHMARK, STRESS_WINDOWS, TRADING_DAYS
from .fastcalc import max_drawdowns, tracking_errors, wealth_paths
from .metrics import max_drawdown, portfolio_returns, tracking_error


def simplex_grid(n_assets: int, step: float = 0.025,
                 bounds: list[tuple[float, float]] | None = None) -> np.ndarray:
    """All weight vectors on a `step` grid that sum to 1 and respect bounds."""
    ticks = int(round(1.0 / step))
    bounds = bounds or [(0.0, 1.0)] * n_assets
    rows = []
    for combo in product(range(ticks + 1), repeat=n_assets - 1):
        if sum(combo) > ticks:
            continue
        full = np.array(list(combo) + [ticks - sum(combo)], dtype=float) / ticks
        if all(lo - 1e-9 <= w <= hi + 1e-9 for w, (lo, hi) in zip(full, bounds)):
            rows.append(full)
    return np.array(rows)


def slice_windows(returns: pd.DataFrame, windows: dict) -> dict:
    """Keep only the stress windows actually covered by the data sample."""
    out = {}
    for name, (start, end) in windows.items():
        block = returns.loc[str(start):str(end)]
        if len(block) > 20:  # ignore windows with too little data to be meaningful
            out[name] = block
    return out


def evaluate_weights(weights: np.ndarray, columns: list[str],
                     full_returns: pd.DataFrame, stress_blocks: dict,
                     rebalance: str = "M") -> dict:
    """Crisis drawdowns + full-sample tracking error for one weight vector."""
    w = dict(zip(columns, weights))
    per_crisis = {
        name: max_drawdown(portfolio_returns(block, w, rebalance))
        for name, block in stress_blocks.items()
    }
    full = portfolio_returns(full_returns, w, rebalance)
    return {
        "weights": w,
        "per_crisis": per_crisis,
        "worst_crisis_dd": min(per_crisis.values()) if per_crisis else 0.0,
        "avg_crisis_dd": float(np.mean(list(per_crisis.values()))) if per_crisis else 0.0,
        "tracking_error": tracking_error(full, full_returns[BENCHMARK]),
        "full_max_dd": max_drawdown(full),
    }


def optimise(returns: pd.DataFrame,
             te_cap: float = 0.10,
             step: float = 0.025,
             bounds: list[tuple[float, float]] | None = None,
             windows: dict | None = None,
             objective: str = "worst",
             rebalance: str = "M") -> tuple[dict, pd.DataFrame]:
    """Search the weight grid for the smallest crisis drawdown within the TE budget.

    objective: "worst" minimises the deepest drawdown across the selected
    crises (a true worst-case shield); "average" minimises the mean crisis
    drawdown (less conservative, less driven by a single episode).

    Returns the best candidate and the full evaluated grid, so the app can plot
    the drawdown / tracking-error trade-off instead of just asserting an answer.
    """
    columns = list(returns.columns)
    stress_blocks = slice_windows(returns, windows or STRESS_WINDOWS)
    grid = simplex_grid(len(columns), step, bounds)
    if len(grid) == 0:
        raise ValueError("No allocation satisfies the weight limits you set.")

    table = pd.DataFrame(grid, columns=columns)

    # Crisis drawdowns: one vectorised pass per stress window.
    crisis_cols = []
    for name, block in stress_blocks.items():
        col = f"dd::{name}"
        table[col] = max_drawdowns(wealth_paths(block, grid, rebalance))
        crisis_cols.append(col)

    table["worst_crisis_dd"] = table[crisis_cols].min(axis=1) if crisis_cols else 0.0
    table["avg_crisis_dd"] = table[crisis_cols].mean(axis=1) if crisis_cols else 0.0

    full_wealth = wealth_paths(returns, grid, rebalance)
    table["full_max_dd"] = max_drawdowns(full_wealth)
    table["tracking_error"] = tracking_errors(
        full_wealth, returns[BENCHMARK].to_numpy(dtype=float), TRADING_DAYS
    )
    table["cagr"] = full_wealth[-1] ** (TRADING_DAYS / len(returns)) - 1.0

    key = "worst_crisis_dd" if objective == "worst" else "avg_crisis_dd"
    feasible = table[table["tracking_error"] <= te_cap]
    if feasible.empty:
        # Never fail silently: fall back to the lowest-TE allocation available
        # and let the caller tell the user the budget was infeasible.
        best_row = table.loc[table["tracking_error"].idxmin()]
        feasible_flag = False
    else:
        # Drawdowns are negative, so the *largest* value is the shallowest loss.
        best_row = feasible.loc[feasible[key].idxmax()]
        feasible_flag = True

    best = {
        "weights": {c: float(best_row[c]) for c in columns},
        "per_crisis": {c.replace("dd::", ""): float(best_row[c]) for c in crisis_cols},
        "worst_crisis_dd": float(best_row["worst_crisis_dd"]),
        "avg_crisis_dd": float(best_row["avg_crisis_dd"]),
        "tracking_error": float(best_row["tracking_error"]),
        "full_max_dd": float(best_row["full_max_dd"]),
        "cagr": float(best_row["cagr"]),
        "te_budget_feasible": feasible_flag,
        "candidates_evaluated": int(len(table)),
        "crises_used": list(stress_blocks.keys()),
    }
    return best, table


def equal_weight(columns: list[str]) -> dict:
    """The mandated benchmark portfolio: 1/N across the universe."""
    return {c: 1.0 / len(columns) for c in columns}
