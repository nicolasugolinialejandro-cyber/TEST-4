"""Vectorised evaluation of many candidate portfolios at once.

metrics.py is the readable, one-portfolio-at-a-time reference implementation.
This module does exactly the same arithmetic with NumPy matrices so the grid
search can price thousands of allocations in a second. verify.py checks the two
implementations agree to 1e-10, which is one of the project's independent
verification tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _period_labels(index: pd.DatetimeIndex, rebalance: str) -> np.ndarray:
    """Integer id of the rebalancing period each date belongs to."""
    if rebalance == "BH":
        return np.zeros(len(index), dtype=int)
    freq = {"M": "M", "Q": "Q", "Y": "Y"}[rebalance]
    codes = pd.PeriodIndex(index, freq=freq)
    _, labels = np.unique(codes.asi8, return_inverse=True)
    return labels


def wealth_paths(returns: pd.DataFrame, weight_matrix: np.ndarray,
                 rebalance: str = "M") -> np.ndarray:
    """Wealth index (T x K) for K candidate portfolios, starting at 1.0.

    weight_matrix: K x N array whose rows sum to 1, columns aligned to `returns`.
    """
    r = returns.to_numpy(dtype=float)
    if rebalance == "D":
        port = r @ weight_matrix.T
        return np.cumprod(1.0 + port, axis=0)

    labels = _period_labels(returns.index, rebalance)
    growth = np.empty_like(r)
    for lab in np.unique(labels):  # within-period compounding, weights drift
        mask = labels == lab
        growth[mask] = np.cumprod(1.0 + r[mask], axis=0)

    levels = growth @ weight_matrix.T  # T x K, resets to ~1 each period
    # Chain the periods together: multiply by the product of previous period ends.
    period_end_rows = np.flatnonzero(np.diff(labels, append=labels[-1] + 1) != 0)
    carry = np.ones(weight_matrix.shape[0])
    out = np.empty_like(levels)
    start = 0
    for end in period_end_rows:
        out[start:end + 1] = levels[start:end + 1] * carry
        carry = out[end]
        start = end + 1
    return out


def path_returns(wealth: np.ndarray) -> np.ndarray:
    """Daily returns implied by a wealth matrix (first row uses a base of 1.0)."""
    prev = np.vstack([np.ones((1, wealth.shape[1])), wealth[:-1]])
    return wealth / prev - 1.0


def max_drawdowns(wealth: np.ndarray) -> np.ndarray:
    """Max drawdown of each column (negative numbers)."""
    peaks = np.maximum.accumulate(wealth, axis=0)
    return (wealth / peaks - 1.0).min(axis=0)


def tracking_errors(wealth: np.ndarray, benchmark_returns: np.ndarray,
                    periods_per_year: int = 252) -> np.ndarray:
    """Annualised tracking error of each column against the benchmark."""
    diff = path_returns(wealth) - benchmark_returns[:, None]
    return diff.std(axis=0, ddof=1) * np.sqrt(periods_per_year)
