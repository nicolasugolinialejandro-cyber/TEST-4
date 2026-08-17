"""Return, risk, drawdown and tracking-error measurement.

Every function here takes plain pandas objects so each one can be re-checked
by hand (see verify.py and tests/).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TRADING_DAYS


def portfolio_returns(returns: pd.DataFrame, weights: dict | pd.Series,
                      rebalance: str = "M") -> pd.Series:
    """Daily returns of a periodically rebalanced portfolio.

    rebalance: "D" (rebalance every day), "M" (month-end), "Q", "Y",
    or "BH" (buy and hold, never rebalanced).

    Between rebalance dates the weights drift with the assets, which is what a
    real investor experiences; assuming daily rebalancing quietly overstates
    the diversification benefit.
    """
    target = pd.Series(weights, dtype=float).reindex(returns.columns).fillna(0.0)
    if target.sum() <= 0:
        raise ValueError("Weights must sum to a positive number.")
    target = target / target.sum()

    if rebalance == "D":
        series = returns.mul(target, axis=1).sum(axis=1)
        series.name = "portfolio"
        return series

    if rebalance == "BH":
        group_keys = pd.Series(0, index=returns.index)
    else:
        freq = {"M": "ME", "Q": "QE", "Y": "YE"}[rebalance]
        group_keys = returns.index.to_period(
            {"M": "M", "Q": "Q", "Y": "Y"}[rebalance]
        )

    out = []
    for _, block in returns.groupby(group_keys):
        # Inside a period the portfolio simply compounds from fixed weights.
        level = (1.0 + block).cumprod().mul(target, axis=1).sum(axis=1)
        out.append(level / level.shift(1).fillna(1.0) - 1.0)

    series = pd.concat(out).sort_index()
    series.name = "portfolio"
    return series


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown path: current level divided by running peak, minus one."""
    level = (1.0 + returns).cumprod()
    return level / level.cummax() - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough loss, as a negative number."""
    if returns.empty:
        return 0.0
    return float(drawdown_series(returns).min())


def annualised_return(returns: pd.Series) -> float:
    """Geometric (CAGR-style) annualised return."""
    if returns.empty:
        return 0.0
    total = float((1.0 + returns).prod())
    years = len(returns) / TRADING_DAYS
    return total ** (1.0 / years) - 1.0 if years > 0 and total > 0 else -1.0


def annualised_vol(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, rf_annual: float = 0.0) -> float:
    """Excess return per unit of volatility, using a constant cash rate."""
    vol = annualised_vol(returns)
    if vol == 0:
        return 0.0
    return (annualised_return(returns) - rf_annual) / vol


def sortino_ratio(returns: pd.Series, rf_annual: float = 0.0) -> float:
    downside = returns[returns < 0]
    dd_vol = float(downside.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(downside) > 1 else 0.0
    if dd_vol == 0:
        return 0.0
    return (annualised_return(returns) - rf_annual) / dd_vol


def tracking_error(returns: pd.Series, benchmark: pd.Series) -> float:
    """Annualised standard deviation of the return difference vs the benchmark."""
    diff = (returns - benchmark).dropna()
    return float(diff.std(ddof=1) * np.sqrt(TRADING_DAYS))


def historical_var(returns: pd.Series, level: float = 0.95) -> float:
    """Daily historical VaR (negative number, e.g. -0.012 = -1.2%)."""
    return float(np.percentile(returns, (1 - level) * 100))


def historical_cvar(returns: pd.Series, level: float = 0.95) -> float:
    """Average loss on days worse than the VaR threshold (expected shortfall)."""
    var = historical_var(returns, level)
    tail = returns[returns <= var]
    return float(tail.mean()) if len(tail) else var


def ulcer_index(returns: pd.Series) -> float:
    """Root-mean-square drawdown: penalises deep *and* long underwater spells."""
    dd = drawdown_series(returns)
    return float(np.sqrt((dd ** 2).mean()))


def summary(returns: pd.Series, benchmark: pd.Series | None = None,
            rf_annual: float = 0.0) -> dict:
    """One dictionary with every headline statistic used in the app."""
    stats = {
        "CAGR": annualised_return(returns),
        "Volatility": annualised_vol(returns),
        "Max drawdown": max_drawdown(returns),
        "Sharpe": sharpe_ratio(returns, rf_annual),
        "Sortino": sortino_ratio(returns, rf_annual),
        "VaR 95% (daily)": historical_var(returns),
        "CVaR 95% (daily)": historical_cvar(returns),
        "Ulcer index": ulcer_index(returns),
    }
    if benchmark is not None:
        stats["Tracking error vs SPY"] = tracking_error(returns, benchmark)
        stats["Beta vs SPY"] = float(
            np.cov(returns, benchmark)[0, 1] / np.var(benchmark, ddof=1)
        )
    return stats
