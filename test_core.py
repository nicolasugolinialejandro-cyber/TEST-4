"""Unit tests for the calculation engines (run with: pytest -q)."""

import numpy as np
import pandas as pd
import pytest

from src.fastcalc import max_drawdowns, wealth_paths
from src.metrics import (annualised_return, max_drawdown, portfolio_returns,
                         tracking_error)
from src.optimizer import equal_weight, simplex_grid


@pytest.fixture
def toy() -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=260)
    rng = np.random.default_rng(42)
    return pd.DataFrame(rng.normal(0.0003, 0.01, (260, 3)), index=idx,
                        columns=["A", "B", "C"])


def test_known_drawdown():
    r = pd.Series([0.10, -0.50, 0.20], index=pd.bdate_range("2020-01-01", periods=3))
    assert max_drawdown(r) == pytest.approx(-0.50)


def test_annualised_return_of_flat_series():
    r = pd.Series(np.zeros(252), index=pd.bdate_range("2020-01-01", periods=252))
    assert annualised_return(r) == pytest.approx(0.0)


def test_single_asset_portfolio_equals_asset(toy):
    port = portfolio_returns(toy, {"A": 1, "B": 0, "C": 0}, "M")
    assert np.allclose(port.to_numpy(), toy["A"].to_numpy())


def test_tracking_error_against_self_is_zero(toy):
    assert tracking_error(toy["A"], toy["A"]) == pytest.approx(0.0)


def test_fast_and_slow_engines_agree(toy):
    w = equal_weight(list(toy.columns))
    slow = max_drawdown(portfolio_returns(toy, w, "M"))
    fast = float(max_drawdowns(wealth_paths(toy, np.array([[1 / 3, 1 / 3, 1 / 3]]), "M"))[0])
    assert slow == pytest.approx(fast, abs=1e-12)


def test_simplex_grid_sums_to_one():
    grid = simplex_grid(4, 0.1)
    assert np.allclose(grid.sum(axis=1), 1.0)
    assert len(grid) == 286  # C(10+3,3)


def test_bounds_are_respected():
    grid = simplex_grid(4, 0.1, [(0.4, 0.7), (0, 1), (0, 1), (0, 1)])
    assert grid[:, 0].min() >= 0.4 - 1e-9 and grid[:, 0].max() <= 0.7 + 1e-9


def test_zero_weights_rejected(toy):
    with pytest.raises(ValueError):
        portfolio_returns(toy, {"A": 0, "B": 0, "C": 0})
