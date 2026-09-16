"""Black–Litterman portfolio research engine.

Loads closing prices from CSV or creates reproducible demo data, estimates
return/risk inputs, builds equilibrium priors, applies confidence-weighted
views, and compares constrained Markowitz and Black–Litterman portfolios.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

TRADING_DAYS = 252


@dataclass(frozen=True)
class Constraints:
    min_weight: float = 0.0
    max_weight: float = 0.40
    allow_short: bool = False

    def bounds(self, n: int) -> list[tuple[float, float]]:
        lower = -self.max_weight if self.allow_short else self.min_weight
        return [(lower, self.max_weight)] * n

    def validate(self, n: int) -> None:
        if self.max_weight <= 0 or self.min_weight > self.max_weight:
            raise ValueError("Invalid weight bounds.")
        if not self.allow_short and self.min_weight * n > 1.0 + 1e-10:
            raise ValueError("Minimum weights cannot sum to more than 100%.")
        if self.max_weight * n < 1.0 - 1e-10:
            raise ValueError("Maximum weights cannot reach a 100% allocation.")


@dataclass
class Portfolio:
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe: float
    label: str


def load_prices(path: str | Path) -> pd.DataFrame:
    """Read a wide CSV whose first column is the date and remaining columns are prices."""
    df = pd.read_csv(path)
    if df.shape[1] < 3:
        raise ValueError("Price CSV requires a date column and at least two securities.")
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    df = df.dropna(subset=[df.columns[0]]).set_index(df.columns[0]).sort_index()
    prices = df.apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if len(prices) < 30:
        raise ValueError("At least 30 aligned observations are required.")
    return prices


def demo_prices(symbols: list[str], sessions: int = 500, seed: int = 11) -> pd.DataFrame:
    """Create correlated synthetic prices for a reproducible offline example."""
    rng = np.random.default_rng(seed)
    n = len(symbols)
    market = rng.normal(0.00025, 0.009, sessions)
    loadings = np.linspace(0.75, 1.20, n)
    idio = rng.normal(0.0, 0.010, size=(sessions, n))
    returns = market[:, None] * loadings + idio
    prices = 100.0 * np.exp(np.cumsum(returns, axis=0))
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=sessions)
    return pd.DataFrame(prices, index=dates, columns=symbols)


def estimate_inputs(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    returns = prices.pct_change().dropna(how="any")
    mean = returns.mean() * TRADING_DAYS
    covariance = returns.cov() * TRADING_DAYS
    return returns, mean, covariance


def market_weights(market_caps: pd.Series) -> pd.Series:
    caps = market_caps.astype(float)
    if (caps <= 0).any() or caps.sum() <= 0:
        raise ValueError("All market capitalisations must be positive.")
    return caps / caps.sum()


def implied_excess_returns(
    covariance: pd.DataFrame,
    weights: pd.Series,
    assumed_market_return: float,
    risk_free_rate: float,
) -> tuple[pd.Series, float]:
    w = weights.reindex(covariance.index).to_numpy()
    market_variance = float(w @ covariance.to_numpy() @ w)
    if market_variance <= 0:
        raise ValueError("Market variance must be positive.")
    risk_aversion = (assumed_market_return - risk_free_rate) / market_variance
    prior = risk_aversion * covariance.to_numpy() @ w
    return pd.Series(prior, index=covariance.index, name="prior_excess"), risk_aversion


def build_views(
    symbols: list[str], views: list[dict], risk_free_rate: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = {symbol: i for i, symbol in enumerate(symbols)}
    p_rows, q_values, confidence = [], [], []
    for view in views:
        row = np.zeros(len(symbols))
        if view["type"] == "absolute":
            row[index[view["asset"]]] = 1.0
            q_value = float(view["value"]) - risk_free_rate
        elif view["type"] == "relative":
            row[index[view["asset_a"]]] = 1.0
            row[index[view["asset_b"]]] = -1.0
            q_value = float(view["value"])
        else:
            raise ValueError(f"Unsupported view type: {view['type']}")
        p_rows.append(row)
        q_values.append(q_value)
        confidence.append(float(view["confidence"]))
    if not p_rows:
        return np.zeros((0, len(symbols))), np.zeros(0), np.zeros(0)
    return np.asarray(p_rows), np.asarray(q_values), np.asarray(confidence)


def view_uncertainty(
    p: np.ndarray, covariance: pd.DataFrame, tau: float, confidence: np.ndarray
) -> np.ndarray:
    if len(p) == 0:
        return np.zeros((0, 0))
    base = np.diag(p @ (tau * covariance.to_numpy()) @ p.T)
    c = np.clip(confidence, 0.01, 0.99)
    return np.diag(np.maximum(base * (1.0 - c) / c, 1e-12))


def black_litterman(
    prior_excess: pd.Series,
    covariance: pd.DataFrame,
    p: np.ndarray,
    q: np.ndarray,
    omega: np.ndarray,
    tau: float,
) -> tuple[pd.Series, pd.DataFrame]:
    sigma = covariance.to_numpy()
    tau_sigma = tau * sigma
    prior = prior_excess.reindex(covariance.index).to_numpy()
    if len(p) == 0:
        posterior = prior.copy()
        estimation_covariance = tau_sigma
    else:
        middle = np.linalg.pinv(p @ tau_sigma @ p.T + omega)
        posterior = prior + tau_sigma @ p.T @ middle @ (q - p @ prior)
        estimation_covariance = tau_sigma - tau_sigma @ p.T @ middle @ p @ tau_sigma
    posterior_covariance = sigma + estimation_covariance
    return (
        pd.Series(posterior, index=covariance.index, name="posterior_excess"),
        pd.DataFrame(posterior_covariance, index=covariance.index, columns=covariance.columns),
    )


def portfolio_metrics(
    weights: np.ndarray, expected_returns: np.ndarray, covariance: np.ndarray, risk_free: float
) -> tuple[float, float, float]:
    expected = float(weights @ expected_returns)
    variance = float(max(weights @ covariance @ weights, 0.0))
    volatility = float(np.sqrt(variance))
    sharpe = (expected - risk_free) / volatility if volatility > 1e-12 else 0.0
    return expected, volatility, sharpe


def optimise(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free: float,
    constraints: Constraints,
    objective: str = "max_sharpe",
    target_return: float | None = None,
) -> Portfolio:
    n = len(expected_returns)
    constraints.validate(n)
    mu, sigma = expected_returns.to_numpy(), covariance.to_numpy()
    bounds = constraints.bounds(n)
    equations = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if objective == "target":
        if target_return is None:
            raise ValueError("Target-return optimisation requires target_return.")
        equations.append({"type": "eq", "fun": lambda w: w @ mu - target_return})

    def score(w: np.ndarray) -> float:
        expected, volatility, sharpe = portfolio_metrics(w, mu, sigma, risk_free)
        if objective == "max_sharpe":
            return -sharpe
        if objective in {"min_variance", "target"}:
            return volatility**2
        raise ValueError(f"Unsupported objective: {objective}")

    rng = np.random.default_rng(123)
    starts = [np.repeat(1.0 / n, n)]
    starts.extend(rng.dirichlet(np.ones(n), size=30))
    best = None
    for start in starts:
        if any(start[i] < bounds[i][0] or start[i] > bounds[i][1] for i in range(n)):
            continue
        result = minimize(
            score, start, method="SLSQP", bounds=bounds, constraints=equations,
            options={"maxiter": 1500, "ftol": 1e-12},
        )
        if result.success and (best is None or result.fun < best.fun):
            best = result
    if best is None:
        raise ValueError("No feasible optimum was found under the selected constraints.")
    expected, volatility, sharpe = portfolio_metrics(best.x, mu, sigma, risk_free)
    labels = {"max_sharpe": "Maximum Sharpe", "min_variance": "Minimum Variance", "target": "Target Return"}
    return Portfolio(pd.Series(best.x, index=expected_returns.index), expected, volatility, sharpe, labels[objective])


def efficient_frontier(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free: float,
    constraints: Constraints,
    points: int = 30,
) -> pd.DataFrame:
    minimum = optimise(expected_returns, covariance, risk_free, constraints, "min_variance")
    bounds = constraints.bounds(len(expected_returns))
    maximum = linprog(-expected_returns.to_numpy(), A_eq=np.ones((1, len(expected_returns))), b_eq=[1.0], bounds=bounds, method="highs")
    if not maximum.success:
        raise ValueError("Unable to determine the feasible maximum return.")
    rows = []
    for target in np.linspace(minimum.expected_return, -maximum.fun, points):
        try:
            result = optimise(expected_returns, covariance, risk_free, constraints, "target", float(target))
            rows.append({"return": result.expected_return, "volatility": result.volatility, "sharpe": result.sharpe})
        except ValueError:
            continue
    return pd.DataFrame(rows)


def concentration(weights: pd.Series) -> float:
    return float(np.square(weights.to_numpy()).sum())


def turnover(old: pd.Series, new: pd.Series) -> float:
    aligned = old.reindex(new.index).fillna(0.0)
    return float(0.5 * np.abs(new.to_numpy() - aligned.to_numpy()).sum())


def run_example(prices: pd.DataFrame) -> tuple[pd.DataFrame, Portfolio, Portfolio]:
    _, historical_returns, covariance = estimate_inputs(prices)
    caps = pd.Series({"RELIANCE": 2_000_000, "TCS": 1_450_000, "HDFCBANK": 1_250_000, "INFY": 650_000, "SBIN": 720_000})
    caps = caps.reindex(prices.columns)
    risk_free, market_return, tau = 0.065, 0.12, 0.05
    prior_excess, _ = implied_excess_returns(covariance, market_weights(caps), market_return, risk_free)
    views = [
        {"type": "absolute", "asset": "SBIN", "value": 0.14, "confidence": 0.65},
        {"type": "absolute", "asset": "RELIANCE", "value": 0.13, "confidence": 0.55},
        {"type": "relative", "asset_a": "HDFCBANK", "asset_b": "TCS", "value": 0.04, "confidence": 0.60},
    ]
    p, q, confidence = build_views(list(prices.columns), views, risk_free)
    posterior_excess, posterior_covariance = black_litterman(
        prior_excess, covariance, p, q, view_uncertainty(p, covariance, tau, confidence), tau
    )
    posterior_total = posterior_excess + risk_free
    limits = Constraints(max_weight=0.40)
    historical = optimise(historical_returns, covariance, risk_free, limits)
    posterior = optimise(posterior_total, posterior_covariance, risk_free, limits)
    comparison = pd.DataFrame(
        {
            "Historical": [historical.expected_return, historical.volatility, historical.sharpe, concentration(historical.weights)],
            "Black-Litterman": [posterior.expected_return, posterior.volatility, posterior.sharpe, concentration(posterior.weights)],
        },
        index=["Expected return", "Volatility", "Sharpe", "Concentration"],
    )
    comparison.loc["Turnover", "Black-Litterman"] = turnover(historical.weights, posterior.weights)
    return comparison, historical, posterior


def main() -> None:
    parser = argparse.ArgumentParser(description="Black–Litterman portfolio research engine")
    parser.add_argument("--prices", help="Wide price CSV: Date, RELIANCE, TCS, HDFCBANK, INFY, SBIN")
    args = parser.parse_args()
    symbols = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "SBIN"]
    prices = load_prices(args.prices) if args.prices else demo_prices(symbols)
    comparison, historical, posterior = run_example(prices[symbols])
    formatted = comparison.apply(lambda column: column.map(lambda x: f"{x:.4f}" if pd.notna(x) else "—"))
    print("\nMODEL COMPARISON\n", formatted)
    print("\nHISTORICAL WEIGHTS\n", historical.weights.map(lambda x: f"{x:.2%}"))
    print("\nBLACK–LITTERMAN WEIGHTS\n", posterior.weights.map(lambda x: f"{x:.2%}"))


if __name__ == "__main__":
    main()
