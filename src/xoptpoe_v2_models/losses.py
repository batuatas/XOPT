"""Prediction and decision-focused losses for XOPTPOE v2."""

from __future__ import annotations

import torch


EPS = 1e-8



def mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return torch.mean((y_pred - y_true) ** 2)



def portfolio_return(weights: torch.Tensor, realized_returns: torch.Tensor) -> torch.Tensor:
    return torch.dot(weights, realized_returns)



def portfolio_variance(weights: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return torch.matmul(weights, torch.matmul(sigma, weights))



def portfolio_volatility(weights: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.clamp(portfolio_variance(weights, sigma), min=EPS))



def portfolio_utility(weights: torch.Tensor, realized_returns: torch.Tensor, sigma: torch.Tensor, lambda_risk: float) -> torch.Tensor:
    return portfolio_return(weights, realized_returns) - 0.5 * float(lambda_risk) * portfolio_variance(weights, sigma)



def portfolio_sharpe(weights: torch.Tensor, realized_returns: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return portfolio_return(weights, realized_returns) / portfolio_volatility(weights, sigma)



def objective_score(
    objective_name: str,
    *,
    weights: torch.Tensor,
    realized_returns: torch.Tensor,
    sigma: torch.Tensor,
    lambda_risk: float,
) -> torch.Tensor:
    if objective_name == "return":
        return portfolio_return(weights, realized_returns)
    if objective_name == "utility":
        return portfolio_utility(weights, realized_returns, sigma, lambda_risk)
    if objective_name == "sharpe":
        return portfolio_sharpe(weights, realized_returns, sigma)
    raise ValueError(f"Unsupported objective_name: {objective_name}")
