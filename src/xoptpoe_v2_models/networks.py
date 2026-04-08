"""Neural network architectures for XOPTPOE v2 PTO and E2E models."""

from __future__ import annotations

import torch
from torch import nn


class PredictorMLP(nn.Module):
    """Paper-style compact feedforward network: 32 -> 16 -> 8 hidden units."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dims: tuple[int, int, int] = (32, 16, 8),
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(inputs)
        output = self.head(hidden)
        return output.squeeze(-1)
