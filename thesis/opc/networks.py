"""Value and policy networks for OPC AlphaZero training.

Both networks share a common trunk (body) that processes the observation tensor.
The policy head outputs logits over the 125 joint actions.
The value head outputs a scalar in (-1, 1) representing expected outcome.

These are used identically by both players: given a player's observation tensor,
the network predicts (policy_logits, value) from that player's perspective.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from opc.game import _OBS_SIZE, _N_JOINT


class ResidualBlock(nn.Module):
    """Two-layer residual block with layer norm."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.norm1(self.fc1(x)))
        x = self.norm2(self.fc2(x))
        return F.relu(x + residual)


class OPCNetwork(nn.Module):
    """Shared trunk with policy and value heads.

    Input: observation tensor of shape (..., OBS_SIZE)
    Output: (policy_logits, value) where policy_logits has shape (..., N_JOINT)
            and value has shape (..., 1)
    """

    def __init__(
        self,
        obs_size: int = _OBS_SIZE,
        n_actions: int = _N_JOINT,
        hidden_size: int = 128,
        n_residual_blocks: int = 4,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        self.trunk = nn.Sequential(
            *[ResidualBlock(hidden_size) for _ in range(n_residual_blocks)]
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, n_actions),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Tanh(),
        )

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(obs)
        x = self.trunk(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value

    def predict(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (policy_probs, value) without gradient tracking."""
        with torch.no_grad():
            logits, value = self(obs)
            probs = F.softmax(logits, dim=-1)
        return probs, value


def build_network(hidden_size: int = 128, n_residual_blocks: int = 4) -> OPCNetwork:
    return OPCNetwork(
        obs_size=_OBS_SIZE,
        n_actions=_N_JOINT,
        hidden_size=hidden_size,
        n_residual_blocks=n_residual_blocks,
    )
