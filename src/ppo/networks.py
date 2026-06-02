"""Actor-Critic Network Topology

Two independent MLP networks:
  - Actor:  10 → 64 → 64 → 2 (mean with Tanh, trainable log_std)
  - Critic: 10 → 64 → 64 → 1 (value, no activation)

All Linear layers use Orthogonal Initialization to prevent
early gradient vanishing/explosion — industry standard for PPO
in continuous action spaces.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np


def orthogonal_init(module: nn.Module, gain: float = np.sqrt(2)) -> None:
    """
    Apply orthogonal (weight) initialization to all Linear layers.

    Orthogonal init is the industry standard for PPO in continuous action
    spaces because it preserves the variance of activations through layers,
    preventing early gradient vanishing or explosion.

    Parameters
    ----------
    module : nn.Module
        The module whose Linear weights will be orthogonally initialized.
    gain : float
        Scaling factor. Default sqrt(2) for ReLU/Tanh cascades.
    """
    for name, param in module.named_parameters():
        if "weight" in name and param.ndim == 2:
            nn.init.orthogonal_(param, gain=gain)
        elif "bias" in name:
            nn.init.constant_(param, 0.0)


class ActorNetwork(nn.Module):
    """
    Policy Network (Actor).

    Architecture:
      Input(10) → Linear(10, 64) → Tanh → Linear(64, 64) → Tanh → Linear(64, 2) → Tanh → mu_t

    The log_std parameter is a standalone nn.Parameter (independent of state),
    so exploration variance decays globally with training, not oscillates
    with local state — critical for stable PPO convergence.
    """

    def __init__(self, state_dim: int = 10, action_dim: int = 2, hidden_dim: int = 64):
        """
        Parameters
        ----------
        state_dim : int
            Dimensionality of state S_t. Default 10.
        action_dim : int
            Dimensionality of action. Default 2 (Δα, Δτ).
        hidden_dim : int
            Hidden layer width. Default 64.
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        # Mean head: output in [-1, 1] via Tanh (but mapping done in ActionMapper)
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        # Standalone log_std: global trainable parameter, NOT state-dependent
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        orthogonal_init(self.net[0], gain=np.sqrt(2))
        orthogonal_init(self.net[2], gain=np.sqrt(2))
        orthogonal_init(self.mu_head, gain=0.01)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        state : torch.Tensor, shape (batch, 10) or (10,)
            State vector S_t.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            mu_t:  mean of action distribution, shape (batch, 2)
            std_t: standard deviation, shape (batch, 2) — expanded from global log_std
        """
        x = self.net(state)
        mu_t = torch.tanh(self.mu_head(x))  # mean in [-1, 1]
        std_t = torch.exp(self.log_std).unsqueeze(0).expand(mu_t.shape[0], -1)
        return mu_t, std_t

    def get_log_prob(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log probability of action under current policy.

        Parameters
        ----------
        state : torch.Tensor, shape (batch, 10)
        action : torch.Tensor, shape (batch, 2), expected in [-1, 1]

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            log_prob: log π(a|s), shape (batch,)
            entropy:  policy entropy, shape (batch,)
        """
        mu_t, std_t = self.forward(state)
        # Normal log_pdf for Tanh-transformed action
        log_std = torch.log(std_t + 1e-6)
        # Inverse Tanh to get pre-activation value (for Gaussian log_prob)
        # Using atanh safely: a_tanh ∈ (-1,1), atanh(a_tanh) = 0.5*ln((1+a)/(1-a))
        action_clipped = torch.clamp(action, -0.9999, 0.9999)
        a_pre = 0.5 * torch.log((1 + action_clipped + 1e-6) / (1 - action_clipped + 1e-6))
        log_prob = -0.5 * (((a_pre - mu_t) / (std_t + 1e-6)) ** 2 + 2 * log_std + np.log(2 * np.pi))
        log_prob = log_prob.sum(dim=-1)  # sum over action dims

        # Entropy of Gaussian with Tanh boundary: H = sum over dims of (log_std + 0.5*(1+log(2π)))
        entropy = (log_std + 0.5 * (1 + np.log(2 * np.pi))).sum(dim=-1)

        return log_prob, entropy


class CriticNetwork(nn.Module):
    """
    Value Network (Critic).

    Architecture:
      Input(10) → Linear(10, 64) → Tanh → Linear(64, 64) → Tanh → Linear(64, 1)

    Completely independent from Actor — no weight sharing.
    Directly estimates V(s_t) for GAE and advantage computation.
    """

    def __init__(self, state_dim: int = 10, hidden_dim: int = 64):
        """
        Parameters
        ----------
        state_dim : int
            Dimensionality of state S_t. Default 10.
        hidden_dim : int
            Hidden layer width. Default 64.
        """
        super().__init__()
        self.state_dim = state_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),  # scalar value, no activation
        )

        orthogonal_init(self.net[0], gain=np.sqrt(2))
        orthogonal_init(self.net[2], gain=np.sqrt(2))
        orthogonal_init(self.net[4], gain=1.0)  # linear output, std=1

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state : torch.Tensor, shape (batch, 10) or (10,)

        Returns
        -------
        torch.Tensor
            V(s_t), shape (batch,) or scalar
        """
        return self.net(state).squeeze(-1)


class ActorCritic(nn.Module):
    """
    Unified Actor-Critic container.

    Holds two completely independent networks:
      - self.actor: policy network (ActorNetwork)
      - self.critic: value network (CriticNetwork)

    Weight sharing is intentionally absent — prevents interference between
    the two learning objectives.
    """

    def __init__(
        self,
        state_dim: int = 10,
        action_dim: int = 2,
        hidden_dim: int = 64,
    ):
        """
        Parameters
        ----------
        state_dim : int
            State dimensionality. Default 10.
        action_dim : int
            Action dimensionality. Default 2.
        hidden_dim : int
            Hidden layer width. Default 64.
        """
        super().__init__()
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim)
        self.critic = CriticNetwork(state_dim, hidden_dim)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convenience: returns (mu_t, V_t)."""
        mu_t, std_t = self.actor(state)
        value_t = self.critic(state)
        return mu_t, value_t
