"""Generalized Advantage Estimation (GAE)

Computes advantage function A_t via temporal-difference error recursion.

Formula:
    A_t = delta_t + (gamma * lambda) * delta_{t+1} + ... + (gamma * lambda)^{T-t+1} * delta_{T-1}

    where delta_t = Reward_t + gamma * V(s_{t+1}) - V(s_t)

GAE provides a bias-variance tradeoff:
  - lambda = 1  → high variance, low bias (Monte-Carlo)
  - lambda = 0  → low variance, high bias (TD(0))
  - lambda ≈ 0.95 → balanced (industry default for PPO)
"""
from __future__ import annotations

import torch
import numpy as np
from typing import Optional


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GAE advantages and value targets.

    Parameters
    ----------
    rewards : torch.Tensor, shape (T,)
        Reward sequence {Reward_0, ..., Reward_{T-1}}.
        Named Reward_t to avoid conflict with R_t (return).
    values : torch.Tensor, shape (T,)
        Value estimates V(s_t) for t = 0..T-1.
    next_values : torch.Tensor, shape (T,)
        Value estimates V(s_{t+1}) for t = 0..T-1.
        For the final step (t=T-1), this should be V(s_T) = 0 if done,
        or bootstrap from the critic for non-terminal states.
    dones : torch.Tensor, shape (T,)
        Boolean tensor indicating episode termination (True = done).
    gamma : float
        TD discount factor. Default 0.99.
    gae_lambda : float
        GAE smoothing parameter. Default 0.95.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        advantages : GAE advantage A_t, shape (T,)
        value_targets : Bootstrap Target for Critic training, shape (T,)
            value_targets = advantages + values
    """
    T = rewards.shape[0]
    device = rewards.device

    # TD errors: delta_t = Reward_t + gamma * V(s_{t+1}) - V(s_t)
    # For done episodes, we do NOT add the bootstrap term
    deltas = rewards + gamma * next_values * (1.0 - dones.float()) - values

    # GAE accumulation (backward pass)
    advantages = torch.zeros_like(rewards)
    running_adv = torch.tensor(0.0, dtype=torch.float32, device=device)

    for t in reversed(range(T)):
        # If episode ended at t, no future bootstrap
        if t == T - 1:
            running_adv = deltas[t]
        else:
            running_adv = deltas[t] + gamma * gae_lambda * running_adv * (1.0 - dones[t].float())
        advantages[t] = running_adv

    value_targets = advantages + values
    return advantages, value_targets


class GAEBuffer:
    """
    In-process GAE calculator for rollout buffers.

    Stores trajectory data and computes GAE on demand.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cpu",
    ):
        """
        Parameters
        ----------
        gamma : float
            TD discount factor. Default 0.99.
        gae_lambda : float
            GAE smoothing parameter. Default 0.95.
        device : str
            torch device for computation.
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

    def compute(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        bootstrap_value: float,
        dones: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute GAE advantages from a completed trajectory.

        Parameters
        ----------
        rewards : np.ndarray, shape (T,)
            Reward sequence.
        values : np.ndarray, shape (T,)
            Value estimates for each state.
        bootstrap_value : float
            V(s_{T}) for the final step (used if not done).
        dones : np.ndarray, shape (T,)
            Episode termination flags.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (advantages, value_targets)
        """
        T = len(rewards)

        # Append bootstrap value at the end
        values_plus = np.append(values, bootstrap_value)

        # Compute deltas
        deltas = np.zeros(T)
        for t in range(T):
            next_val = values_plus[t + 1] if not dones[t] else 0.0
            deltas[t] = rewards[t] + self.gamma * next_val - values[t]

        # GAE accumulation (backward)
        advantages = np.zeros(T)
        running_adv = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                running_adv = deltas[t]
            else:
                running_adv = deltas[t] + self.gamma * self.gae_lambda * running_adv * (1.0 - dones[t])
            advantages[t] = running_adv

        value_targets = advantages + values
        return advantages, value_targets
