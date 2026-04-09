"""PPO Loss Functions

Implements the clipped surrogate objective for Actor
and the value function loss for Critic.

Loss_clip(θ) = -min( r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t )

Where:
  r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
  ε (epsilon) = clip range, default 0.2

Entropy bonus encourages early exploration.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np


def ppo_clip_loss(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    epsilon: float = 0.2,
) -> torch.Tensor:
    """
    PPO clipped surrogate loss.

    L_clip(θ) = -min( r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t )

    The min operator ensures the loss never exceeds the clipped version,
    preventing destructively large policy updates.

    Parameters
    ----------
    log_probs_new : torch.Tensor, shape (batch,)
        Log probability of actions under NEW policy π_θ.
    log_probs_old : torch.Tensor, shape (batch,)
        Log probability of actions under OLD policy π_θ_old.
    advantages : torch.Tensor, shape (batch,)
        GAE advantage estimates A_t.
    epsilon : float
        Clip range. Default 0.2.

    Returns
    -------
    torch.Tensor
        Scalar clipped surrogate loss.
    """
    # Probability ratio r_t(θ) = exp(log π_new - log π_old)
    ratio = torch.exp(log_probs_new - log_probs_old)

    # Unclipped objective
    surr_unclipped = ratio * advantages

    # Clipped objective
    ratio_clipped = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
    surr_clipped = ratio_clipped * advantages

    # PPO takes the MIN of clipped and unclipped (for policy gradient sign)
    # When advantage > 0 (good action): unclipped allows high prob → risk of overstep
    # When advantage < 0 (bad action): unclipped allows low prob → risk of overstep
    # Clipping caps both directions, protecting from catastrophic updates
    clipped_surrogate = torch.min(surr_unclipped, surr_clipped)

    return -clipped_surrogate.mean()


def entropy_loss(log_stds: torch.Tensor) -> torch.Tensor:
    """
    Policy entropy loss (for gradient ascent).

    H = sum_d (log std_d + 0.5 * (1 + log(2π)))
    Used as NEGATIVE entropy loss in total actor loss: L_actor = L_clip - c_ent * H

    Higher entropy → more exploration. We SUBTRACT entropy to penalize
    entropy collapse (premature convergence to deterministic policy).

    Parameters
    ----------
    log_stds : torch.Tensor, shape (action_dim,)
        Global log standard deviations.

    Returns
    -------
    torch.Tensor
        Scalar entropy loss (to be subtracted).
    """
    entropy = (log_stds + 0.5 * (1.0 + np.log(2 * np.pi))).sum()
    return -entropy  # negative because we want gradient ASCENT on entropy


def critic_loss(
    values_pred: torch.Tensor,
    value_targets: torch.Tensor,
    clip_fraction: float = 1.0,
) -> torch.Tensor:
    """
    Critic (Value) loss: MSE between predicted value and GAE target.

    L_critic = c_vf * 0.5 * (V_φ(s_t) - V_target_t)^2

    Parameters
    ----------
    values_pred : torch.Tensor, shape (batch,)
        Critic predictions V_φ(s_t).
    value_targets : torch.Tensor, shape (batch,)
        GAE advantage-augmented targets: A_t + V(s_t).
    clip_fraction : float
        Optional VF clipping weight. Default 1.0 (no clipping).
        In some PPO variants, the value function is also clipped.

    Returns
    -------
    torch.Tensor
        Scalar critic loss.
    """
    return 0.5 * F.mse_loss(values_pred, value_targets)


def total_ppo_loss(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    values_pred: torch.Tensor,
    value_targets: torch.Tensor,
    log_stds: torch.Tensor,
    epsilon: float = 0.2,
    c_entropy: float = 0.01,
    c_vf: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """
    Combined PPO total loss for single backward pass.

    L_total = L_clip - c_entropy * H + c_vf * L_critic

    Parameters
    ----------
    log_probs_new / log_probs_old : torch.Tensor
        New and old log probabilities.
    advantages / value_targets : torch.Tensor
        GAE advantages and value targets.
    log_stds : torch.Tensor
        Global log_std parameters (for entropy).
    epsilon : float
        PPO clip range.
    c_entropy : float
        Entropy coefficient. Default 0.01.
    c_vf : float
        Value function coefficient. Default 1.0.

    Returns
    -------
    Tuple[torch.Tensor, dict]
        (total_loss, loss_dict) where dict contains individual components.
    """
    loss_clip = ppo_clip_loss(log_probs_new, log_probs_old, advantages, epsilon)
    loss_entropy = entropy_loss(log_stds) * c_entropy
    loss_vf = critic_loss(values_pred, value_targets, c_vf)

    total = loss_clip + loss_entropy + c_vf * loss_vf

    loss_dict = {
        "loss_total": total.item(),
        "loss_clip": loss_clip.item(),
        "loss_entropy": loss_entropy.item(),
        "loss_vf": loss_vf.item(),
    }
    return total, loss_dict
