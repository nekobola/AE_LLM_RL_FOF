"""PPO Module

Proximal Policy Optimization meta-controller for AE-LLM-RL-FOF.
Actor-Critic architecture with GAE advantage estimation.
"""
from src.ppo.networks import ActorCritic, ActorNetwork, CriticNetwork
from src.ppo.gae import compute_gae, GAEBuffer
from src.ppo.loss import (
    ppo_clip_loss,
    entropy_loss,
    critic_loss,
    total_ppo_loss,
)
from src.ppo.buffer import RolloutBuffer
from src.ppo.trainer import PPOTrainer

__all__ = [
    "ActorCritic",
    "ActorNetwork",
    "CriticNetwork",
    "compute_gae",
    "GAEBuffer",
    "ppo_clip_loss",
    "entropy_loss",
    "critic_loss",
    "total_ppo_loss",
    "RolloutBuffer",
    "PPOTrainer",
]
