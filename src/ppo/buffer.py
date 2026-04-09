"""Rollout Buffer

Fixed-capacity buffer for collecting T-step trajectories.

Critical design constraint:
  - Must NOT perform single-step online updates (causes NN divergence)
  - Epoch-based batch updates only when buffer is full
  - Data structure supports efficient mini-batch splitting
"""
from __future__ import annotations

import numpy as np
from typing import Optional


class RolloutBuffer:
    """
    Fixed-capacity rollout buffer for PPO.

    Stores per-step data:
      states, actions, rewards, values, log_probs, dones

    When full, the buffer is shuffled and yielded as mini-batches
    for multiple PPO epoch updates (K=4..10).

    All arrays are stored as np.ndarray for maximum compatibility
    with vectorized GAE computation in gae.py.
    """

    def __init__(self, buffer_size: int, state_dim: int = 10, action_dim: int = 2):
        """
        Parameters
        ----------
        buffer_size : int
            Maximum number of steps per rollout (T). Default 100.
        state_dim : int
            State dimensionality. Default 10.
        action_dim : int
            Action dimensionality. Default 2.
        """
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.bool_)

        self._ptr: int = 0      # next write position
        self._size: int = 0     # actual number of stored steps
        self.full: bool = False

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        """
        Append one step of data to the buffer.

        Parameters
        ----------
        state : np.ndarray, shape (state_dim,)
        action : np.ndarray, shape (action_dim,)
        reward : float
            Reward_t (not r_port)
        value : float
            V(s_t) from critic
        log_prob : float
            log π(a_t|s_t) under current policy
        done : bool
            Whether episode terminated at this step.
        """
        self.states[self._ptr] = state
        self.actions[self._ptr] = action
        self.rewards[self._ptr] = reward
        self.values[self._ptr] = value
        self.log_probs[self._ptr] = log_prob
        self.dones[self._ptr] = done

        self._ptr = (self._ptr + 1) % self.buffer_size
        if self._ptr == 0:
            self.full = True
        self._size = min(self._size + 1, self.buffer_size)

    def is_full(self) -> bool:
        """Return True if buffer has reached full capacity."""
        return self.full or self._size == self.buffer_size

    def get_all(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return all stored data (no padding).

        Returns
        -------
        Tuple[np.ndarray, ...]
            (states, actions, rewards, values, log_probs, dones)
            All arrays have shape (size,) or (size, dim).
        """
        if self.full:
            return (
                self.states,
                self.actions,
                self.rewards,
                self.values,
                self.log_probs,
                self.dones,
            )
        return (
            self.states[:self._size],
            self.actions[:self._size],
            self.rewards[:self._size],
            self.values[:self._size],
            self.log_probs[:self._size],
            self.dones[:self._size],
        )

    def shuffle_and_split(
        self,
        mini_batch_size: int,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Shuffle stored data and split into mini-batches.

        Must be called ONLY when buffer is full.

        Parameters
        ----------
        mini_batch_size : int
            Number of samples per mini-batch. Default 32.

        Returns
        -------
        List[Tuple[states, actions, advantages, value_targets, old_log_probs]]
            List of mini-batches (each a 5-tuple of arrays).
        """
        states, actions, rewards, values, log_probs, dones = self.get_all()
        T = len(states)

        # Shuffle indices
        indices = np.arange(T)
        np.random.shuffle(indices)

        mini_batches = []
        for start in range(0, T, mini_batch_size):
            end = min(start + mini_batch_size, T)
            batch_idx = indices[start:end]
            # Advantages and value_targets are computed externally and stored separately
            # Here we return raw arrays; caller must supply advantages and value_targets
            mini_batches.append((
                states[batch_idx],
                actions[batch_idx],
                rewards[batch_idx],
                values[batch_idx],
                log_probs[batch_idx],
            ))
        return mini_batches

    def compute_gae_and_split(
        self,
        bootstrap_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        mini_batch_size: int = 32,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Compute GAE and return shuffled mini-batches with advantages.

        Convenience method: GAE + shuffle + split in one call.

        Parameters
        ----------
        bootstrap_value : float
            V(s_{T}) for the final step (bootstrap from current critic).
        gamma : float
            TD discount. Default 0.99.
        gae_lambda : float
            GAE lambda. Default 0.95.
        mini_batch_size : int
            Mini-batch size. Default 32.

        Returns
        -------
        List[Tuple[states, actions, rewards, advantages, value_targets, old_log_probs]]
        """
        from src.ppo.gae import GAEBuffer

        states, actions, rewards, values, log_probs, dones = self.get_all()

        # Compute GAE
        gae = GAEBuffer(gamma=gamma, gae_lambda=gae_lambda)
        advantages, value_targets = gae.compute(rewards, values, bootstrap_value, dones)

        # Normalize advantages (optional but recommended)
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Shuffle
        T = len(states)
        indices = np.arange(T)
        np.random.shuffle(indices)

        mini_batches = []
        for start in range(0, T, mini_batch_size):
            end = min(start + mini_batch_size, T)
            batch_idx = indices[start:end]
            mini_batches.append((
                states[batch_idx],
                actions[batch_idx],
                rewards[batch_idx],
                advantages[batch_idx],
                value_targets[batch_idx],
                log_probs[batch_idx],
            ))
        return mini_batches

    def clear(self) -> None:
        """Reset buffer after update. Used after K epochs complete."""
        self._ptr = 0
        self._size = 0
        self.full = False

    def __len__(self) -> int:
        """Current number of stored steps."""
        return self._size
