"""PPO Trainer

Epoch-based batch training loop for Actor-Critic PPO.

Workflow:
  1. Rollout:  Collect T steps into RolloutBuffer (NO single-step updates)
  2. Compute:   GAE advantages from stored rewards + bootstrap value
  3. Update:    K epochs of mini-batch SGD on Actor + Critic jointly
  4. Clear:     Reset buffer, collect next rollout with updated policy
"""
from __future__ import annotations

import torch
import torch.optim as optim
import numpy as np
from typing import Optional, Callable

from src.ppo.networks import ActorCritic
from src.ppo.buffer import RolloutBuffer
from src.ppo.loss import total_ppo_loss


class PPOTrainer:
    """
    PPO Trainer with epoch-based batch updates.

    Implements the standard PPO paradigm:
      - Rollout → Buffer → GAE → Mini-batch SGD (K epochs) → Repeat
    """

    def __init__(
        self,
        actor_critic: ActorCritic,
        config: dict,
        device: str = "cpu",
    ):
        """
        Parameters
        ----------
        actor_critic : ActorCritic
            The Actor-Critic network to train.
        config : dict
            Must contain:
              - ppo.clip_epsilon (default 0.2)
              - ppo.c_entropy (default 0.01)
              - ppo.c_vf (default 1.0)
              - ppo.lr (default 3e-4)
              - ppo.buffer_size (default 100)
              - ppo.mini_batch_size (default 32)
              - ppo.k_epochs (default 4)
              - ppo.gamma (default 0.99)
              - ppo.gae_lambda (default 0.95)
        device : str
            torch device.
        """
        self.ac = actor_critic
        self.device = device
        self.ac.to(device)

        ppo_cfg = config.get("ppo", {})
        self.clip_epsilon = ppo_cfg.get("clip_epsilon", 0.2)
        self.c_entropy = ppo_cfg.get("c_entropy", 0.01)
        self.c_vf = ppo_cfg.get("c_vf", 1.0)
        self.lr = ppo_cfg.get("lr", 3e-4)
        self.buffer_size = ppo_cfg.get("buffer_size", 100)
        self.mini_batch_size = ppo_cfg.get("mini_batch_size", 32)
        self.k_epochs = ppo_cfg.get("k_epochs", 4)
        self.gamma = ppo_cfg.get("gamma", 0.99)
        self.gae_lambda = ppo_cfg.get("gae_lambda", 0.95)
        self.max_grad_norm = ppo_cfg.get("max_grad_norm", 0.5)

        state_dim = config.get("env", {}).get("state_dim", 10)
        action_dim = config.get("env", {}).get("action_dim", 2)

        self.buffer = RolloutBuffer(
            buffer_size=self.buffer_size,
            state_dim=state_dim,
            action_dim=action_dim,
        )

        # Shared optimizer for Actor + Critic
        self.optimizer = optim.Adam(self.ac.parameters(), lr=self.lr, eps=1e-5)

        self._step_count: int = 0

    @torch.no_grad()
    def collect_rollout(
        self,
        env,
        max_steps: Optional[int] = None,
    ) -> None:
        """
        Collect one full rollout into the buffer.

        Uses current Actor-Critic policy to act in env.
        Stores (s, a, Reward_t, V(s), log π(a|s), done) per step.

        Parameters
        ----------
        env : gymnasium.Env
            MDPEnvironment from module 5.
        max_steps : int, optional
            Override buffer size for this rollout.
        """
        if max_steps is None:
            max_steps = self.buffer_size

        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)

        for _ in range(max_steps):
            state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)

            # Actor forward pass
            mu_t, _ = self.ac.actor(state_t)
            std_t = torch.exp(self.ac.actor.log_std).unsqueeze(0)

            # Sample action from Gaussian policy
            dist = torch.distributions.Normal(mu_t, std_t)
            action_np = mu_t.cpu().numpy().squeeze()  # deterministic for collection
            log_prob = dist.log_prob(torch.from_numpy(action_np).unsqueeze(0)).sum(dim=-1).item()

            # Critic forward pass
            value_t = self.ac.critic(state_t).item()

            # Step env
            next_state, reward_t, terminated, truncated, info = env.step(
                torch.from_numpy(action_np).float().unsqueeze(0).to(self.device)
            )
            next_state = np.array(next_state, dtype=np.float32)

            # Add to buffer
            self.buffer.add(
                state=state,
                action=action_np.astype(np.float32),
                reward=float(reward_t),
                value=float(value_t),
                log_prob=float(log_prob),
                done=bool(terminated or truncated),
            )

            state = next_state
            self._step_count += 1

            if terminated or truncated:
                state, _ = env.reset()
                state = np.array(state, dtype=np.float32)

    def update(self) -> dict:
        """
        Perform K epochs of PPO update on the full buffer.

        Call ONLY after buffer is full.

        Returns
        -------
        dict
            Aggregated loss statistics over all epochs and mini-batches.
        """
        if not self.buffer.is_full():
            raise RuntimeError(
                "PPOTrainer.update() called but buffer is not full. "
                "Call collect_rollout() until buffer.is_full() == True."
            )

        # Bootstrap value: V(s_{T}) from current critic
        with torch.no_grad():
            final_state, _ = self.buffer.get_all()[0], None
            # Use last stored state (most recent)
            states_all, _, _, values_all, _, dones_all = self.buffer.get_all()
            final_state_tensor = torch.from_numpy(states_all[-1]).unsqueeze(0).to(self.device)
            bootstrap_value = float(self.ac.critic(final_state_tensor).item())

        # Compute GAE and get mini-batches
        mini_batches = self.buffer.compute_gae_and_split(
            bootstrap_value=bootstrap_value,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            mini_batch_size=self.mini_batch_size,
        )

        # Aggregate loss stats
        epoch_stats = {
            "loss_total": [],
            "loss_clip": [],
            "loss_entropy": [],
            "loss_vf": [],
        }

        for _ in range(self.k_epochs):
            for batch in mini_batches:
                states_b, actions_b, rewards_b, advantages_b, value_targets_b, log_probs_old_b = batch

                # To tensors
                states_t = torch.from_numpy(states_b).to(self.device)
                actions_t = torch.from_numpy(actions_b).to(self.device)
                advantages_t = torch.from_numpy(advantages_b).to(self.device)
                value_targets_t = torch.from_numpy(value_targets_b).to(self.device)
                log_probs_old_t = torch.from_numpy(log_probs_old_b).to(self.device)

                # Forward pass
                values_pred_t = self.ac.critic(states_t)
                log_probs_new_t, _ = self.ac.actor.get_log_prob(states_t, actions_t)

                # Compute total loss
                loss, loss_dict = total_ppo_loss(
                    log_probs_new=log_probs_new_t,
                    log_probs_old=log_probs_old_t,
                    advantages=advantages_t,
                    values_pred=values_pred_t,
                    value_targets=value_targets_t,
                    log_stds=self.ac.actor.log_std,
                    epsilon=self.clip_epsilon,
                    c_entropy=self.c_entropy,
                    c_vf=self.c_vf,
                )

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Record
                for k in epoch_stats:
                    epoch_stats[k].append(loss_dict[k])

        # Clear buffer after K epochs
        self.buffer.clear()

        # Summary
        summary = {k: float(np.mean(v)) for k, v in epoch_stats.items()}
        summary["n_updates"] = self._step_count
        return summary

    def state_dict(self) -> dict:
        """Return optimizer + actor-critic state for checkpointing."""
        return {
            "ac": self.ac.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step_count": self._step_count,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore from checkpoint."""
        self.ac.load_state_dict(state["ac"])
        self.optimizer.load_state_dict(state["optimizer"])
        self._step_count = state["step_count"]
