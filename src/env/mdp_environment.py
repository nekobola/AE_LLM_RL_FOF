"""MDP Environment

 Gymnasium-compatible environment wrapping the full AE-LLM-RL-FOF system.

State space S_t ∈ R^10 (assembled by StateAssembler).
Action space: continuous a_{1,t}, a_{2,t} ∈ [-1, 1] (PPO Tanh output).
Transition: deterministic via ActionMapper + compute engine.
Reward: Regime-Conditional (RewardFunction).

Key invariants:
  - Look-ahead Bias STRICTLY PROHIBITED: all evaluation windows locked to [t-1, t]
  - All hyperparameters from config.yaml (no hardcoding)
  - Naming: Reward_t, r_port, kappa (not gamma)
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from typing import Optional, Tuple, Dict, Any

from src.env.action_mapper import ActionMapper
from src.env.regret_engine import RegretEngine
from src.env.state_assembler import StateAssembler, StateTuple
from src.env.reward_function import RewardFunction
from src.env.metrics_utils import calculate_sharpe_ratio


class MDPEnvironment(gym.Env):
    """
    Gymnasium-compatible MDP for AE-LLM-RL-FOF PPO training.

    Observation: 10-dim state S_t (np.ndarray)
    Action:      2-dim continuous (a1, a2) ∈ [-1, 1] (Box space)
    Reward:      scalar float (Reward_t)
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Dict[str, Any]):
        """
        Parameters
        ----------
        config : Dict[str, Any]
            Full configuration dict. Must contain keys:
              - action_mapper.*
              - regret_engine.ema_decay
              - reward_function.{lambda_turnover, lambda_te, kappa, eta}
              - state_assembler.{sharpe_clip_low, sharpe_clip_high}
              - env.{tau_min, tau_max, initial_alpha, initial_tau}
        """
        super().__init__()
        self.config = config

        # ---- Sub-components ----
        am_cfg = config.get("action_mapper", {})
        self.action_mapper = ActionMapper(
            alpha_min=am_cfg.get("alpha_min", -0.5),
            alpha_max=am_cfg.get("alpha_max", 0.1),
            tau_delta_range=am_cfg.get("tau_delta_range", 0.1),
        )

        re_cfg = config.get("regret_engine", {})
        self.regret_engine = RegretEngine(
            ema_decay=re_cfg.get("ema_decay", 0.8),
        )

        sa_cfg = config.get("state_assembler", {})
        self.state_assembler = StateAssembler(
            sharpe_clip_low=sa_cfg.get("sharpe_clip_low", -3.0),
            sharpe_clip_high=sa_cfg.get("sharpe_clip_high", 3.0),
        )

        rf_cfg = config.get("reward_function", {})
        self.reward_fn = RewardFunction(
            lambda_turnover=rf_cfg.get("lambda_turnover", 0.001),
            lambda_te=rf_cfg.get("lambda_te", 0.005),
            kappa_mdd=rf_cfg.get("kappa", 2.0),
            eta_regret=rf_cfg.get("eta", 1.0),
        )

        # ---- Env hyper-parameters ----
        env_cfg = config.get("env", {})
        self.tau_min = env_cfg.get("tau_min", 0.0)
        self.tau_max = env_cfg.get("tau_max", 1.0)
        self.initial_alpha = env_cfg.get("initial_alpha", 0.5)
        self.initial_tau = env_cfg.get("initial_tau", 0.5)
        self.episode_max_steps = env_cfg.get("episode_max_steps", 252)

        # ---- Space definitions ----
        self.observation_dim = 10
        self.action_dim = 2
        self.observation_space = gym.spaces.Box(
            low=-5.0, high=5.0, shape=(self.observation_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )

        # ---- Episode state ----
        self._step_count: int = 0
        self._alpha: float = 0.5
        self._tau: float = 0.5
        self._w_final_prev: Optional[np.ndarray] = None
        self._equity_curve: list[float] = []
        self._hwm: float = 1.0
        self._prev_returns_5d: Optional[np.ndarray] = None  # [t-1, t] window

    # --------------------------------------------------------------------- #
    # Public Gymnasium API                                                   #
    # --------------------------------------------------------------------- #

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset to initial conditions for a new episode."""
        super().reset(seed=seed)

        self._step_count = 0
        self._alpha = self.initial_alpha
        self._tau = self.initial_tau
        self._w_final_prev = None
        self._equity_curve = [1.0]
        self._hwm = 1.0
        self._prev_returns_5d = None

        self.regret_engine.reset()

        # Build dummy initial state
        S_0 = self.state_assembler.assemble(
            ae_error=0.0,
            vol_mkt_20d=0.15,
            llm_macro=50.0,
            llm_sentiment=50.0,
            llm_risk=50.0,
            port_sharpe_20d=0.0,
            port_mdd_current=0.0,
            regret_ema_norm=0.0,
            tau_prev=self.initial_tau,
            alpha_prev=self.initial_alpha,
        )

        return S_0, {}

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Parameters
        ----------
        action : np.ndarray, shape (2,)
            Raw PPO actor output [a1, a2] ∈ [-1, 1].

        Returns
        -------
        Tuple[np.ndarray, float, bool, bool, Dict]
            (next_state, Reward_t, terminated, truncated, info)
        """
        a1, a2 = float(action[0]), float(action[1])

        # ---- 1. Action Mapping ----
        delta_alpha, delta_tau = self.action_mapper.map(a1, a2)

        alpha_new = self.action_mapper.clip_alpha(self._alpha + delta_alpha)
        tau_new = self.action_mapper.clip_tau(
            self._tau + delta_tau, self.tau_min, self.tau_max
        )

        # ---- 2. Compute next-state signals (caller must inject live data) ----
        # Here we produce the state vector; real data injection handled via info dict
        # passed from outer loop. This stub returns zeroed metrics.
        ae_error_t = 0.0   # replaced by live E_t in real step
        vol_mkt_20d = 0.15
        llm_macro = 50.0
        llm_sentiment = 50.0
        llm_risk = 50.0
        port_sharpe_20d = 0.0
        r_port = 0.0

        # ---- 3. Update Regret Engine (tied to previous period) ----
        if self._prev_returns_5d is not None and self._w_final_prev is not None:
            regret_ema, regret_ema_norm = self.regret_engine.compute(
                self._w_final_prev, self._prev_returns_5d
            )
        else:
            regret_ema, regret_ema_norm = 0.0, 0.0

        # ---- 4. Reward computation (placeholders; replace with live data) ----
        # In production: unpack live_port_returns, live_benchmark_returns, etc.
        w_dummy = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        port_ret_dummy = np.array([0.0])
        bench_ret_dummy = np.array([0.0])
        equity_dummy = np.array(self._equity_curve)

        Reward_t = self.reward_fn.compute(
            ae_error=ae_error_t,
            threshold_tau=self._tau,
            r_port=r_port,
            w_final_t=w_dummy,
            w_final_t_minus_1=w_dummy if self._w_final_prev is None else self._w_final_prev,
            port_returns=port_ret_dummy,
            benchmark_returns=bench_ret_dummy,
            equity_curve=equity_dummy,
            regret_ema_t=regret_ema,
        )

        # ---- 5. Assemble next state ----
        S_t = self.state_assembler.assemble(
            ae_error=ae_error_t,
            vol_mkt_20d=vol_mkt_20d,
            llm_macro=llm_macro,
            llm_sentiment=llm_sentiment,
            llm_risk=llm_risk,
            port_sharpe_20d=port_sharpe_20d,
            port_mdd_current=0.0,
            regret_ema_norm=regret_ema_norm,
            tau_prev_norm=tau_new,
            alpha_prev=alpha_new,
        )

        # ---- 6. Update internal counters ----
        self._alpha = alpha_new
        self._tau = tau_new
        self._w_final_prev = w_dummy
        self._step_count += 1

        terminated = self._step_count >= self.episode_max_steps
        truncated = False

        info: Dict[str, Any] = {
            "alpha": self._alpha,
            "tau": self._tau,
            "regret_ema": regret_ema,
            "regret_ema_norm": regret_ema_norm,
        }

        return S_t, Reward_t, terminated, truncated, info

    def close(self) -> None:
        """Clean up resources."""
        pass

    # --------------------------------------------------------------------- #
    # Public helpers for outer training loop                                #
    # --------------------------------------------------------------------- #

    def inject_live_data(self, data: Dict[str, Any]) -> None:
        """
        Inject live market / LLM data from outer loop into the env.
        Called every step before step().

        Expected keys:
          ae_error, vol_mkt_20d, llm_macro, llm_sentiment, llm_risk,
          port_sharpe_20d, r_port,
          returns_window_5d (shape 2x5: [t-1, t]), equity_curve,
          w_final_t (current weights)
        """
        self._live_data = data

    def set_w_cand_inverse_vol(self, returns_5d: np.ndarray) -> None:
        """Update the inverse-vol expert candidate using available history."""
        self.regret_engine.update_w_cand_inverse_vol(returns_5d)
