"""Gymnasium environment for PPO training (Stage 7 single-track V3.1).

PPO output: action ∈ [-1, 1] (Tanh, 1-dim)
MDP output: V3.1 theta ∈ [0, 2] (PPO controls exponential tilt gain)
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from src.compute.v31_engine import V31Engine
from src.env.action_mapper import ThetaActionMapper
from src.env.metrics_utils import calculate_current_drawdown, calculate_sharpe_ratio
from src.env.regret_engine import RegretEngine
from src.env.reward_function import RewardFunction
from src.env.state_assembler import StateAssembler


class MDPEnvironment(gym.Env):
    """Stage 7: Gymnasium-compatible MDP for PPO-controlled V3.1 theta."""

    metadata = {"render_modes": []}

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        # Stage 7: ThetaActionMapper replaces (alpha, tau) ActionMapper
        am_cfg = config.get("action_mapper", {})
        self.action_mapper = ThetaActionMapper(
            theta_min=am_cfg.get("theta_min", 0.0),
            theta_neutral=am_cfg.get("theta_neutral", 1.0),
            theta_max=am_cfg.get("theta_max", 2.0),
        )

        re_cfg = config.get("regret_engine", {})
        self.regret_engine = RegretEngine(ema_decay=re_cfg.get("ema_decay", 0.8))

        sa_cfg = config.get("state_assembler", {})
        env_cfg = config.get("env", {})
        self.state_assembler = StateAssembler(
            sharpe_clip_low=sa_cfg.get("sharpe_clip_low", -3.0),
            sharpe_clip_high=sa_cfg.get("sharpe_clip_high", 3.0),
            theta_min=env_cfg.get("theta_min", 0.0),
            theta_max=env_cfg.get("theta_max", 2.0),
        )

        rf_cfg = config.get("reward_function", {})
        self.reward_fn = RewardFunction(
            lambda_turnover=rf_cfg.get("lambda_turnover", 0.001),
            lambda_theta_change=rf_cfg.get("lambda_theta_change", 0.005),
            lambda_mdd=rf_cfg.get("lambda_mdd", 2.0),
            mdd_target=rf_cfg.get("mdd_target", 0.05),
            lambda_signal=rf_cfg.get("lambda_signal", 0.01),
        )

        self.v31_engine = V31Engine(config)
        self.initial_theta = env_cfg.get("initial_theta", 1.0)
        self.episode_max_steps = env_cfg.get("episode_max_steps", 252)

        # Stage 7: 9-dim state, 1-dim action
        self.observation_dim = 9
        self.action_dim = 1
        self.observation_space = gym.spaces.Box(
            low=-5.0, high=5.0, shape=(self.observation_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )

        self._step_count = 0
        self._theta = self.initial_theta
        self._w_prev: Optional[np.ndarray] = None
        self._equity_curve: list[float] = [1.0]
        self._portfolio_returns_history: list[float] = []
        self._pending_returns_window: Optional[np.ndarray] = None
        self._live_data: Optional[Dict[str, Any]] = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._step_count = 0
        self._theta = self.initial_theta
        self._w_prev = None
        self._equity_curve = [1.0]
        self._portfolio_returns_history = []
        self._pending_returns_window = None
        self._live_data = None
        self.regret_engine.reset()

        state = self.state_assembler.assemble(
            ae_error=0.0,
            vol_mkt_20d=0.15,
            llm_macro=50.0,
            llm_sentiment=50.0,
            llm_risk=50.0,
            port_sharpe_20d=0.0,
            port_mdd_current=0.0,
            regret_ema_norm=0.0,
            theta_prev=self.initial_theta,
        )
        return state, {}

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action_arr = np.atleast_1d(np.asarray(action, dtype=np.float32))
        a = float(action_arr[0])
        theta_new = self.action_mapper.map(a)

        if self._live_data is None:
            raise RuntimeError(
                "MDPEnvironment.step() requires live data injection. "
                "Call env.inject_live_data(data) before each step()."
            )
        ld = self._live_data

        ae_error_t = float(ld.get("ae_error", 0.0))
        vol_mkt_20d = float(ld.get("vol_mkt_20d", 0.15))
        llm_macro = float(ld.get("llm_macro", 50.0))
        llm_sentiment = float(ld.get("llm_sentiment", 50.0))
        llm_risk = float(ld.get("llm_risk", 50.0))

        # Stage 7: AE threshold tau comes from AE regime config, not PPO.
        # Use a fixed default; could be made regime-conditional in future.
        tau_fixed = float(ld.get("tau", 15.0))

        if self._pending_returns_window is not None:
            current_asset_returns = np.asarray(self._pending_returns_window[-1], dtype=np.float64)
        else:
            current_asset_returns = np.asarray(ld.get("asset_returns_t", [0.0] * 5), dtype=np.float64)
        current_asset_returns = np.nan_to_num(current_asset_returns, nan=0.0)

        if self._pending_returns_window is not None and self._w_prev is not None:
            regret_ema, regret_ema_norm = self.regret_engine.compute(
                self._w_prev,
                self._pending_returns_window,
            )
        else:
            regret_ema, regret_ema_norm = 0.0, 0.0

        # V3.1 compute with PPO-supplied theta
        w_event = self.v31_engine.compute(
            self._pending_returns_window if self._pending_returns_window is not None
            else np.random.randn(5, 5) * 0.01,
            llm_macro=llm_macro,
            llm_sentiment=llm_sentiment,
            llm_risk=llm_risk,
            ae_error=ae_error_t,
            tau=tau_fixed,
            theta=theta_new,
        )
        w_event = np.clip(w_event, 0.0, 1.0)
        w_event = w_event / (w_event.sum() + 1e-9)

        w_final_t = w_event
        w_final_t_minus_1 = self._w_prev if self._w_prev is not None else w_final_t

        r_port = float(np.dot(w_final_t, current_asset_returns))

        port_returns = np.asarray(self._portfolio_returns_history + [r_port], dtype=np.float64)
        equity_live = np.asarray(
            self._equity_curve + [self._equity_curve[-1] * (1.0 + r_port)],
            dtype=np.float64,
        )

        # Signal strength: |LLM signals away from neutral| + AE regime intensity
        llm_signal_strength = (abs(llm_macro - 50) + abs(llm_sentiment - 50) + abs(llm_risk - 50)) / 150.0
        ae_signal_strength = min(abs(ae_error_t - tau_fixed) / 30.0, 1.0)
        signal_strength = float(np.clip((llm_signal_strength + ae_signal_strength) / 2.0, 0.0, 1.0))

        reward_t = self.reward_fn.compute(
            r_port=r_port,
            w_t=w_final_t,
            w_t_minus_1=w_final_t_minus_1,
            theta_t=theta_new,
            theta_t_minus_1=self._theta,
            equity_curve=equity_live,
            signal_strength=signal_strength,
        )

        port_mdd_current = calculate_current_drawdown(equity_live)
        port_sharpe_20d = calculate_sharpe_ratio(port_returns[-20:]) if len(port_returns) >= 2 else 0.0
        next_state = self.state_assembler.assemble(
            ae_error=ae_error_t,
            vol_mkt_20d=vol_mkt_20d,
            llm_macro=llm_macro,
            llm_sentiment=llm_sentiment,
            llm_risk=llm_risk,
            port_sharpe_20d=port_sharpe_20d,
            port_mdd_current=port_mdd_current,
            regret_ema_norm=regret_ema_norm,
            theta_prev=theta_new,
        )

        self._theta = theta_new
        self._w_prev = w_final_t.copy()
        self._equity_curve = equity_live.tolist()
        self._portfolio_returns_history = port_returns.tolist()
        self._step_count += 1

        terminated = self._step_count >= self.episode_max_steps
        truncated = False
        info = {
            "theta": self._theta,
            "regret_ema": regret_ema,
            "regret_ema_norm": regret_ema_norm,
            "r_port": r_port,
            "signal_strength": signal_strength,
        }
        return next_state, reward_t, terminated, truncated, info

    def close(self) -> None:
        pass

    def inject_live_data(self, data: Dict[str, Any]) -> None:
        self._live_data = data
        returns_window = data.get("returns_window_5d")
        if returns_window is None:
            self._pending_returns_window = None
            return
        arr = np.asarray(returns_window, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] == 5:
            self._pending_returns_window = arr
        else:
            self._pending_returns_window = None

    def set_w_cand_inverse_vol(self, returns_5d: np.ndarray) -> None:
        self.regret_engine.update_w_cand_inverse_vol(returns_5d)
