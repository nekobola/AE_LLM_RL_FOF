"""Gymnasium environment for PPO training."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from src.env.action_mapper import ActionMapper
from src.env.metrics_utils import calculate_current_drawdown, calculate_sharpe_ratio
from src.env.regret_engine import RegretEngine
from src.env.reward_function import RewardFunction
from src.env.state_assembler import StateAssembler


class MDPEnvironment(gym.Env):
    """Gymnasium-compatible MDP for the dual-track controller."""

    metadata = {"render_modes": []}

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        am_cfg = config.get("action_mapper", {})
        self.action_mapper = ActionMapper(
            alpha_min=am_cfg.get("alpha_min", -0.5),
            alpha_max=am_cfg.get("alpha_max", 0.1),
            tau_delta_range=am_cfg.get("tau_delta_range", 0.1),
            alpha_bias=am_cfg.get("alpha_bias", -0.05),
        )

        re_cfg = config.get("regret_engine", {})
        self.regret_engine = RegretEngine(ema_decay=re_cfg.get("ema_decay", 0.8))

        sa_cfg = config.get("state_assembler", {})
        env_cfg = config.get("env", {})
        self.state_assembler = StateAssembler(
            sharpe_clip_low=sa_cfg.get("sharpe_clip_low", -3.0),
            sharpe_clip_high=sa_cfg.get("sharpe_clip_high", 3.0),
            tau_min=env_cfg.get("tau_min", 0.0),
            tau_max=env_cfg.get("tau_max", 50.0),
        )

        rf_cfg = config.get("reward_function", {})
        self.reward_fn = RewardFunction(
            lambda_turnover=rf_cfg.get("lambda_turnover", 0.001),
            lambda_te=rf_cfg.get("lambda_te", 0.005),
            kappa_mdd=rf_cfg.get("kappa", 2.0),
            eta_regret=rf_cfg.get("eta", 1.0),
            switch_bull_reward=rf_cfg.get("switch_bull_reward", 0.010),
            switch_bear_reward=rf_cfg.get("switch_bear_reward", 0.010),
            switch_bull_penalty=rf_cfg.get("switch_bull_penalty", 0.015),
            switch_bear_penalty=rf_cfg.get("switch_bear_penalty", 0.015),
            lambda_alpha_direct=rf_cfg.get("lambda_alpha_direct", 0.05),
            lambda_endpoint=rf_cfg.get("lambda_endpoint", 0.10),
            lambda_relative=rf_cfg.get("lambda_relative", 1.0),
        )

        self.tau_min = env_cfg.get("tau_min", 0.0)
        self.tau_max = env_cfg.get("tau_max", 50.0)
        self.initial_alpha = env_cfg.get("initial_alpha", 0.5)
        self.initial_tau = env_cfg.get("initial_tau", 20.0)
        self.episode_max_steps = env_cfg.get("episode_max_steps", 252)

        self.observation_dim = 10
        self.action_dim = 2
        self.observation_space = gym.spaces.Box(
            low=-5.0, high=5.0, shape=(self.observation_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )

        self._step_count = 0
        self._alpha = self.initial_alpha
        self._tau = self.initial_tau
        self._w_final_prev: Optional[np.ndarray] = None
        self._equity_curve: list[float] = []
        self._normal_equity_curve: list[float] = []
        self._portfolio_returns_history: list[float] = []
        self._benchmark_returns_history: list[float] = []
        self._normal_returns_history: list[float] = []
        self._pending_returns_window: Optional[np.ndarray] = None
        self._live_data: Optional[Dict[str, Any]] = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self._step_count = 0
        self._alpha = self.initial_alpha
        self._tau = self.initial_tau
        self._w_final_prev = None
        self._equity_curve = [1.0]
        self._normal_equity_curve = [1.0]
        self._portfolio_returns_history = []
        self._benchmark_returns_history = []
        self._normal_returns_history = []
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
            tau_prev=self.initial_tau,
            alpha_prev=self.initial_alpha,
        )
        return state, {}

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        a1, a2 = float(action[0]), float(action[1])
        delta_alpha, delta_tau = self.action_mapper.map(a1, a2)
        alpha_new = self.action_mapper.clip_alpha(self._alpha + delta_alpha)
        tau_new = self.action_mapper.clip_tau(self._tau + delta_tau, self.tau_min, self.tau_max)

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

        if self._pending_returns_window is not None:
            current_asset_returns = np.asarray(self._pending_returns_window[-1], dtype=np.float64)
        else:
            current_asset_returns = np.asarray(ld.get("asset_returns_t", [0.0] * 5), dtype=np.float64)
        current_asset_returns = np.nan_to_num(current_asset_returns, nan=0.0)

        if self._pending_returns_window is not None and self._w_final_prev is not None:
            regret_ema, regret_ema_norm = self.regret_engine.compute(
                self._w_final_prev,
                self._pending_returns_window,
            )
        else:
            regret_ema, regret_ema_norm = 0.0, 0.0

        w_normal_t = np.asarray(ld.get("w_normal_t", [0.2] * 5), dtype=np.float64)
        w_event_t = np.asarray(
            ld.get("w_event_t", [0.0, 0.0, 0.33, 0.33, 0.34]),
            dtype=np.float64,
        )

        w_final_t = alpha_new * w_event_t + (1.0 - alpha_new) * w_normal_t
        w_final_t = np.clip(w_final_t, 0.0, 1.0)
        w_final_t = w_final_t / (w_final_t.sum() + 1e-9)
        w_final_t_minus_1 = self._w_final_prev if self._w_final_prev is not None else w_final_t

        r_port = float(np.dot(w_final_t, current_asset_returns))
        r_normal = float(np.dot(w_normal_t, current_asset_returns))
        benchmark_return = float(current_asset_returns[0]) if current_asset_returns.size > 0 else 0.0

        port_returns = np.asarray(self._portfolio_returns_history + [r_port], dtype=np.float64)
        benchmark_returns = np.asarray(
            self._benchmark_returns_history + [benchmark_return],
            dtype=np.float64,
        )

        equity_live = np.asarray(
            self._equity_curve + [self._equity_curve[-1] * (1.0 + r_port)],
            dtype=np.float64,
        )
        normal_equity_live = np.asarray(
            self._normal_equity_curve + [self._normal_equity_curve[-1] * (1.0 + r_normal)],
            dtype=np.float64,
        )

        regime_bull = ae_error_t < self._tau
        reward_t = self.reward_fn.compute(
            ae_error=ae_error_t,
            threshold_tau=self._tau,
            r_port=r_port,
            w_final_t=w_final_t,
            w_final_t_minus_1=w_final_t_minus_1,
            port_returns=port_returns,
            benchmark_returns=benchmark_returns,
            equity_curve=equity_live,
            regret_ema_t=regret_ema,
            regime_bull=regime_bull,
            alpha_prev=self._alpha,
            alpha_current=alpha_new,
            normal_return_t=r_normal,
            normal_equity_curve=normal_equity_live,
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
            tau_prev=tau_new,
            alpha_prev=alpha_new,
        )

        self._alpha = alpha_new
        self._tau = tau_new
        self._w_final_prev = w_final_t.copy()
        self._equity_curve = equity_live.tolist()
        self._normal_equity_curve = normal_equity_live.tolist()
        self._portfolio_returns_history = port_returns.tolist()
        self._benchmark_returns_history = benchmark_returns.tolist()
        self._normal_returns_history.append(r_normal)
        self._step_count += 1

        terminated = self._step_count >= self.episode_max_steps
        truncated = False
        info = {
            "alpha": self._alpha,
            "tau": self._tau,
            "regret_ema": regret_ema,
            "regret_ema_norm": regret_ema_norm,
            "r_port": r_port,
            "r_normal": r_normal,
        }
        return next_state, reward_t, terminated, truncated, info

    def close(self) -> None:
        pass

    def inject_live_data(self, data: Dict[str, Any]) -> None:
        """Inject per-step historical/live features before calling step()."""
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
