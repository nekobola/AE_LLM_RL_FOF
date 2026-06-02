"""Reward function for the PPO meta-controller."""
from __future__ import annotations

import numpy as np

from src.env.metrics_utils import calculate_current_drawdown, calculate_tracking_error


class RewardFunction:
    """Regime-conditional reward used by the PPO controller."""

    SWITCH_BULL_REWARD: float = 0.010
    SWITCH_BEAR_REWARD: float = 0.010
    SWITCH_BULL_PENALTY: float = 0.015
    SWITCH_BEAR_PENALTY: float = 0.015

    def __init__(
        self,
        lambda_turnover: float = 0.001,
        lambda_te: float = 0.005,
        kappa_mdd: float = 2.0,
        eta_regret: float = 1.0,
        switch_bull_reward: float = 0.010,
        switch_bear_reward: float = 0.010,
        switch_bull_penalty: float = 0.015,
        switch_bear_penalty: float = 0.015,
        lambda_alpha_direct: float = 0.05,
        lambda_endpoint: float = 0.10,
        lambda_relative: float = 1.0,
    ):
        self.lambda_turnover = lambda_turnover
        self.lambda_te = lambda_te
        self.kappa = kappa_mdd
        self.eta = eta_regret
        self.SWITCH_BULL_REWARD = switch_bull_reward
        self.SWITCH_BEAR_REWARD = switch_bear_reward
        self.SWITCH_BULL_PENALTY = switch_bull_penalty
        self.SWITCH_BEAR_PENALTY = switch_bear_penalty
        self.lambda_alpha_direct = lambda_alpha_direct
        self.lambda_endpoint = lambda_endpoint
        self.lambda_relative = lambda_relative

    def compute(
        self,
        ae_error: float,
        threshold_tau: float,
        r_port: float,
        w_final_t: np.ndarray,
        w_final_t_minus_1: np.ndarray,
        port_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        equity_curve: np.ndarray,
        regret_ema_t: float,
        regime_bull: bool,
        alpha_prev: float,
        alpha_current: float,
        normal_return_t: float = 0.0,
        normal_equity_curve: np.ndarray | None = None,
    ) -> float:
        """Compute the scalar reward for the current decision step."""
        turnover = float(np.sum(np.abs(w_final_t - w_final_t_minus_1)))

        switch_reward = 0.0
        switch_threshold = 0.50
        if alpha_current > switch_threshold and not regime_bull:
            switch_reward -= self.SWITCH_BEAR_PENALTY
        elif alpha_current < switch_threshold and regime_bull:
            switch_reward -= self.SWITCH_BULL_PENALTY
        elif alpha_current > switch_threshold and regime_bull:
            switch_reward += self.SWITCH_BULL_REWARD
        elif alpha_current < switch_threshold and not regime_bull:
            switch_reward += self.SWITCH_BEAR_REWARD

        if ae_error < threshold_tau:
            te = calculate_tracking_error(port_returns, benchmark_returns)
            reward_t = r_port - (self.lambda_turnover * turnover) - (self.lambda_te * te)
        else:
            mdd = calculate_current_drawdown(equity_curve)
            mdd_normal = 0.0
            if normal_equity_curve is not None and len(normal_equity_curve) > 1:
                mdd_normal = calculate_current_drawdown(normal_equity_curve)

            relative_return = r_port - normal_return_t
            reward_t = (
                (self.lambda_relative * relative_return)
                - (self.lambda_turnover * turnover)
                - (self.kappa * max(0.0, mdd - mdd_normal))
            )

        endpoint_penalty = self.lambda_endpoint * abs(alpha_current - 0.5)
        reward_t -= endpoint_penalty

        # Small consistency nudge: large one-step flips are allowed, but not free.
        reward_t -= 0.01 * abs(alpha_current - alpha_prev)

        return float(reward_t + switch_reward)
