"""Reward function for the PPO meta-controller (Stage 7 single-track V3.1).

Stage 7 redesign: NormalTrack is removed. PPO controls V3.1 theta directly.
Reward = portfolio return, penalized for turnover, theta jitter, drawdown.
A signal-strength bonus rewards aggressive theta when LLM/AE signals agree.
"""
from __future__ import annotations

import numpy as np

from src.env.metrics_utils import calculate_current_drawdown


class RewardFunction:
    """Single-track reward for PPO-controlled V3.1 theta.

    Components (all in same units as r_port, which is weekly decimal return):
      r_t = r_port_t
            - lambda_turnover * Σ|w_t - w_{t-1}|           (turnover cost)
            - lambda_theta_change * |θ_t - θ_{t-1}|         (theta jitter cost)
            - lambda_mdd * max(0, mdd_t - mdd_target)       (drawdown penalty)
            + lambda_signal * signal_strength * (θ_t - 1)   (signal-conditional bonus)
    """

    def __init__(
        self,
        lambda_turnover: float = 0.001,
        lambda_theta_change: float = 0.0005,
        lambda_mdd: float = 0.5,
        mdd_target: float = 0.05,
        lambda_signal: float = 0.08,
        lambda_theta_baseline: float = 0.04,
    ):
        self.lambda_turnover = lambda_turnover
        self.lambda_theta_change = lambda_theta_change
        self.lambda_mdd = lambda_mdd
        self.mdd_target = mdd_target
        self.lambda_signal = lambda_signal
        self.lambda_theta_baseline = lambda_theta_baseline

    def compute(
        self,
        r_port: float,
        w_t: np.ndarray,
        w_t_minus_1: np.ndarray,
        theta_t: float,
        theta_t_minus_1: float,
        equity_curve: np.ndarray,
        signal_strength: float = 0.0,
    ) -> float:
        """Compute scalar reward for one decision step.

        Parameters
        ----------
        r_port : float
            Realized portfolio weekly return (decimal).
        w_t, w_t_minus_1 : np.ndarray
            Current and previous portfolio weights (5-dim).
        theta_t, theta_t_minus_1 : float
            Current and previous V3.1 theta ∈ [0, 2].
        equity_curve : np.ndarray
            Cumulative NAV up to and including current step.
        signal_strength : float
            Composite |LLM signals + AE regime| ∈ [0, ~1].
            Higher = stronger signal → reward aggressive theta more.
        """
        turnover = float(np.sum(np.abs(w_t - w_t_minus_1)))
        theta_change = abs(theta_t - theta_t_minus_1)
        mdd = calculate_current_drawdown(equity_curve) if len(equity_curve) > 1 else 0.0

        # Signal-conditional bonus: aggressive theta (theta > 1) is rewarded
        # when signals are strong; defensive theta (theta < 1) is implicitly
        # rewarded by lower turnover (less to penalize).
        signal_bonus = self.lambda_signal * signal_strength * (theta_t - 1.0)

        # Theta-baseline bonus: keep theta near 1.0 (V3.1 neutral default).
        # 0.005 * (1 - |theta - 1|): max bonus 0.005 at theta=1, decays to 0
        # at theta=0 or theta=2. Prevents PPO from collapsing to theta=0
        # while still allowing it to deviate when signal_bonus outweighs.
        baseline_bonus = self.lambda_theta_baseline * (1.0 - abs(theta_t - 1.0))

        reward = (
            r_port
            - self.lambda_turnover * turnover
            - self.lambda_theta_change * theta_change
            - self.lambda_mdd * max(0.0, mdd - self.mdd_target)
            + signal_bonus
            + baseline_bonus
        )
        return float(reward)
