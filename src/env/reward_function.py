"""Reward Function

Regime-Conditional Reward for the PPO agent.

Normal Regime (E_t < τ_t):
  Reward_t = r_port - (lambda_1 * Turnover) - (lambda_2 * TrackingError)

Event Regime (E_t >= τ_t):
  Reward_t = r_port - (lambda_1 * Turnover) - (kappa * mdd) - (eta * Regret_ema_t)

Key naming constraints (anti-RL-library-conflict):
  - RL reward variable: Reward_t (NEVER R_t or R)
  - Portfolio return: r_port
  - Max-drawdown penalty coefficient: kappa (NEVER gamma)
  - All Python variables: snake_case
"""
from __future__ import annotations

import numpy as np
from src.env.metrics_utils import calculate_tracking_error, calculate_current_drawdown


class RewardFunction:
    """
    Regime-conditional reward function.

    Switches evaluation regime based on the AE reconstruction error threshold τ_t.
    """

    def __init__(
        self,
        lambda_turnover: float = 0.001,
        lambda_te: float = 0.005,
        kappa_mdd: float = 2.0,
        eta_regret: float = 1.0,
    ):
        """
        Parameters
        ----------
        lambda_turnover : float
            Penalty coefficient for portfolio turnover.
            Default 0.001 per 1-unit weight change.
        lambda_te : float
            Tracking error penalty coefficient (normal regime).
            Default 0.005 per 1-unit TE.
        kappa_mdd : float
            Max drawdown penalty coefficient (event regime).
            Named kappa to avoid conflict with RL discount factor gamma.
            Default 2.0.
        eta_regret : float
            Regret penalty coefficient (event regime).
            Default 1.0.
        """
        self.lambda_turnover = lambda_turnover
        self.lambda_te = lambda_te
        self.kappa = kappa_mdd
        self.eta = eta_regret

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
    ) -> float:
        """
        Compute Reward_t for the current step.

        Parameters
        ----------
        ae_error : float
            AE reconstruction error E_t.
        threshold_tau : float
            Current threshold τ_t.
        r_port : float
            Portfolio absolute return for current period.
        w_final_t : np.ndarray, shape (5,)
            Portfolio weights at end of current period.
        w_final_t_minus_1 : np.ndarray, shape (5,)
            Portfolio weights at start of current period.
        port_returns : np.ndarray, shape (N,)
            Portfolio return series for TE calculation.
        benchmark_returns : np.ndarray, shape (N,)
            Benchmark return series for TE calculation.
        equity_curve : np.ndarray, shape (T,)
            Cumulative NAV curve.
        regret_ema_t : float
            Smoothed regret value from RegretEngine.

        Returns
        -------
        float
            Reward_t scalar.
        """
        # Turnover = sum(|w_t - w_{t-1}|)
        turnover = float(np.sum(np.abs(w_final_t - w_final_t_minus_1)))

        if ae_error < threshold_tau:
            # ---- Normal Regime: close-to-benchmark, control friction ----
            te = calculate_tracking_error(port_returns, benchmark_returns)
            reward_t = r_port - (self.lambda_turnover * turnover) - (self.lambda_te * te)
        else:
            # ---- Event Regime: aggressive defence, punish DD and regret ----
            mdd = calculate_current_drawdown(equity_curve)
            reward_t = (
                r_port
                - (self.lambda_turnover * turnover)
                - (self.kappa * mdd)
                - (self.eta * regret_ema_t)
            )

        return float(reward_t)
