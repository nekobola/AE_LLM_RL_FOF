"""Action Mapper

Asymmetric mapping from PPO Actor Tanh outputs to portfolio execution deltas.

Stage 6 (legacy): Fusion ratio + threshold deltas.
  - Fusion-ratio delta (Δα_t): asymmetric map [-1,1] → [-0.5, 0.1]
  - Threshold delta (Δτ_t): symmetric map [-1,1] → [-0.1, 0.1]

Stage 7 (current): PPO controls V3.1 theta directly.
  - PPO output a ∈ [-1, 1] (Tanh) → theta ∈ [0, 2]
  - theta = 0  → pure ERC (b0 uniform, no tilting)
  - theta = 1  → balanced
  - theta = 2  → aggressive concentration
"""
from __future__ import annotations

import numpy as np
from typing import Tuple


class ActionMapper:
    """Stage 6: PPO outputs (a1, a2) → (delta_alpha, delta_tau).

    Kept for backward compatibility with checkpoints; not used in Stage 7
    inference path.
    """

    def __init__(
        self,
        alpha_min: float = -0.5,
        alpha_max: float = 0.1,
        tau_delta_range: float = 0.1,
        alpha_bias: float = 0.0,
    ):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.tau_delta_range = tau_delta_range
        self.alpha_bias = alpha_bias

    def map(self, a1: float, a2: float) -> Tuple[float, float]:
        delta_alpha = self._map_alpha(a1)
        delta_tau = self._map_tau(a2)
        return delta_alpha, delta_tau

    def _map_alpha(self, a1: float) -> float:
        alpha_range = (self.alpha_max - self.alpha_min) / 2.0
        delta = a1 * alpha_range + self.alpha_bias
        return float(np.clip(delta, self.alpha_min, self.alpha_max))

    def _map_tau(self, a2: float) -> float:
        return a2 * self.tau_delta_range

    @staticmethod
    def clip_alpha(alpha: float) -> float:
        return float(np.clip(alpha, 0.0, 1.0))

    @staticmethod
    def clip_tau(tau: float, tau_min: float, tau_max: float) -> float:
        return float(np.clip(tau, tau_min, tau_max))


class ThetaActionMapper:
    """Stage 7: PPO output a ∈ [-1, 1] → V3.1 theta ∈ [0, 2].

    Linear mapping centered at theta=1.0 (V3.1 default behavior).
      a = -1 → theta = 0  (pure ERC, no signal tilt)
      a =  0 → theta = 1  (balanced, V3.1 neutral)
      a = +1 → theta = 2  (aggressive concentration on high-score assets)

    PPO is free to learn any regime-conditional theta ∈ [0, 2]. The 0
    boundary is "no information, just equal-weight" — a natural floor.
    """

    THETA_MIN: float = 0.0
    THETA_NEUTRAL: float = 1.0
    THETA_MAX: float = 2.0

    def __init__(
        self,
        theta_min: float = 0.0,
        theta_neutral: float = 1.0,
        theta_max: float = 2.0,
    ):
        assert theta_min < theta_neutral < theta_max
        self.theta_min = theta_min
        self.theta_neutral = theta_neutral
        self.theta_max = theta_max
        # slope: a ∈ [-1, 1] maps to theta ∈ [theta_min, theta_max]
        # Use theta_neutral as the center: a=0 → theta_neutral
        self._slope = (theta_max - theta_min) / 2.0
        self._offset = (theta_max + theta_min) / 2.0

    def map(self, a: float) -> float:
        """Map raw PPO actor output to theta.

        Parameters
        ----------
        a : float
            Tanh-bounded actor output, ∈ [-1, 1].

        Returns
        -------
        float
            V3.1 theta ∈ [theta_min, theta_max].
        """
        theta = self._offset + a * self._slope
        return float(np.clip(theta, self.theta_min, self.theta_max))

    @staticmethod
    def clip_theta(theta: float) -> float:
        return float(np.clip(theta, 0.0, 2.0))

