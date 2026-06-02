"""Action Mapper

Asymmetric mapping from PPO Actor Tanh outputs to portfolio execution deltas.

Design principle: building positions must be SMOOTH; escaping crises must be DECISIVE.
- Fusion-ratio delta (Δα_t): asymmetric map [-1,1] → [-0.5, 0.1]
  - Full negative (-1): cut 50% equity in one week (two-week full liquidation)
  - Full positive (+1): add 10% per week (anti-chasing)
- Threshold delta (Δτ_t): symmetric map [-1,1] → [-0.1, 0.1]
"""
from __future__ import annotations

import numpy as np
from typing import Tuple


class ActionMapper:
    """
    Maps raw PPO actor outputs (a1, a2 ∈ [-1, 1] after Tanh) to
    executable portfolio deltas (Δα_t, Δτ_t).
    """

    def __init__(
        self,
        alpha_min: float = -0.5,
        alpha_max: float = 0.1,
        tau_delta_range: float = 0.1,
        alpha_bias: float = 0.0,
    ):
        """
        Parameters
        ----------
        alpha_min : float
            Minimum fusion-ratio delta per week (cut speed).
            Default -0.5 means full liquidation in 2 weeks.
        alpha_max : float
            Maximum fusion-ratio delta per week (add speed).
            Default 0.1 means max 10% weekly addition.
        tau_delta_range : float
            Symmetric threshold delta range. Default ±0.1.
        alpha_bias : float
            Bias term to break a1=0 stable不动点.
            Default -0.05: a1=0 → delta=-0.05 (slight drift toward lower alpha).
            After bias: delta = a1*alpha_range + alpha_bias, clamped to [alpha_min, alpha_max].
        """
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.tau_delta_range = tau_delta_range
        self.alpha_bias = alpha_bias

    def map(self, a1: float, a2: float) -> Tuple[float, float]:
        """
        Parameters
        ----------
        a1 : float
            Raw actor output for fusion-ratio delta, ∈ [-1, 1].
        a2 : float
            Raw actor output for threshold delta, ∈ [-1, 1].

        Returns
        -------
        Tuple[float, float]
            (delta_alpha, delta_tau) — executable deltas.
        """
        delta_alpha = self._map_alpha(a1)
        delta_tau = self._map_tau(a2)
        return delta_alpha, delta_tau

    def _map_alpha(self, a1: float) -> float:
        """
        Biased symmetric map: [-1, 1] → [alpha_min, alpha_max] with small negative bias.
        
        delta = a1 * alpha_range + bias
        a1=-1 → delta = -alpha_range + bias ≈ -0.35 (max defensive)
        a1=0  → delta = bias = -0.05 (slight defensive drift, BREAKS不动点)
        a1=+1 → delta = +alpha_range + bias ≈ +0.25 (max offensive)
        
        Bias ensures a1=0 is not a stable equilibrium — alpha drifts slightly lower
        until actor learns to output a1 > 0 to compensate.
        """
        alpha_range = (self.alpha_max - self.alpha_min) / 2.0
        delta = a1 * alpha_range + self.alpha_bias
        return float(np.clip(delta, self.alpha_min, self.alpha_max))

    def _map_tau(self, a2: float) -> float:
        """
        Symmetric map: [-1, 1] → [-tau_delta_range, +tau_delta_range]
        """
        return a2 * self.tau_delta_range

    @staticmethod
    def clip_alpha(alpha: float) -> float:
        """Clip fusion ratio to valid range [0.0, 1.0]."""
        return float(np.clip(alpha, 0.0, 1.0))

    @staticmethod
    def clip_tau(tau: float, tau_min: float, tau_max: float) -> float:
        """Clip threshold to valid range [tau_min, tau_max]."""
        return float(np.clip(tau, tau_min, tau_max))
