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
        """
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.tau_delta_range = tau_delta_range

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
        Asymmetric map: [-1, 1] → [alpha_min, alpha_max]
        Full negative (-1) → alpha_min (fast cut)
        Full positive (+1) → alpha_max (slow add)
        Uses linear interpolation with asymmetric bounds.
        """
        # Scale from [-1, 1] to [0, 1]
        t = (a1 + 1.0) / 2.0  # t ∈ [0, 1]
        # Map to asymmetric range
        return self.alpha_min + t * (self.alpha_max - self.alpha_min)

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
