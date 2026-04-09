"""Regret Engine (Hindsight Evaluation Engine)

Constructs an orthogonal expert library as the ex-post evaluation benchmark.
Strict time-alignment with the evaluation period; zero Look-ahead Bias.

Expert weight formations (5-dim):
  W_cand[0]  = [0, 0, 0, 0, 1.0]         — Absolute cash expert
  W_cand[1]  = [0, 0, 0, 1.0, 0]         — Absolute gold expert
  W_cand[2]  = [0, 0, 1.0, 0, 0]         — Absolute pure-bond expert
  W_cand[3]  = inverse-vol weights        — Baseline defensive expert
  W_cand[4-9]= equally-spaced grid combos — 6 grid points in [bond, hedge, cash]
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Static expert candidate library  W_cand  (5 assets: V1-V5)
# ---------------------------------------------------------------------------
_W_CANDIDATES: list[np.ndarray] = [
    # Absolute specialists
    np.array([0.0, 0.0, 0.0, 0.0, 1.0]),   # cash
    np.array([0.0, 0.0, 0.0, 1.0, 0.0]),   # gold
    np.array([0.0, 0.0, 1.0, 0.0, 0.0]),   # pure bond
    # Baseline: inverse-vol (placeholder; updated in reset())
    np.array([0.25, 0.25, 0.25, 0.125, 0.125]),
    # Grid points: bond-hedge-cash space
    np.array([0.0, 0.0, 0.6, 0.4, 0.0]),
    np.array([0.0, 0.0, 0.4, 0.6, 0.0]),
    np.array([0.0, 0.0, 0.7, 0.2, 0.1]),
    np.array([0.0, 0.0, 0.5, 0.3, 0.2]),
    np.array([0.0, 0.0, 0.3, 0.5, 0.2]),
    np.array([0.0, 0.0, 0.2, 0.4, 0.4]),
]


class RegretEngine:
    """
    Hindsight evaluation engine.

    At each decision point t, strictly evaluates the period [t-1, t]:

    r_actual  = performance of w_final_prev under true market in [t-1, t]
    r_opt     = max_{w ∈ W_cand} performance of w under true market in [t-1, t]

    Regret_raw  = max(0, r_opt - r_actual)
    Regret_ema  = 0.8 * Regret_ema_prev + 0.2 * Regret_raw
    """

    def __init__(self, ema_decay: float = 0.8):
        """
        Parameters
        ----------
        ema_decay : float
            EMA smoothing factor. Default 0.8 (heavily weight historical).
        """
        self.ema_decay = ema_decay
        self.regret_ema: float = 0.0
        self.regret_ema_max_hist: float = 1e-6  # avoid div-by-zero; updated live
        self._w_cand = [w.copy() for w in _W_CANDIDATES]

    def reset(self) -> None:
        """Reset internal EMA state on new episode."""
        self.regret_ema = 0.0
        self.regret_ema_max_hist = 1e-6

    def update_w_cand_inverse_vol(self, returns_5d: np.ndarray) -> None:
        """
        Replace W_cand[3] with current inverse-vol weights.

        Must be called at episode start (burn-in period),
        strictly using data available up to t-1.

        Parameters
        ----------
        returns_5d : np.ndarray, shape (N, 5)
            Historical 5-asset return matrix.
        """
        if returns_5d.shape[0] < 20:
            return
        vol = np.std(returns_5d, ddof=1)
        inv_vol = 1.0 / (vol + 1e-9)
        weights = inv_vol / inv_vol.sum()
        self._w_cand[3] = weights

    def compute(
        self,
        w_final_prev: np.ndarray,
        returns_window: np.ndarray,
    ) -> tuple[float, float]:
        """
        Compute regret for the just-completed period [t-1, t].

        Parameters
        ----------
        w_final_prev : np.ndarray, shape (5,)
            Portfolio fusion weights used in the period.
        returns_window : np.ndarray, shape (5,)  or (2, 5)
            Asset return(s) over the period.
            If shape (2, 5): interpret as [prev_period_return, curr_return]
              — we use only the last row (most recent period).
            Convention: first row = t-1, last row = t.

        Returns
        -------
        Tuple[float, float]
            (regret_ema, regret_ema_normalized) ∈ [0, 1]
        """
        # Extract actual period return for the portfolio
        if returns_window.ndim == 2:
            period_return = returns_window[-1]  # last row = current period
        else:
            period_return = returns_window

        r_actual = float(np.dot(w_final_prev, period_return))

        # Evaluate all 10 experts
        expert_returns = np.array([float(np.dot(w, period_return)) for w in self._w_cand])
        r_opt = float(np.max(expert_returns))

        # Raw regret
        regret_raw = max(0.0, r_opt - r_actual)

        # EMA smoothing
        self.regret_ema = self.ema_decay * self.regret_ema + (1.0 - self.ema_decay) * regret_raw

        # Track running max for normalization
        if self.regret_ema > self.regret_ema_max_hist:
            self.regret_ema_max_hist = self.regret_ema

        regret_ema_norm = float(np.clip(self.regret_ema / self.regret_ema_max_hist, 0.0, 1.0))

        return self.regret_ema, regret_ema_norm

    @property
    def w_candidates(self) -> list[np.ndarray]:
        """Read-only access to expert weight library."""
        return [w.copy() for w in self._w_cand]
