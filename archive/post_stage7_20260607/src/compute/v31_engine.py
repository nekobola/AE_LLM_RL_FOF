"""V31 Engine: thin wrapper around EventTrackV31.

Stage 7 refactor: NormalTrack removed. The system output is directly
EventTrackV31's w_event. PPO controls the V3.1 theta (exponential tilt gain)
via this engine.
"""
from __future__ import annotations
import logging
import numpy as np

from .event_track_v3_1 import EventTrackV31

logger = logging.getLogger(__name__)


class V31Engine:
    """Single-track engine. PPO controls theta; AE controls V_DEFENSE mixing.

    Pipeline (replaces dual_track_engine.compute()):
        1.  f = [m, s, r, equity_stress, sat_lead]    (5-D feature)
        2.  s = W @ f                                  (matrix-normalized scores)
        3.  bear_pressure = sigmoid((E_t - τ) / scale)  (AE shifter)
        4.  s_final = (1 - bear_pressure) * s + bear_pressure * V_DEFENSE
        5.  b = b0 * exp(θ * s_final) / Σ              (PPO-controlled θ)
        6.  project b onto box-constrained simplex

    Returns
    -------
    w_event : np.ndarray
        shape (5,) portfolio weights summing to 1, in
        asset order [broad, satellite, fi, gold, cash].
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.event_track = EventTrackV31()

    def compute(
        self,
        returns_5d: np.ndarray,
        llm_macro: float = 50.0,
        llm_sentiment: float = 50.0,
        llm_risk: float = 50.0,
        ae_error: float | None = None,
        tau: float | None = None,
        theta: float = 0.7,
    ) -> np.ndarray:
        """Compute V3.1 portfolio weights with PPO-supplied theta.

        Parameters
        ----------
        returns_5d : np.ndarray
            shape (5, T) daily returns matrix.
        llm_macro, llm_sentiment, llm_risk : float
            LLM signals in [0, 100].
        ae_error, tau : float, optional
            AE reconstruction error and threshold for V_DEFENSE shifter.
        theta : float
            PPO-controlled exponential tilt gain ∈ [0, 2].
            theta=0  → pure ERC (b0 uniform)
            theta=1  → balanced
            theta=2  → aggressive concentration
        """
        return self.event_track.compute(
            returns_5d,
            llm_macro=llm_macro,
            llm_sentiment=llm_sentiment,
            llm_risk=llm_risk,
            ae_error=ae_error,
            tau=tau,
            theta=theta,
        )
