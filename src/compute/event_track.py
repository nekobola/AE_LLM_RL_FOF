import logging
import numpy as np

logger = logging.getLogger(__name__)


class EventTrack:
    """Three-prototype event track driven by LLM risk and market stress."""

    IDX_BROAD = 0
    IDX_SATELLITE = 1
    IDX_FI = 2
    IDX_SAFE = 3
    IDX_CASH = 4
    DEFAULT_EPSILON = 0.001

    def __init__(self, epsilon: float | None = None):
        self.EPSILON = epsilon if epsilon is not None else self.DEFAULT_EPSILON

        self.prototype_crisis = np.array([0.05, 0.00, 0.45, 0.35, 0.15], dtype=float)
        self.prototype_reflation = np.array([0.18, 0.10, 0.18, 0.34, 0.20], dtype=float)
        self.prototype_growth = np.array([0.32, 0.48, 0.08, 0.05, 0.07], dtype=float)
        self.base_neutral = np.array([0.18, 0.12, 0.28, 0.22, 0.20], dtype=float)

    def compute(
        self,
        returns_5d: np.ndarray,
        llm_macro: float = 50.0,
        llm_sentiment: float = 50.0,
        llm_risk: float = 50.0,
    ) -> np.ndarray:
        """
        Return a 5-asset event allocation by mixing three event prototypes.

        Prototypes
        ----------
        crisis:
            Fast risk-off, bond/gold/cash dominant.
        reflation:
            Inflation / commodity / policy shock, gold and cash elevated.
        growth:
            Theme / policy-driven upside event, satellite equity elevated.
        """
        sigmas = self._safe_sigmas(returns_5d)
        broad_vol = sigmas[self.IDX_BROAD]
        sat_vol = sigmas[self.IDX_SATELLITE]
        fi_vol = sigmas[self.IDX_FI]
        gold_vol = sigmas[self.IDX_SAFE]

        macro = float(np.clip((llm_macro - 50.0) / 50.0, -1.0, 1.0))
        sentiment = float(np.clip((llm_sentiment - 50.0) / 50.0, -1.0, 1.0))
        risk = float(np.clip((llm_risk - 50.0) / 50.0, -1.0, 1.0))

        equity_stress = float(np.clip((broad_vol + sat_vol) / (fi_vol + gold_vol + 1e-9) - 1.0, 0.0, 2.0))
        equity_stress /= 2.0
        sat_lead = float(np.clip((sat_vol - broad_vol) / (sat_vol + broad_vol + 1e-9), -1.0, 1.0))

        crisis_score = (
            1.35 * max(risk, 0.0)
            + 0.55 * max(-macro, 0.0)
            + 0.45 * max(-sentiment, 0.0)
            + 0.65 * equity_stress
        )
        reflation_score = (
            0.85 * max(risk, 0.0)
            + 0.55 * max(macro, 0.0)
            + 0.20 * max(-sentiment, 0.0)
            + 0.25 * max(gold_vol - fi_vol, 0.0) / (gold_vol + fi_vol + 1e-9)
        )
        growth_score = (
            1.25 * max(macro, 0.0)
            + 1.20 * max(sentiment, 0.0)
            + 0.60 * max(-risk, 0.0)
            + 0.45 * max(sat_lead, 0.0)
        )

        raw_scores = np.array([crisis_score, reflation_score, growth_score], dtype=float)
        proto_weights = self._softmax(raw_scores)

        w_proto = (
            proto_weights[0] * self.prototype_crisis
            + proto_weights[1] * self.prototype_reflation
            + proto_weights[2] * self.prototype_growth
        )

        growth_impulse = 0.45 * max(macro, 0.0) + 0.45 * max(sentiment, 0.0) + 0.20 * max(-risk, 0.0)
        event_intensity = float(
            np.clip(
                0.50 * max(abs(risk), 0.0)
                + 0.30 * max(abs(macro), 0.0)
                + 0.25 * max(abs(sentiment), 0.0)
                + 0.25 * equity_stress
                + 0.25 * growth_impulse,
                0.0,
                1.0,
            )
        )

        w_event = (1.0 - event_intensity) * self.base_neutral + event_intensity * w_proto
        w_event = np.clip(w_event, 0.0, 1.0)
        w_event /= w_event.sum()

        logger.debug(
            "[EventTrack] macro=%.1f sent=%.1f risk=%.1f intensity=%.2f proto=%s weights=%s",
            llm_macro,
            llm_sentiment,
            llm_risk,
            event_intensity,
            np.round(proto_weights, 3).tolist(),
            np.round(w_event, 3).tolist(),
        )

        return w_event

    def _safe_sigmas(self, returns_5d: np.ndarray) -> np.ndarray:
        sigmas = np.std(returns_5d, axis=1, ddof=1).astype(float)
        sigmas = np.where(~np.isfinite(sigmas) | (sigmas <= 0), self.EPSILON, sigmas)
        return sigmas

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        centered = values - np.max(values)
        exp_values = np.exp(centered)
        return exp_values / (exp_values.sum() + 1e-9)
