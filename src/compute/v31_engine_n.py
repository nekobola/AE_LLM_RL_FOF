"""V31 Engine 8-asset: thin wrapper around EventTrackV31N.

Stage 7c: 8-ETF pipeline with AsymmetricSelector (LLM scores → top 8 ETFs)
fed into EventTrackV31N (matrix scores + AE shifter + b-as-Policy).

Asset order matches EventTrackV31N:
  0: 511010 (国债 ETF)
  1: 518880 (黄金 ETF)
  2: 511020 (信用债 ETF)
  3: 159985 (商品 ETF)
  4: 512100 (中证1000 ETF)
  5: 515050 (红利低波 ETF)
  6: 159919 (中证1000 老 ETF)
  7: 510300 (沪深300 ETF)
"""
from __future__ import annotations
import logging
import numpy as np

from .event_track_v3_1_n import EventTrackV31N

logger = logging.getLogger(__name__)


class V31EngineN:
    """8-asset single-track engine. PPO controls theta.

    Returns 8-dim weights per call (sum=1).
    """

    ASSET_CODES = [
        "511010",  # 国债 ETF
        "518880",  # 黄金 ETF
        "511020",  # 信用债 ETF
        "159985",  # 商品 ETF
        "512100",  # 中证1000 ETF
        "515050",  # 红利低波 ETF
        "159915",  # 创业板 ETF (Phase 2 替换 159919, 不同行业 + 更高 Sharpe)
        "510300",  # 沪深300 ETF
    ]

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.event_track = EventTrackV31N()

    def compute(
        self,
        returns_5d: np.ndarray,
        llm_macro: float = 50.0,
        llm_sentiment: float = 50.0,
        llm_risk: float = 50.0,
        ae_error: float | None = None,
        tau: float | None = None,
        theta: float = 0.7,
        b0: np.ndarray | None = None,
        w_matrix: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute 8-ETF portfolio weights with PPO-supplied theta,
        optional data-driven B0 and W matrix (Phase 2 hybrid).
        """
        return self.event_track.compute(
            returns_5d,
            llm_macro=llm_macro,
            llm_sentiment=llm_sentiment,
            llm_risk=llm_risk,
            ae_error=ae_error,
            tau=tau,
            theta=theta,
            b0=b0,
            w_matrix=w_matrix,
        )

    def compute_b0(
        self,
        returns_window: np.ndarray,
        use_erc: bool = True,
    ) -> np.ndarray:
        """Compute pure data-driven B0 from rolling 8-ETF weekly returns.

        No shrinkage, no hand-tuned prior. B0 is 100% derived from the
        rolling covariance of 8 ETF weekly returns. No look-ahead bias.
        """
        return self.event_track.compute_b0_from_returns(
            returns_window,
            bounds=self.event_track.BOUNDS,
            use_erc=use_erc,
        )

    def compute_w_hybrid(
        self,
        features_5d_window: np.ndarray,
        etf_returns_window: np.ndarray,
        max_abs: float = 1.0,
    ) -> np.ndarray | None:
        """Compute Phase 2 hybrid W matrix: W_hybrid = sign(W_SIGN) * |D_scale|.

        Steps:
          1. D_scale = OLS(etf_returns_8d, features_5d)  shape (8, 5)
          2. W_hybrid = sign(W_SIGN) * |D_scale|  (preserve structural sign)
          3. Clip to [-max_abs, max_abs]

        Returns None if data insufficient; caller falls back to class W.
        """
        D_scale = self.event_track.compute_d_scale_from_features(
            features_5d_window, etf_returns_window
        )
        if D_scale is None:
            return None
        return self.event_track.build_w_hybrid(D_scale, max_abs=max_abs)
