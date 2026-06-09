"""EventTrack V2: Signal-Tilted Risk Budgeting.

Replaces the three-prototype / softmax-blend design (V1) with a single
Risk Budgeting (RB) objective whose target risk shares are linearly
interpolated across three simplex vertices (B_BEAR, B_NEUTRAL, B_GROWTH)
driven by (LLM signals, AE regime).

Numerical design refinements (vs naive RB):
  1. Objective in polynomial form  L = sum (w_i*(Sw)_i - b_i*(w'Sw))^2
     to avoid 1/port_var torsion near zero variance.
  2. Asymmetric initial guess w0 = [0.21, 0.19, 0.20, 0.18, 0.22]
     to escape the boundary zero-gradient deadlock SLSQP hits at equal weights.
  3. beta_neutral = max(0, 1 - beta_bear - beta_growth) post mutex-scaling
     guarantees the simplex sum holds for any LLM signal input.
"""
from __future__ import annotations
import logging
import numpy as np
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class EventTrackV2:
    """Signal-Tilted Risk Budgeting EventTrack.

    Asset order: 0=broad, 1=satellite, 2=fi, 3=safe, 4=cash
    """

    IDX_BROAD = 0
    IDX_SATELLITE = 1
    IDX_FI = 2
    IDX_SAFE = 3
    IDX_CASH = 4

    # Three risk-budget simplex vertices (sum to 1 each).
    # All entries carry financial meaning; see docs/strategy_details.md §9.3.
    B_BEAR = np.array([0.05, 0.05, 0.45, 0.35, 0.10], dtype=float)
    B_NEUTRAL = np.array([0.20, 0.20, 0.20, 0.20, 0.20], dtype=float)
    B_GROWTH = np.array([0.30, 0.45, 0.10, 0.10, 0.05], dtype=float)

    # Asymmetric initial guess to avoid ERC equal-weight zero-gradient deadlock.
    # Note: cash is intentionally low (0.10) because RB is inverse-vol — a high
    # cash starting point biases the solver into a low-vol corner.
    W0 = np.array([0.22, 0.18, 0.25, 0.20, 0.15], dtype=float)

    # Bounds: more aggressive than NormalTrack to permit signal-tilted extremes.
    # Cash cap is tight (0.15) because RB on low-vol cash otherwise saturates.
    BOUNDS = [
        (0.05, 0.50),  # broad
        (0.00, 0.45),  # satellite
        (0.00, 0.60),  # fi
        (0.00, 0.30),  # safe
        (0.00, 0.15),  # cash
    ]

    # Bear-regime bounds: 强制防御配置
    # (固收下限 0.30, 黄金下限 0.20, 卫星上限 0.10, 宽基上限 0.20)
    BEAR_BOUNDS = [
        (0.00, 0.20),  # broad
        (0.00, 0.10),  # satellite
        (0.30, 0.60),  # fi
        (0.20, 0.40),  # safe
        (0.00, 0.20),  # cash
    ]

    # β total cap: the smaller the cap, the closer w is forced to a vertex.
    BETA_TOTAL_CAP = 0.90

    # Solver tolerance
    SOLVER_FTOL = 1e-9
    SOLVER_MAXITER = 500

    # LedoitWolf needs at least 2 samples (with shrinkage it stabilizes)
    MIN_SAMPLES = 2

    def compute(
        self,
        returns_5d: np.ndarray,
        llm_macro: float = 50.0,
        llm_sentiment: float = 50.0,
        llm_risk: float = 50.0,
        ae_error: float | None = None,
        tau: float | None = None,
    ) -> np.ndarray:
        """Return 5-asset EventTrack weights via Signal-Tilted RB.

        Pipeline:
          1. If ae_error > tau: force BEAR vertex (β_bear=0.85, β_neutral=0.15).
          2. Else: compute β_bear, β_growth from LLM signals (d1=macro, d2=risk, d3=sent).
          3. β_neutral = max(0, 1 - β_bear - β_growth).
          4. Interpolate b = β_bear*B_BEAR + β_neutral*B_NEUTRAL + β_growth*B_GROWTH.
          5. Solve RB objective (polynomial form) with SLSQP, asymmetric w0.
        """
        beta_bear, beta_growth = self._signal_to_betas(
            llm_macro, llm_sentiment, llm_risk, ae_error, tau
        )
        total = beta_bear + beta_growth
        if total > self.BETA_TOTAL_CAP:
            scale = self.BETA_TOTAL_CAP / total
            beta_bear *= scale
            beta_growth *= scale
        beta_neutral = max(0.0, 1.0 - beta_bear - beta_growth)

        b = (
            beta_bear * self.B_BEAR
            + beta_neutral * self.B_NEUTRAL
            + beta_growth * self.B_GROWTH
        )
        b = np.clip(b, 0.0, None)
        b = b / b.sum()

        n_samples = returns_5d.shape[1]
        if n_samples < self.MIN_SAMPLES:
            # 单期无法估计协方差 → 用目标 b 本身作为权重(β-decayed 顶点)
            w = b.copy()
            w = np.clip(w, 0.0, 1.0)
            w = w / w.sum()
            logger.debug(
                f"[EventTrackV2] 样本数 {n_samples} < {self.MIN_SAMPLES},回退 b-as-w"
            )
            return w

        cov_estimator = LedoitWolf()
        cov_estimator.fit(returns_5d.T)
        Sigma = cov_estimator.covariance_
        # Normalize Sigma by mean diagonal to make the RB landscape O(1).
        # This is scale-invariant: the optimum w is unchanged because both
        # (w*Sw) and (w'Σw) scale by the same factor, so (w*Sw - b*p) is invariant.
        sigma_scale = float(np.mean(np.diag(Sigma)))
        if sigma_scale > 0:
            Sigma = Sigma / sigma_scale

        # Bear regime: 用更紧的 bounds 强制防御配置
        # (固收下限 0.30, 卫星上限 0.10, 黄金下限 0.25)
        is_bear = ae_error is not None and tau is not None and ae_error > tau
        bounds_list = self.BEAR_BOUNDS if is_bear else self.BOUNDS

        def objective_risk_budget(w, cov_matrix, b_target):
            Sw = cov_matrix @ w
            port_var = w @ Sw
            return float(np.sum((w * Sw - b_target * port_var) ** 2))

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        result = minimize(
            objective_risk_budget,
            self.W0,
            args=(Sigma, b),
            method="SLSQP",
            bounds=bounds_list,
            constraints=constraints,
            options={"ftol": self.SOLVER_FTOL, "maxiter": self.SOLVER_MAXITER},
        )

        if not result.success:
            logger.warning(
                f"[EventTrackV2] SLSQP 失败: {result.message},回退 w0"
            )
            w = self.W0.copy()
        else:
            w = result.x

        w = np.clip(w, 0.0, 1.0)
        w = w / w.sum()

        logger.debug(
            "[EventTrackV2] β_bear=%.2f β_neutral=%.2f β_growth=%.2f | b=%s | w=%s",
            beta_bear, beta_neutral, beta_growth,
            np.round(b, 3).tolist(),
            np.round(w, 3).tolist(),
        )
        return w

    def _signal_to_betas(
        self,
        llm_macro: float,
        llm_sentiment: float,
        llm_risk: float,
        ae_error: float | None,
        tau: float | None,
    ) -> tuple[float, float]:
        """Map (d1=macro, d2=risk, d3=sent) to (β_bear, β_growth).

        AE hard switch:
          ae_error > tau: β_bear=0.85, β_growth=0.0 (force bear vertex)
        Otherwise:
          β_bear   = 0.50 * max(0, (50-d_macro)/50)   + 0.30 * max(0, (d_risk-70)/30)
          β_growth = 0.45 * max(0, (d_macro-60)/40)   + 0.40 * max(0, (d_sent-65)/35)
        Caller applies BETA_TOTAL_CAP and the β_neutral = max(0, 1 - sum) guard.
        """
        if ae_error is not None and tau is not None and ae_error > tau:
            return 0.85, 0.0

        beta_bear = (
            0.60 * max(0.0, (50.0 - llm_macro) / 50.0)
            + 0.40 * max(0.0, (llm_risk - 70.0) / 30.0)
        )
        beta_growth = (
            0.70 * max(0.0, (llm_macro - 60.0) / 40.0)
            + 0.55 * max(0.0, (llm_sentiment - 65.0) / 35.0)
        )
        return beta_bear, beta_growth
