"""EventTrack V3: Asset-level Exponential Tilting (full simplex freedom).

Replaces V2's convex-hull interpolation (3 vertices → 2D triangle) with
per-asset score exponentials that span the full 4-simplex Δ⁴.

Key design refinements over V2:
  1. 5-D score vector (one per asset) → full Δ⁴ coverage; stagflation,
     fiat-debasement, and other non-consensus scenarios are reachable.
  2. Exponential tilting = softmax with ERC reference b0 = [0.2]^5
     (mathematically natural and bounds-respecting).
  3. AE soft injection via sigmoid(α·(E_t - τ)): no hard switch, C^∞
     smooth across regime boundary; allows PPO's α meta-fusion to do
     the regime gating instead.

Inheritance from V2:
  - Polynomial RB objective (no 1/w'Σw torsion)
  - Asymmetric w0 = [0.22, 0.18, 0.25, 0.20, 0.15]
  - LedoitWolf covariance + scale normalization
  - Bear bounds (when regime is detected)
"""
from __future__ import annotations
import logging
import numpy as np
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class EventTrackV3:
    """Asset-level Exponential Tilting EventTrack.

    Pipeline:
      1. m, s, r ← LLM signals in [-1, 1]
      2. equity_stress, sat_lead ← market σ features
      3. per-asset scores s_i = linear combo of (m, s, r, eq_stress, sat_lead)
      4. bear_pressure ← sigmoid((E_t - τ) / σ)  [AE soft injection]
      5. θ_eff = THETA * (1 + 0.5 * bear_pressure)
      6. b = b0 * exp(θ_eff * scores) / Σ  [exponential tilting]
      7. Solve RB objective (polynomial form) with SLSQP
    """

    IDX_BROAD = 0
    IDX_SATELLITE = 1
    IDX_FI = 2
    IDX_SAFE = 3
    IDX_CASH = 4

    # ERC reference: uniform allocation (b0 = 1/N)
    B0 = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=float)

    # Sensitivity: 1/τ in softmax language. θ=1.0 means a 1-unit score
    # difference gives e¹ ≈ 2.7x relative weight.
    THETA = 1.0

    # AE soft injection: sigmoid scale; larger → sharper transition
    AE_SIGMOID_SCALE = 0.5  # ΔE = 2 above τ → bear_pressure ≈ 0.73
    AE_THETA_BOOST = 0.5    # bear regime increases θ by 50%

    # Asymmetric initial guess (avoid ERC zero-gradient deadlock)
    W0 = np.array([0.22, 0.18, 0.25, 0.20, 0.15], dtype=float)

    # Bounds (same as V2)
    BOUNDS = [
        (0.05, 0.50),  # broad
        (0.00, 0.45),  # satellite
        (0.00, 0.60),  # fi
        (0.00, 0.30),  # safe
        (0.00, 0.15),  # cash
    ]

    # V3 故意不用 BEAR_BOUNDS — 这是 V3 区别于 V2 的核心设计:
    # 防御配置应该由 b (sigmoid-smooth) 驱动,不是 bounds 硬开关
    # 如此实现端到端 C^∞ 平滑,无 hard switch

    SOLVER_FTOL = 1e-9
    SOLVER_MAXITER = 500
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
        """Return 5-asset EventTrack weights via Exponential Tilting + RB."""
        sigmas = self._safe_sigmas(returns_5d)

        # LLM signals → [-1, 1]
        m = float(np.clip((llm_macro - 50.0) / 50.0, -1.0, 1.0))
        s = float(np.clip((llm_sentiment - 50.0) / 50.0, -1.0, 1.0))
        r = float(np.clip((llm_risk - 50.0) / 50.0, -1.0, 1.0))

        # Market structure features
        equity_stress = float(
            np.clip(
                (sigmas[self.IDX_BROAD] + sigmas[self.IDX_SATELLITE])
                / (sigmas[self.IDX_FI] + sigmas[self.IDX_SAFE] + 1e-9) - 1.0,
                0.0, 2.0,
            )
        ) / 2.0  # [0, 1]
        sat_lead = float(
            np.clip(
                (sigmas[self.IDX_SATELLITE] - sigmas[self.IDX_BROAD])
                / (sigmas[self.IDX_SATELLITE] + sigmas[self.IDX_BROAD] + 1e-9),
                -1.0, 1.0,
            )
        )

        # Per-asset scores (5-D freedom)
        scores = self._per_asset_scores(m, s, r, equity_stress, sat_lead)

        # AE soft injection: C^∞ smooth, no hard switch
        if ae_error is not None and tau is not None and tau > 0:
            bear_pressure = self._sigmoid(
                (ae_error - tau) / max(tau * self.AE_SIGMOID_SCALE, 1e-9)
            )
        else:
            bear_pressure = 0.0

        theta_eff = self.THETA * (1.0 + self.AE_THETA_BOOST * bear_pressure)

        # Exponential tilting
        b = self.B0 * np.exp(theta_eff * scores)
        b_sum = b.sum()
        if b_sum <= 0 or not np.isfinite(b_sum):
            b = self.B0.copy()
        else:
            b = b / b_sum

        # Sample-size fallback
        n_samples = returns_5d.shape[1]
        if n_samples < self.MIN_SAMPLES:
            w = b.copy()
            return self._normalize(w)

        # LedoitWolf + scale normalization
        cov_estimator = LedoitWolf()
        cov_estimator.fit(returns_5d.T)
        Sigma = cov_estimator.covariance_
        sigma_scale = float(np.mean(np.diag(Sigma)))
        if sigma_scale > 0:
            Sigma = Sigma / sigma_scale

        # 关键设计: 不使用 BEAR_BOUNDS,统一用 BOUNDS
        # 防御配置完全由 b(sigmoid 平滑)驱动,RB 在统一 bounds 下求解
        bounds_list = self.BOUNDS

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

        if result.success:
            w = result.x
        else:
            logger.warning(f"[EventTrackV3] SLSQP 失败: {result.message},回退 b-as-w")
            w = b.copy()

        return self._normalize(w)

    def _per_asset_scores(
        self,
        m: float,
        s: float,
        r: float,
        equity_stress: float,
        sat_lead: float,
    ) -> np.ndarray:
        """5-D score vector, one per asset.

        Default coefficients (illustrative, can be tuned):
          s_broad = m + s - r           (growth - risk)
          s_sat   = m + s - r + sat_lead  (growth + sat bonus)
          s_fi    = -m - s + equity_stress  (defensive bond)
          s_gold  = r                    (pure risk-off)
          s_cash  = -m + r               (anti-growth + risk-off)
        """
        return np.array([
            m + s - r,                    # broad
            m + s - r + sat_lead,         # satellite
            -m - s + equity_stress,       # fi
            r,                            # gold
            -m + r,                       # cash
        ], dtype=float)

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Numerically stable sigmoid."""
        if x >= 0:
            return 1.0 / (1.0 + np.exp(-x))
        ex = np.exp(x)
        return ex / (1.0 + ex)

    @staticmethod
    def _normalize(w: np.ndarray) -> np.ndarray:
        w = np.clip(w, 0.0, 1.0)
        s = w.sum()
        if s <= 0 or not np.isfinite(s):
            return np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        return w / s

    def _safe_sigmas(self, returns_5d: np.ndarray) -> np.ndarray:
        sigmas = np.std(returns_5d, axis=1, ddof=1).astype(float)
        return np.where(~np.isfinite(sigmas) | (sigmas <= 0), 1e-3, sigmas)
