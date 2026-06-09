"""EventTrack V3.1: V3 audit fixes (matrix-normalized scores + AE shifter).

Fixes over V3:
  1. Scale-mismatch trap: V3's hardcoded scores had heterogeneous ranges
     (s_sat ∈ [-4, 4] vs s_gold ∈ [-1, 1]). This let satellite dominate
     gold in exponential tilting regardless of signal direction.

     V3.1 fix: replace additive scores with matrix multiplication
         s = W · f
     where W is a 5x5 sensitivity matrix with rows normalized so each
     asset's score is bounded in [-1, 1].

  2. Gold tragedy: in crisis, V3 produced s_fi > s_cash > s_gold, with
     e^3 : e^2 : e^1 ≈ 20:7:3 — bonds crushed gold. V2's strategic intent
     (gold 45% risk share) was silently overwritten.

     V3.1 fix: matrix W gives s_fi = s_gold = 1 in pure crisis (tied),
     then AE shifter (Fix 2) breaks the tie in gold's favor.

  3. AE gain-multiplier paradox: theta_eff = theta * (1 + 0.5 * bear_pressure)
     amplifies whatever the LLM signals say. In a bull-end + AE-bear
     divergence, this REINFORCES the bull thesis.

     V3.1 fix: AE is a SHIFTER (convex combo with V_DEFENSE), not a
     GAIN. bear_pressure directly rotates the score vector toward
     the defense manifold, regardless of LLM signal direction.
"""
from __future__ import annotations
import logging
import numpy as np

logger = logging.getLogger(__name__)


class EventTrackV31:
    """V3.1: Matrix-normalized scores + AE shifter + b-as-policy.

    Asset order: 0=broad, 1=satellite, 2=fi, 3=safe, 4=cash

    Pipeline:
      1. f = [m, s, r, equity_stress, sat_lead]    (5-D feature)
      2. s = W @ f                                  (matrix-normalized scores)
      3. bear_pressure = sigmoid((E_t - τ) / scale)
      4. s_final = (1 - bear_pressure) * s + bear_pressure * V_DEFENSE
      5. b = b0 * exp(θ * s_final) / Σ              (exponential tilting)
      6. project b onto box-constrained simplex     (b-as-policy)

    V3.1 deliberately drops RB: the b vector already encodes strategic intent
    (LLM × matrix × shifter), and RB inverse-vol would silently re-allocate
    weight toward low-vol fi at gold's expense — undoing the audit fixes.
    """

    IDX_BROAD = 0
    IDX_SATELLITE = 1
    IDX_FI = 2
    IDX_SAFE = 3
    IDX_CASH = 4

    # ── Fix 1: 5×5 sensitivity matrix ──
    # Rows are assets (broad, sat, fi, gold, cash)
    # Cols are features (m, s, r, equity_stress, sat_lead)
    # Each row's range is bounded in [-1, 1] (or close to it)
    W = np.array([
        # m      s      r     eq    sat_lead
        [ 1/3,   1/3,  -1/3,  0.0,  0.0 ],   # broad:  growth (m+s) - r
        [ 1/4,   1/4,  -1/4,  1/4,  0.0 ],   # sat:    growth - r + sat_lead bonus
        [-1/3,  -1/3,   0.0,  1/3,  0.0 ],   # fi:     anti-equity + stress
        [-1/3,   0.0,   2/3,  0.0,  0.0 ],   # gold:   -m/3 + 2r/3 (pure risk-off)
        [-1/2,   0.0,   1/2,  0.0,  0.0 ],   # cash:   -m/2 + r/2 (anti-growth + risk-off)
    ], dtype=float)

    # ── Fix 2: AE shifter (NOT gain) ──
    # bear_pressure → 1 forces scores toward V_DEFENSE
    # Designed to recover V2's b_bear ≈ (0.05, 0.05, 0.40, 0.40, 0.10)
    V_DEFENSE = np.array([-1.0, -1.0, 0.5, 1.0, 0.5], dtype=float)

    # AE sigmoid scale (transition sharpness)
    AE_SIGMOID_SCALE = 0.5

    # ERC reference: uniform allocation
    B0 = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=float)

    # Sensitivity (no longer amplified by AE).
    # Stage 6: 1.0 -> 0.7 — 减 sharp, 让 weekly std 回升, edge std 提升.
    # THETA 控制 b 分布的 "sharpness": 大 -> 极端 (sat/gold 单边), 小 -> 接近 b0 均匀.
    THETA = 0.7

    # Bounds (no BEAR_BOUNDS — V3.1 is fully smooth).
    # Gold upper bound 0.30→0.40 to let gold reach V2's 0.35 defensive share.
    # Fi upper bound 0.60→0.50 to prevent fi crowding gold via inverse-vol.
    # Cash upper bound 0.15→0.20 to allow V2's 0.10 cash share headroom.
    BOUNDS = [
        (0.05, 0.50),
        (0.00, 0.45),
        (0.00, 0.50),
        (0.00, 0.40),
        (0.00, 0.20),
    ]

    SOLVER_FTOL = 1e-9
    SOLVER_MAXITER = 500
    MIN_SAMPLES = 2
    # Projection tolerance for box-constrained simplex
    PROJ_TOL = 1e-9
    PROJ_MAXITER = 50

    def compute(
        self,
        returns_5d: np.ndarray,
        llm_macro: float = 50.0,
        llm_sentiment: float = 50.0,
        llm_risk: float = 50.0,
        ae_error: float | None = None,
        tau: float | None = None,
    ) -> np.ndarray:
        sigmas = self._safe_sigmas(returns_5d)

        # LLM signals in [-1, 1]
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
        ) / 2.0
        sat_lead = float(
            np.clip(
                (sigmas[self.IDX_SATELLITE] - sigmas[self.IDX_BROAD])
                / (sigmas[self.IDX_SATELLITE] + sigmas[self.IDX_BROAD] + 1e-9),
                -1.0, 1.0,
            )
        )

        # Feature vector
        f = np.array([m, s, r, equity_stress, sat_lead], dtype=float)

        # ── Fix 1: matrix-normalized scores ──
        scores = self.W @ f

        # ── Fix 2: AE soft injection as SHIFTER (not gain) ──
        if ae_error is not None and tau is not None and tau > 0:
            bear_pressure = self._sigmoid(
                (ae_error - tau) / max(tau * self.AE_SIGMOID_SCALE, 1e-9)
            )
        else:
            bear_pressure = 0.0

        if bear_pressure > 0:
            scores = (1.0 - bear_pressure) * scores + bear_pressure * self.V_DEFENSE

        # Exponential tilting
        b = self.B0 * np.exp(self.THETA * scores)
        b_sum = b.sum()
        if b_sum <= 0 or not np.isfinite(b_sum):
            b = self.B0.copy()
        else:
            b = b / b_sum

        # ── V3.1: b-as-weights with bounds enforcement ──
        # V3.1's exp-tilting + V_DEFENSE shifter already encodes the strategic
        # intent (LLM signals + AE regime). RB inverse-vol was overriding this
        # intent by pulling weight back toward low-vol assets in normal regime.
        # The cleanest fix: treat the b vector itself as the policy and just
        # enforce bounds + sum-to-one. This keeps the design coherent:
        # matrix scores → b → w (no silent RB re-balancing).
        n_samples = returns_5d.shape[1]
        if n_samples < self.MIN_SAMPLES:
            return self._normalize(b)

        w = self._project_to_simplex(b, self.BOUNDS)
        return w

    @staticmethod
    def _project_to_simplex(
        b: np.ndarray,
        bounds: list[tuple[float, float]],
    ) -> np.ndarray:
        """Project b onto the box-constrained simplex (Σw=1, lo≤w≤hi).

        Uses iterative clipping: clip to bounds, redistribute excess/deficit
        to non-saturated coordinates, repeat until convergence.
        """
        w = np.clip(b, 0.0, None).astype(float)
        for _ in range(50):
            for i, (lo, hi) in enumerate(bounds):
                w[i] = min(max(w[i], lo), hi)
            s = w.sum()
            if s <= 0:
                w = np.array([0.2] * 5)
                continue
            w = w / s
            if abs(w.sum() - 1.0) < 1e-9 and all(
                lo - 1e-9 <= w[i] <= hi + 1e-9 for i, (lo, hi) in enumerate(bounds)
            ):
                break
        return w

    @staticmethod
    def _sigmoid(x: float) -> float:
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
