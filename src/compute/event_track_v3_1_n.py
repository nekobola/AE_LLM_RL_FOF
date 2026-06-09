"""EventTrack V3.1 N: 8-ETF version for dynamic selection pipeline.

8-asset variant of V3.1, fed by AsymmetricSelector (LLM scores → top 8 ETFs).
Same 3 audit fixes as V3.1 (5-asset):
  1. Matrix-normalized scores
  2. AE shifter (NOT gain)
  3. b-as-policy (no RB inverse-vol)

8-asset selection (5y WFO tuned 2026-06-07):
  0: 511010 (国债 ETF)    — Sharpe 1.85, Calmar 2.23
  1: 518880 (黄金 ETF)    — Sharpe 1.60, Calmar 2.01
  2: 511020 (信用债 ETF)  — Sharpe 0.77
  3: 159985 (商品 ETF)    — Sharpe 0.47
  4: 512100 (中证1000 ETF) — Sharpe 0.31, high beta
  5: 515050 (红利低波 ETF) — Sharpe 0.40, defensive
  6: 159919 (中证1000 老 ETF) — Sharpe -0.08 (fallback)
  7: 510300 (沪深300 ETF) — Sharpe -0.13 (fallback)
"""
from __future__ import annotations
import logging
import numpy as np

logger = logging.getLogger(__name__)


class EventTrackV31N:
    """V3.1 8-asset: matrix scores + AE shifter + b-as-Policy box projection.

    5-dim features f = [m, s, r, equity_stress, sat_lead]
    8-dim scores s = W @ f     (W is 8x5, each row bounded in [-1, 1])
    """

    IDX_BOND_RATES = 0     # 511010
    IDX_GOLD = 1           # 518880
    IDX_BOND_CREDIT = 2    # 511020
    IDX_COMMODITY = 3      # 159985
    IDX_CSI1000 = 4        # 512100
    IDX_DIVLOWVOL = 5      # 515050
    IDX_GEM = 6            # 159915 创业板 (Phase 2 替换 159919)
    IDX_CSI300 = 7         # 510300

    N_ASSETS = 8

    # ── 8×5 sensitivity matrix ──────────────────────────────────────────
    # Rows are 8 ETFs. Cols are features (m, s, r, equity_stress, sat_lead).
    # Each row's range is bounded in [-1, 1].
    #
    # W: legacy hand-tuned matrix (kept for backward compat and emergency fallback).
    # W_SIGN: structural sign-only matrix (Phase 2). When w_matrix parameter is
    #         provided, it's hybrid:  W_hybrid = sign(W_SIGN) ⊙ |D_scale|,
    #         where D_scale comes from rolling OLS of 8 ETF returns on 5-dim
    #         market features. Engineering: structural sign + data magnitude.
    W = np.array([
        # m       s       r       eq      sat_lead
        [-1/3,  -1/3,    0.0,    1/3,    0.0],   # 0  bond_rates:  anti-equity + stress
        [-1/3,   0.0,    2/3,    0.0,    0.0],   # 1  gold:        -m/3 + 2r/3
        [-1/3,  -1/3,    0.0,    1/3,    0.0],   # 2  bond_credit: anti-equity + stress
        [-1/3,   0.0,    1/3,    0.0,    0.0],   # 3  commodity:   anti-m/3 + r/3
        [ 1/3,   1/3,   -1/3,    0.0,    0.0],   # 4  csi1000:     growth (m+s) - r
        [ 1/4,   1/4,   -1/4,    1/4,    0.0],   # 5  div_lowvol:  growth - r + sat_lead (mild)
        [ 1/3,   1/3,   -1/3,    0.0,    0.0],   # 6  gem (159915): aggressive growth (科技/医药/新能源)
        [ 0.0,   0.0,   -1/3,    0.0,    0.0],   # 7  csi300:      only anti-risk, no growth  (5y -10.9%)
    ], dtype=float)

    # ── W_SIGN: structural sign matrix (Phase 2, no magnitude tuning) ────
    # Each entry is in {-1, 0, +1}, derived from financial theory:
    #   +1: "asset i tends to rise when feature j is up"
    #   -1: "asset i tends to fall when feature j is up"
    #    0: "no clear structural relationship"
    # Magnitude comes from data (D_scale, rolling OLS).
    W_SIGN = np.array([
        # m     s     r     eq    sat_lead
        [-1,   -1,    0,   +1,    0],   # 0  bond_rates:  flight-to-quality
        [-1,    0,   +1,    0,    0],   # 1  gold:         risk hedge
        [-1,   -1,    0,   +1,    0],   # 2  bond_credit:  similar to bond_rates
        [-1,    0,   +1,    0,    0],   # 3  commodity:    inflation/risk hedge
        [+1,   +1,   -1,    0,    0],   # 4  csi1000:      high-beta growth
        [+1,   +1,   -1,   +1,    0],   # 5  div_lowvol:   growth + sat_lead (defensive satellite)
        [+1,   +1,   -1,    0,    0],   # 6  gem:          创业板 (科技/医药/新能源, growth)
        [ 0,    0,   -1,    0,    0],   # 7  csi300:       anti-risk only (broad index)
    ], dtype=float)

    # ── AE shifter (NOT gain) ────────────────────────────────────────────
    # bear_pressure → 1 forces scores toward V_DEFENSE
    # V_DEFENSE design: 100% defensive (bond 50% + gold 35% + commodity 15%)
    V_DEFENSE = np.array([
        0.5,   # 0  bond_rates (high in crisis)
        1.0,   # 1  gold (highest)
        0.5,   # 2  bond_credit
        0.0,   # 3  commodity (neutral in crisis)
        -1.0,  # 4  csi1000 (no equity in crisis)
        -1.0,  # 5  div_lowvol
        -1.0,  # 6  csi1000_old
        -1.0,  # 7  csi300
    ], dtype=float)

    AE_SIGMOID_SCALE = 0.5

    # ERC reference: bias toward defensive in 8-asset (gold+FI dominate)
    B0 = np.array([0.20, 0.20, 0.15, 0.10, 0.10, 0.10, 0.075, 0.075], dtype=float)

    # Bounds (smooth, no regime hard-switch)
    #   bonds: high max (40-45% lets defensive auto-weight)
    #   gold: 40% max (V3.1 design)
    #   equity ETFs: 20% cap each to prevent overconcentration in bull
    BOUNDS = [
        (0.00, 0.45),  # 0  bond_rates
        (0.00, 0.40),  # 1  gold
        (0.00, 0.30),  # 2  bond_credit
        (0.00, 0.20),  # 3  commodity
        (0.00, 0.25),  # 4  csi1000
        (0.00, 0.20),  # 5  div_lowvol
        (0.00, 0.20),  # 6  csi1000_old
        (0.00, 0.20),  # 7  csi300
    ]

    SOLVER_FTOL = 1e-9
    SOLVER_MAXITER = 500
    MIN_SAMPLES = 2
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
        theta: float = 0.7,
        b0: np.ndarray | None = None,
        w_matrix: np.ndarray | None = None,
    ) -> np.ndarray:
        """8-asset V3.1.compute. Returns 8-dim weights summing to 1.

        returns_5d : np.ndarray, shape (5, T) — only 5 features are used
                     (m, s, r, equity_stress, sat_lead) regardless of asset count
        b0 : np.ndarray, shape (8,), optional
            Data-driven base weights (e.g., inverse-vol from rolling 104w
            covariance of 8 ETF weekly returns). If None, falls back to
            the hand-tuned B0.
        w_matrix : np.ndarray, shape (8, 5), optional
            Hybrid sensitivity matrix (Phase 2). If provided, used instead
            of the class W default. W_hybrid = sign(W_SIGN) * |D_scale| where
            D_scale is data-driven from rolling OLS.
        """
        sigmas = self._safe_sigmas(returns_5d)

        m = float(np.clip((llm_macro - 50.0) / 50.0, -1.0, 1.0))
        s = float(np.clip((llm_sentiment - 50.0) / 50.0, -1.0, 1.0))
        r = float(np.clip((llm_risk - 50.0) / 50.0, -1.0, 1.0))

        # 5-dim features (same as V3.1 5-asset, not 8)
        equity_stress = float(
            np.clip(
                (sigmas[0] + sigmas[1])
                / (sigmas[2] + sigmas[3] + 1e-9) - 1.0,
                0.0, 2.0,
            )
        ) / 2.0
        sat_lead = float(
            np.clip(
                (sigmas[1] - sigmas[0])
                / (sigmas[1] + sigmas[0] + 1e-9),
                -1.0, 1.0,
            )
        )
        f = np.array([m, s, r, equity_stress, sat_lead], dtype=float)

        # Fix 1: matrix-normalized scores (8 assets)
        # Use the passed-in w_matrix if provided (Phase 2 hybrid), else class W
        if w_matrix is not None and np.asarray(w_matrix).shape == (8, 5):
            W_use = np.asarray(w_matrix, dtype=float)
        else:
            W_use = self.W
        scores = W_use @ f

        # Fix 2: AE soft injection as SHIFTER
        if ae_error is not None and tau is not None and tau > 0:
            bear_pressure = self._sigmoid(
                (ae_error - tau) / max(tau * self.AE_SIGMOID_SCALE, 1e-9)
            )
        else:
            bear_pressure = 0.0

        if bear_pressure > 0:
            scores = (1.0 - bear_pressure) * scores + bear_pressure * self.V_DEFENSE

        # Choose base weights: data-driven if provided, else hand-tuned
        b0_use = self.B0 if b0 is None else np.asarray(b0, dtype=float)
        if b0_use.shape != (8,):
            b0_use = self.B0
        if not np.isfinite(b0_use).all() or (b0_use <= 0).any():
            b0_use = self.B0

        # Exponential tilting
        b = b0_use * np.exp(theta * scores)
        b_sum = b.sum()
        if b_sum <= 0 or not np.isfinite(b_sum):
            b = b0_use.copy()
        else:
            b = b / b_sum

        n_samples = returns_5d.shape[1] if returns_5d.ndim > 1 else 0
        if n_samples < self.MIN_SAMPLES:
            return self._normalize(b)

        w = self._project_to_simplex(b, self.BOUNDS)
        return w

    @staticmethod
    def compute_b0_from_returns(
        returns_window: np.ndarray,
        bounds: list[tuple[float, float]] | None = None,
        use_erc: bool = True,
    ) -> np.ndarray:
        """Compute data-driven B0 from rolling weekly returns.

        No shrinkage, no hand-tuned prior. B0 is 100% derived from the
        rolling covariance of 8 ETF weekly returns. No look-ahead bias.

        Two modes:
          1. ERC (Equal Risk Contribution, Spinu 2013):
             Solve  min_w sum_i (log w_i - log (Σ w)_i)^2  s.t. w > 0, sum w = 1
             "Each asset contributes equal risk to portfolio".
          2. Inverse-volatility (simpler):  w_i = (1/σ_i) / sum_j (1/σ_j)

        Parameters
        ----------
        returns_window : np.ndarray, shape (T, 8) or (8, T)
            T weeks of weekly returns for the 8 ETFs (T >= 26 recommended).
        bounds : list of (lo, hi), optional
            Box constraints on B0. If None, no bounds.
            If provided, project onto box-constrained simplex.
        use_erc : bool
            If True, use ERC; if False, use inverse-vol.

        Returns
        -------
        np.ndarray, shape (8,)
            Base weights summing to 1.
        """
        # Normalize shape to (8, T)
        r = np.asarray(returns_window, dtype=float)
        if r.ndim != 2:
            raise ValueError(f"returns_window must be 2D, got {r.ndim}D")
        if r.shape[0] == 8:
            r = r  # already (8, T)
        elif r.shape[1] == 8:
            r = r.T  # transpose (T, 8) -> (8, T)
        else:
            raise ValueError(f"returns_window shape {r.shape} not 8-compatible")

        T = r.shape[1]
        if T < 26:
            # Too few samples, return equal weight
            return np.array([0.125] * 8)

        # Ledoit-Wolf style shrinkage to diagonal (simple version)
        # Sample covariance
        sample_cov = np.cov(r, ddof=1)  # (8, 8)
        if not np.isfinite(sample_cov).all():
            return np.array([0.125] * 8)

        # Shrinkage intensity: ρ = 0.5 (heuristic, can be tuned)
        diag_mean = np.trace(sample_cov) / 8
        shrunk_cov = 0.5 * sample_cov + 0.5 * diag_mean * np.eye(8)

        if not use_erc:
            # Simple inverse-volatility
            sigmas = np.sqrt(np.diag(shrunk_cov))
            sigmas = np.maximum(sigmas, 1e-3)
            inv_vol = 1.0 / sigmas
            b0 = inv_vol / inv_vol.sum()
        else:
            # ERC via Spinu (2013) iterative scheme
            # Start with inverse-vol
            sigmas = np.sqrt(np.diag(shrunk_cov))
            sigmas = np.maximum(sigmas, 1e-3)
            w = 1.0 / sigmas
            w = w / w.sum()

            # Iterate:  w_i <- (w_i * (Σ w)_i) ^ 0.5 / sum, but Spinu's form
            # Use simple fixed-point: w_i^{(k+1)} = exp(mean(log(Σw) - log w)) pattern
            for _ in range(100):
                risk_contrib = shrunk_cov @ w  # (8,)
                rc_i = w * risk_contrib  # marginal risk = w_i * (Σw)_i
                # Target: equal RC across assets
                target_rc = rc_i.mean()
                # Adjust weights toward target
                w_new = w * (target_rc / np.maximum(rc_i, 1e-9)) ** 0.5
                w_new = w_new / w_new.sum()
                if np.abs(w_new - w).max() < 1e-6:
                    break
                w = w_new
            b0 = w

        # Apply bounds if provided
        if bounds is not None:
            b0 = EventTrackV31N._project_to_simplex(b0, bounds)

        return b0

    @staticmethod
    def _project_to_simplex(
        b: np.ndarray,
        bounds: list[tuple[float, float]],
    ) -> np.ndarray:
        """Project b onto the box-constrained simplex."""
        w = np.clip(b, 0.0, None).astype(float)
        for _ in range(50):
            for i, (lo, hi) in enumerate(bounds):
                w[i] = min(max(w[i], lo), hi)
            s = w.sum()
            if s <= 0:
                w = np.array([0.125] * 8)
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
    def compute_tau_from_ae_errors(
        ae_errors_window: np.ndarray,
        quantile: float = 0.7,
    ) -> float | None:
        """Compute data-driven tau from rolling AE errors (Phase 3).

        tau is set such that `quantile` fraction of historical AE errors
        fall below it (i.e., `(1 - quantile)` fraction of weeks are
        "regime bear"). For quantile=0.7, 30% of weeks are bear regime.

        Parameters
        ----------
        ae_errors_window : np.ndarray, shape (T,)
            T weeks of AE reconstruction errors from burn-in / lookback.
        quantile : float in (0, 1)
            Fraction of weeks to be classified as "regime calm".
            Default 0.7 (30% bear weeks).

        Returns
        -------
        float, or None if data insufficient.
        """
        e = np.asarray(ae_errors_window, dtype=float)
        e = e[np.isfinite(e) & (e > 0)]
        if len(e) < 26:
            return None
        # Use np.percentile (more numerically stable than np.quantile)
        return float(np.percentile(e, quantile * 100))

    @staticmethod
    def compute_d_scale_from_features(
        features_5d_window: np.ndarray,
        etf_returns_window: np.ndarray,
    ) -> np.ndarray | None:
        """Compute data-driven D_scale (8, 5) via rolling OLS.

        Regress 8 ETF weekly returns on 5-dim market features:
          Y (T, 8) = 8 ETF returns
          X (T, 5) = market features [m, s, r, equity_stress, sat_lead]
          β (5, 8) = (X'X)^-1 X'Y
          D_scale (8, 5) = β.T

        Each D_scale[i, j] measures: "when market feature j moves by 1 std,
        ETF i's weekly return moves by D_scale[i, j]".

        Parameters
        ----------
        features_5d_window : np.ndarray, shape (T, 5)
            T weeks of 5-dim market features. Features should be roughly
            in [-1, 1] range (LLM-normalized, vol ratios).
        etf_returns_window : np.ndarray, shape (T, 8)
            T weeks of 8 ETF weekly returns.

        Returns
        -------
        np.ndarray, shape (8, 5), or None if insufficient data / singular.
        """
        X = np.asarray(features_5d_window, dtype=float)
        Y = np.asarray(etf_returns_window, dtype=float)
        if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
            return None
        if X.shape[0] < 26 or X.shape[1] != 5 or Y.shape[1] != 8:
            return None
        if not np.isfinite(X).all() or not np.isfinite(Y).all():
            return None

        # Standardize X (each column) for numerical stability.
        # Don't center Y (we want raw beta magnitudes).
        X_mean = X.mean(axis=0, keepdims=True)
        X_std = X.std(axis=0, keepdims=True)
        X_std = np.where(X_std < 1e-9, 1.0, X_std)
        X_norm = (X - X_mean) / X_std

        # OLS: β = (X'X)^-1 X'Y, shape (5, 8)
        XtX = X_norm.T @ X_norm
        # Add small ridge for stability
        XtX += 1e-4 * np.eye(5)
        try:
            beta = np.linalg.solve(XtX, X_norm.T @ Y)
        except np.linalg.LinAlgError:
            return None
        if not np.isfinite(beta).all():
            return None

        D_scale = beta.T  # (8, 5)
        return D_scale

    @staticmethod
    def build_w_hybrid(
        d_scale: np.ndarray,
        max_abs: float = 1.0,
    ) -> np.ndarray:
        """Build W_hybrid = sign(W_SIGN) * |D_scale|, clipped to [-max_abs, max_abs].

        Preserves the structural sign (from W_SIGN, hard-coded) while letting
        the magnitude come from data (D_scale from OLS). Clipping prevents
        score explosion in V3.1N.
        """
        D = np.asarray(d_scale, dtype=float)
        if D.shape != (8, 5):
            return EventTrackV31N.W
        W_hybrid = np.sign(EventTrackV31N.W_SIGN) * np.abs(D)
        # Zero out entries where W_SIGN is 0 (no structural relationship)
        W_hybrid = W_hybrid * (EventTrackV31N.W_SIGN != 0).astype(float)
        # Clip magnitude
        W_hybrid = np.clip(W_hybrid, -max_abs, max_abs)
        return W_hybrid

    @staticmethod
    def _normalize(w: np.ndarray) -> np.ndarray:
        w = np.clip(w, 0.0, 1.0)
        s = w.sum()
        if s <= 0 or not np.isfinite(s):
            return np.array([0.125] * 8)
        return w / s

    def _safe_sigmas(self, returns_5d: np.ndarray) -> np.ndarray:
        sigmas = np.std(returns_5d, axis=1, ddof=1).astype(float)
        return np.where(~np.isfinite(sigmas) | (sigmas <= 0), 1e-3, sigmas)
