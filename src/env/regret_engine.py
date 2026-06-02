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

import logging
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Static expert candidate library  W_cand  (5 assets: V1-V5)
# All candidates are pre-normalized to sum=1.0 for fair comparison
# ---------------------------------------------------------------------------
_W_CANDIDATES: list[np.ndarray] = [
    # ---- Absolute specialists (defensive) ----
    np.array([0.0, 0.0, 0.0, 0.0, 1.0]),   # cash
    np.array([0.0, 0.0, 0.0, 1.0, 0.0]),   # gold
    np.array([0.0, 0.0, 1.0, 0.0, 0.0]),   # pure bond
    # Baseline: inverse-vol (placeholder; updated in update_w_cand_inverse_vol)
    np.array([0.25, 0.25, 0.25, 0.125, 0.125]),
    # ---- Grid points: bond-hedge-cash space (defensive) ----
    np.array([0.0, 0.0, 0.6, 0.4, 0.0]),
    np.array([0.0, 0.0, 0.4, 0.6, 0.0]),
    np.array([0.0, 0.0, 0.7, 0.2, 0.1]),
    np.array([0.0, 0.0, 0.5, 0.3, 0.2]),
    np.array([0.0, 0.0, 0.3, 0.5, 0.2]),
    np.array([0.0, 0.0, 0.2, 0.4, 0.4]),
    # ---- Offensive experts (equity-heavy) ----
    np.array([0.6, 0.0, 0.2, 0.1, 0.1]),   # 60/40 CSI300/bond
    np.array([0.5, 0.1, 0.2, 0.1, 0.1]),   # 60/40 with satellite
    np.array([0.7, 0.0, 0.15, 0.1, 0.05]), # 70/30 aggressive
    np.array([1.0, 0.0, 0.0, 0.0, 0.0]),   # 100% CSI300
    np.array([0.0, 1.0, 0.0, 0.0, 0.0]),   # 100% CSI1000
    np.array([0.4, 0.2, 0.2, 0.1, 0.1]),   # 60/40 with satellite split
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

    # Minimum samples required for reliable inverse-vol weight estimation
    MIN_INVERSE_VOL_SAMPLES = 20

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
        self._inverse_vol_valid: bool = False  # whether W_cand[3] has been updated

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
        if returns_5d.shape[0] < self.MIN_INVERSE_VOL_SAMPLES:
            logger.warning(
                f"[RegretEngine] 数据不足{self.MIN_INVERSE_VOL_SAMPLES}天 "
                f"(仅{returns_5d.shape[0]})，逆波动率候选人未更新"
            )
            self._inverse_vol_valid = False
            return

        vol = np.std(returns_5d, ddof=1, axis=0)

        # P0-Fix: NaN/零波动率保护
        if vol.ndim > 0:
            vol = np.asarray(vol, dtype=np.float64).flatten()
            valid_mask = ~(np.isnan(vol) | (vol <= 1e-9))
            if not np.any(valid_mask):
                logger.warning("[RegretEngine] 所有资产波动率为NaN或零，逆波动率候选人未更新")
                self._inverse_vol_valid = False
                return
            inv_vol = np.zeros_like(vol)
            inv_vol[valid_mask] = 1.0 / vol[valid_mask]
            inv_vol[~valid_mask] = 0.0
        else:
            inv_vol = np.array([1.0 / vol]) if vol > 1e-9 else np.array([0.0])

        total = inv_vol.sum()
        if total <= 0:
            logger.warning("[RegretEngine] 逆波动率和为零，候选人未更新")
            self._inverse_vol_valid = False
            return

        weights = inv_vol / total
        # 归一化确保sum=1.0
        self._w_cand[3] = weights / weights.sum()
        self._inverse_vol_valid = True
        logger.debug(f"[RegretEngine] 逆波动率候选人已更新: {self._w_cand[3]}")

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
        # P1-Fix: 维度校验
        w_final_prev = np.asarray(w_final_prev, dtype=np.float64).flatten()
        if w_final_prev.shape[0] != 5:
            raise ValueError(
                f"[RegretEngine] w_final_prev维度错误: expected (5,), got {w_final_prev.shape}"
            )

        # Extract actual period return for the portfolio
        if returns_window.ndim == 2:
            period_return = np.asarray(returns_window[-1], dtype=np.float64).flatten()
        else:
            period_return = np.asarray(returns_window, dtype=np.float64).flatten()

        if period_return.shape[0] != 5:
            raise ValueError(
                f"[RegretEngine] returns_window维度错误: expected (5,) or (N,5), got {returns_window.shape}"
            )

        # P1-Fix: NaN处理 — 含NaN的资产收益替换为0（中性），避免NaN污染
        nan_mask = np.isnan(period_return)
        if np.any(nan_mask):
            logger.warning(f"[RegretEngine] 检测到NaN收益: {nan_mask}，替换为0（中性）")
            period_return = np.where(nan_mask, 0.0, period_return)

        r_actual = float(np.dot(w_final_prev, period_return))

        # Evaluate all 10 experts
        expert_returns = []
        for w in self._w_cand:
            ret = float(np.dot(w, period_return))
            # NaN检测：理论上不应该有，但防御性处理
            if np.isnan(ret):
                ret = 0.0
            expert_returns.append(ret)

        expert_returns = np.array(expert_returns, dtype=np.float64)
        # 使用nanmax避免NaN污染，若全为NaN则r_opt=0
        r_opt = float(np.nanmax(expert_returns)) if not np.all(np.isnan(expert_returns)) else 0.0

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

    @property
    def inverse_vol_valid(self) -> bool:
        """Whether the inverse-vol candidate has been updated with sufficient data."""
        return self._inverse_vol_valid
