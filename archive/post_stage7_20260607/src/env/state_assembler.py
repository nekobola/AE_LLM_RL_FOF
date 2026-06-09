"""State Assembler

Stage 7: 9-dim state for the single-track PPO-theta controller.
Removed alpha_prev and tau_prev (no fusion, no AE threshold control).
Replaced with theta_prev (PPO's previous V3.1 tilt gain).

Dimensions:
  1.  E_t_zscore            — AE reconstruction error, Z-score standardized
  2.  vol_mkt_20d_norm      — Market 20d annualized vol, MinMax → [0, 1]
  3.  llm_macro_norm        — LLM macro顺风度 (d1-50)/50 ∈ [-1, 1]
  4.  llm_sent_norm         — LLM sentiment (d2-50)/50 ∈ [-1, 1]
  5.  llm_risk_norm         — LLM tail risk d3/100 ∈ [0, 1]
  6.  port_sharpe_20d      — Portfolio 20d Sharpe, hard-clipped to [-3, 3]
  7.  port_mdd_current      — Current drawdown ∈ [0, 1]
  8.  regret_ema_norm      — Regret EMA normalized to [0, 1]
  9.  theta_prev_norm       — Previous V3.1 theta, MinMax → [0, 1]
"""
from __future__ import annotations
import logging

import numpy as np
from typing import NamedTuple


class StateTuple(NamedTuple):
    """9-dim state vector with named fields."""
    e_t_zscore: float          # dim 1
    vol_mkt_20d_norm: float     # dim 2
    llm_macro_norm: float       # dim 3
    llm_sent_norm: float        # dim 4
    llm_risk_norm: float        # dim 5
    port_sharpe_20d: float      # dim 6
    port_mdd_current: float     # dim 7
    regret_ema_norm: float      # dim 8
    theta_prev_norm: float      # dim 9 (was tau_prev_norm)


class StateAssembler:
    """Stage 7: Assembles 9-dim state S_t.

    Replaces alpha_prev/tau_prev with theta_prev. The MinMax range for
    theta is [0, 2] (theta_min, theta_max).
    """

    def __init__(
        self,
        sharpe_clip_low: float = -3.0,
        sharpe_clip_high: float = 3.0,
        d3_rolling_window: int = 5,
        d3_zscore_cap: float = 2.5,
        d3_week_change_cap: float = 30.0,
        theta_min: float = 0.0,
        theta_max: float = 2.0,
    ):
        self.sharpe_clip_low = sharpe_clip_low
        self.sharpe_clip_high = sharpe_clip_high
        self.d3_rolling_window = d3_rolling_window
        self.d3_zscore_cap = d3_zscore_cap
        self.d3_week_change_cap = d3_week_change_cap

        self._ae_mean: float = 0.0
        self._ae_std: float = 1.0
        self._vol_min: float = 0.0
        self._vol_max: float = 1.0
        self._theta_min: float = theta_min
        self._theta_max: float = theta_max
        self._d3_history: list[float] = []

    def fit_normalizers(
        self,
        ae_errors: np.ndarray,
        vol_series: np.ndarray,
        theta_series: np.ndarray | None = None,
    ) -> None:
        """Fit normalizer parameters from historical data."""
        if len(ae_errors) > 1:
            self._ae_mean = float(np.mean(ae_errors))
            self._ae_std = float(np.std(ae_errors, ddof=1))
        if len(vol_series) > 1:
            self._vol_min = float(np.min(vol_series))
            self._vol_max = float(np.max(vol_series))
        if theta_series is not None and len(theta_series) > 1:
            self._theta_min = float(np.min(theta_series))
            self._theta_max = float(np.max(theta_series))

    def assemble(
        self,
        ae_error: float,
        vol_mkt_20d: float,
        llm_macro: float,
        llm_sentiment: float,
        llm_risk: float,
        port_sharpe_20d: float,
        port_mdd_current: float,
        regret_ema_norm: float,
        theta_prev: float,
    ) -> np.ndarray:
        """Assemble 9-dim state vector.

        Parameters
        ----------
        ae_error, vol_mkt_20d, llm_*, port_sharpe_20d, port_mdd_current,
        regret_ema_norm, theta_prev :
            see module docstring.
        """
        ae_zscore = (ae_error - self._ae_mean) / (self._ae_std + 1e-9)
        vol_norm = self._minmax_map(vol_mkt_20d, self._vol_min, self._vol_max)
        llm_macro_norm = (llm_macro - 50.0) / 50.0
        llm_sent_norm = (llm_sentiment - 50.0) / 50.0
        llm_risk_clean = self._clean_d3(llm_risk)
        llm_risk_norm = llm_risk_clean / 100.0
        sharpe_clipped = float(np.clip(port_sharpe_20d, self.sharpe_clip_low, self.sharpe_clip_high))
        mdd_current = port_mdd_current
        regret_norm = regret_ema_norm
        theta_prev_norm = self._minmax_map(theta_prev, self._theta_min, self._theta_max)

        return np.array([
            ae_zscore,
            vol_norm,
            llm_macro_norm,
            llm_sent_norm,
            llm_risk_norm,
            sharpe_clipped,
            mdd_current,
            regret_norm,
            theta_prev_norm,
        ], dtype=np.float32)

    def _clean_d3(self, d3_raw: float) -> float:
        d3 = float(np.clip(d3_raw, 0.0, 100.0))
        self._d3_history.append(d3)
        if len(self._d3_history) > self.d3_rolling_window:
            self._d3_history.pop(0)
        if len(self._d3_history) >= 3:
            median = float(np.median(self._d3_history))
            std = float(np.std(self._d3_history, ddof=1)) + 1e-9
            z_score = abs(d3 - median) / std
            if z_score > self.d3_zscore_cap:
                d3 = median
                logger = logging.getLogger(__name__)
                logger.debug(f"[StateAssembler] d3 异常清洗: {d3_raw:.1f} -> {d3:.1f}")
        if len(self._d3_history) >= 2:
            d3_prev = self._d3_history[-2]
            week_change = d3 - d3_prev
            if abs(week_change) > self.d3_week_change_cap:
                d3 = d3_prev + np.sign(week_change) * self.d3_week_change_cap
                d3 = float(np.clip(d3, 0.0, 100.0))
        self._d3_history[-1] = d3
        return d3

    @staticmethod
    def _minmax_map(value: float, v_min: float, v_max: float) -> float:
        if v_max <= v_min:
            return 0.0
        return float(np.clip((value - v_min) / (v_max - v_min), 0.0, 1.0))

