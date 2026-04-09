"""State Assembler

Assembles the 10-dimensional dense state vector S_t ∈ R^10 for the PPO agent.
All dimensions are mapped to roughly [-1, 1] or [0, 1] for stable NN training.

Dimensions:
  1.  E_t_zscore            — AE reconstruction error, Z-score standardized
  2.  vol_mkt_20d_norm      — Market 20d annualized vol, MinMax → [0, 1]
  3.  llm_macro_norm        — LLM macro顺风度 (d1-50)/50 ∈ [-1, 1]
  4.  llm_sent_norm         — LLM sentiment (d2-50)/50 ∈ [-1, 1]
  5.  llm_risk_norm         — LLM tail risk d3/100 ∈ [0, 1]
  6.  port_sharpe_20d      — Portfolio 20d Sharpe, hard-clipped to [-3, 3]
  7.  port_mdd_current      — Current drawdown ∈ [0, 1]
  8.  regret_ema_norm      — Regret EMA normalized to [0, 1]
  9.  tau_prev_norm         — Previous threshold τ_{t-1}, MinMax → [0, 1]
  10. alpha_prev            — Previous fusion ratio α_{t-1} ∈ [0, 1]
"""
from __future__ import annotations

import numpy as np
from typing import NamedTuple


class StateTuple(NamedTuple):
    """10-dim state vector with named fields."""
    e_t_zscore: float          # dim 1
    vol_mkt_20d_norm: float     # dim 2
    llm_macro_norm: float       # dim 3
    llm_sent_norm: float        # dim 4
    llm_risk_norm: float        # dim 5
    port_sharpe_20d: float      # dim 6
    port_mdd_current: float     # dim 7
    regret_ema_norm: float      # dim 8
    tau_prev_norm: float        # dim 9
    alpha_prev: float           # dim 10


class StateAssembler:
    """
    Assembles 10-dim state S_t from live regime indicators and portfolio metrics.
    All normalization parameters are fit on historical data and stored as attributes.
    """

    def __init__(
        self,
        sharpe_clip_low: float = -3.0,
        sharpe_clip_high: float = 3.0,
    ):
        """
        Parameters
        ----------
        sharpe_clip_low / sharpe_clip_high : float
            Hard-clip range for port_sharpe_20d.
        """
        self.sharpe_clip_low = sharpe_clip_low
        self.sharpe_clip_high = sharpe_clip_high

        # Running stats for Z-score of reconstruction error
        self._ae_mean: float = 0.0
        self._ae_std: float = 1.0

        # Running stats for vol_mkt MinMax
        self._vol_min: float = 0.0
        self._vol_max: float = 1.0

        # Running stats for tau MinMax
        self._tau_min: float = 0.0
        self._tau_max: float = 1.0

    def fit_normalizers(self, ae_errors: np.ndarray, vol_series: np.ndarray, tau_series: np.ndarray) -> None:
        """
        Fit all normalizer parameters from historical data.
        Must be called during burn-in with data up to t-1.

        Parameters
        ----------
        ae_errors : np.ndarray, shape (N,)
            Historical AE reconstruction error series.
        vol_series : np.ndarray, shape (N,)
            Historical 20d annualized vol series.
        tau_series : np.ndarray, shape (N,)
            Historical threshold τ series.
        """
        if len(ae_errors) > 1:
            self._ae_mean = float(np.mean(ae_errors))
            self._ae_std = float(np.std(ae_errors, ddof=1))
        if len(vol_series) > 1:
            self._vol_min = float(np.min(vol_series))
            self._vol_max = float(np.max(vol_series))
        if len(tau_series) > 1:
            self._tau_min = float(np.min(tau_series))
            self._tau_max = float(np.max(tau_series))

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
        tau_prev: float,
        alpha_prev: float,
    ) -> np.ndarray:
        """
        Assemble 10-dim state vector.

        Parameters
        ----------
        ae_error : float
            Raw AE reconstruction error E_t.
        vol_mkt_20d : float
            Market 20d annualized volatility.
        llm_macro : float
            LLM macro顺风度 score d1 ∈ [0, 100].
        llm_sentiment : float
            LLM sentiment score d2 ∈ [0, 100].
        llm_risk : float
            LLM tail risk score d3 ∈ [0, 100].
        port_sharpe_20d : float
            Portfolio 20d realized Sharpe ratio.
        port_mdd_current : float
            Current drawdown ∈ [0, 1].
        regret_ema_norm : float
            Normalized regret EMA ∈ [0, 1].
        tau_prev : float
            Threshold τ_{t-1}.
        alpha_prev : float
            Fusion ratio α_{t-1} ∈ [0, 1].

        Returns
        -------
        np.ndarray, shape (10,)
            State vector S_t.
        """
        # 1. AE reconstruction error — Z-score
        ae_zscore = (ae_error - self._ae_mean) / (self._ae_std + 1e-9)

        # 2. Market vol — MinMax → [0, 1]
        vol_norm = self._minmax_map(vol_mkt_20d, self._vol_min, self._vol_max)

        # 3. LLM macro顺风度 — (d1-50)/50 ∈ [-1, 1]
        llm_macro_norm = (llm_macro - 50.0) / 50.0

        # 4. LLM sentiment — (d2-50)/50 ∈ [-1, 1]
        llm_sent_norm = (llm_sentiment - 50.0) / 50.0

        # 5. LLM tail risk — d3/100 ∈ [0, 1]
        llm_risk_norm = llm_risk / 100.0

        # 6. Portfolio Sharpe — hard clip to [-3, 3]
        sharpe_clipped = float(np.clip(port_sharpe_20d, self.sharpe_clip_low, self.sharpe_clip_high))

        # 7. Portfolio MDD — already in [0, 1]
        mdd_current = port_mdd_current

        # 8. Regret EMA norm — already [0, 1]
        regret_norm = regret_ema_norm

        # 9. Tau prev — MinMax → [0, 1]
        tau_prev_norm = self._minmax_map(tau_prev, self._tau_min, self._tau_max)

        # 10. Alpha prev — already [0, 1]
        alpha = alpha_prev

        return np.array([
            ae_zscore,
            vol_norm,
            llm_macro_norm,
            llm_sent_norm,
            llm_risk_norm,
            sharpe_clipped,
            mdd_current,
            regret_norm,
            tau_prev_norm,
            alpha,
        ], dtype=np.float32)

    @staticmethod
    def _minmax_map(value: float, v_min: float, v_max: float) -> float:
        """Map value to [0, 1] via MinMax. Returns 0 if v_min == v_max."""
        if v_max <= v_min:
            return 0.0
        return float(np.clip((value - v_min) / (v_max - v_min), 0.0, 1.0))
