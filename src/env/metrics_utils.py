"""Metrics Utilities

Pure functional / static methods for high-frequency env calls.
All computations are vectorized via NumPy; zero for-loops.
"""
from __future__ import annotations

import numpy as np


def calculate_tracking_error(
    port_returns: np.ndarray,
    bench_returns: np.ndarray,
    annualize: bool = True,
) -> float:
    """
    Tracking Error (TE)

    Physical meaning: degree of portfolio deviation from benchmark (wide-base).
    In normal regime, severe penalty on deviation; demands close beta tracking.

    TE = sqrt(252) * std(r_port - r_bench, ddof=1)

    Parameters
    ----------
    port_returns : np.ndarray, shape (N,)
        Portfolio daily return series over last N days.
    bench_returns : np.ndarray, shape (N,)
        Benchmark (wide-base slot ETF) daily return series over last N days.
    annualize : bool
        If True, multiply by sqrt(252) for annualization.

    Returns
    -------
    float
        Annualized tracking error.
    """
    active_returns = port_returns - bench_returns
    te = float(np.std(active_returns, ddof=1))
    if annualize:
        te = te * np.sqrt(252)
    return te


def calculate_current_drawdown(equity_curve: np.ndarray) -> float:
    """
    Current Drawdown (CDD)

    Physical meaning: at current step, how far has the account fallen from
    its historical high-water mark. Provides immediate penalty gradient,
    clearer than traditional MDD for RL.

    CDD_t = (HWM_t - P_t) / HWM_t

    For O(1) incremental update in env loop, maintain self.hwm externally.

    Parameters
    ----------
    equity_curve : np.ndarray, shape (T,)
        Cumulative NAV curve since portfolio inception.

    Returns
    -------
    float
        Current drawdown in [0, 1]. 0 if at new high.
    """
    current_nav = equity_curve[-1]
    historical_peak = np.max(equity_curve)

    if current_nav >= historical_peak:
        return 0.0

    return float((historical_peak - current_nav) / historical_peak)


def calculate_current_drawdown_incremental(
    current_nav: float,
    hwm: float,
) -> tuple[float, float]:
    """
    O(1) incremental CDD update.

    Parameters
    ----------
    current_nav : float
        NAV at current timestep.
    hwm : float
        Historical high-water mark prior to this step.

    Returns
    -------
    tuple[float, float]
        (cdd, new_hwm) — cdd in [0,1], new_hwm = max(hwm, current_nav).
    """
    if current_nav >= hwm:
        return 0.0, current_nav
    return float((hwm - current_nav) / hwm), hwm


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualize: bool = True,
) -> float:
    """
    Sharpe Ratio.

    Sharpe = (mean(r) - r_f) / std(r)

    Returns
    -------
    float
        Annualized Sharpe ratio.
    """
    excess = returns - risk_free_rate
    mean_ret = float(np.mean(excess))
    std_ret = float(np.std(excess, ddof=1))
    if std_ret == 0:
        return 0.0
    sharpe = mean_ret / std_ret
    if annualize:
        sharpe = sharpe * np.sqrt(252)
    return sharpe
