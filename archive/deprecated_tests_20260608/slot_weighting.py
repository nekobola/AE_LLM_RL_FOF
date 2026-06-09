"""Slot-specific P-weight vectors for score computation.

Score = p1*d1 + p2*d2 - p3*d3
where:
  d1 = policy_signal_score   (政策信号维度)
  d2 = sector_momentum_score(行业动量维度)
  d3 = valuation_risk_score  (估值风险维度)
"""

from typing import Dict

import numpy as np

# 各插槽P权重向量（固定）
P_VECTORS: Dict[str, np.ndarray] = {
    "wide_base": np.array([0.6, 0.1, 0.3]),
    "satellite": np.array([0.2, 0.6, 0.2]),
    "fixed_income": np.array([0.5, 0.0, 0.5]),
    "hedging": np.array([0.0, 0.0, 0.0]),  # 避险槽固定，不评分
    "cash": np.array([0.0, 0.0, 0.0]),     # 现金槽固定，不评分
}


def compute_slot_score(d1: float, d2: float, d3: float, pool_type: str) -> float:
    """
    Compute weighted score for a given slot type.

    Args:
        d1: Policy signal dimension score
        d2: Sector momentum dimension score
        d3: Valuation risk dimension score (penalizes, hence minus sign)
        pool_type: One of 'wide_base', 'satellite', 'fixed_income'

    Returns:
        Composite score as float
    """
    if pool_type not in P_VECTORS:
        raise ValueError(f"Unknown pool_type: {pool_type}")
    p = P_VECTORS[pool_type]
    return float(p[0] * d1 + p[1] * d2 - p[2] * d3)
