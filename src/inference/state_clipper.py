"""RL状态硬截断 (Hard Clipping)

Final_State = np.clip(E_t_zscore, clip_min, clip_max)
约束：PPO 网络的输入状态必须是有界的，避免极端值破坏梯度。
"""
from __future__ import annotations

import numpy as np


def clip_state(
    zscore: float,
    clip_min: float = -5.0,
    clip_max: float = 5.0,
) -> float:
    """
    对 Z-score 执行硬截断。

    Parameters
    ----------
    zscore : float
        输入 Z-score 值
    clip_min : float
        截断下界，默认 -5.0
    clip_max : float
        截断上界，默认 +5.0

    Returns
    -------
    float
        截断后的值，保证 clip_min <= result <= clip_max
    """
    if clip_min > clip_max:
        raise ValueError("clip_min must be <= clip_max")
    return float(np.clip(zscore, clip_min, clip_max))


class StateClipper:
    """
    可配置截断器。

    保存 clip_min / clip_max 配置，支持实例化调用。
    """

    def __init__(self, clip_min: float = -5.0, clip_max: float = 5.0):
        if clip_min > clip_max:
            raise ValueError("clip_min must be <= clip_max")
        self.clip_min = clip_min
        self.clip_max = clip_max

    def clip(self, zscore: float) -> float:
        """对输入 Z-score 执行截断。"""
        return clip_state(zscore, self.clip_min, self.clip_max)

    def __repr__(self) -> str:
        return f"StateClipper(clip_min={self.clip_min}, clip_max={self.clip_max})"
