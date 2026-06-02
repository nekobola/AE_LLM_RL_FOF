"""EMA状态机递推滤波器

E_t_smoothed = α * E_t_raw + (1-α) * E_{t-1}_smoothed

约束：必须以状态机模式递推计算，严禁调用全局窗口函数。
"""
from __future__ import annotations


class EMAFilter:
    """
    EMA 状态机递推滤波器。

    使用状态机模式，每次 step() 调用基于上一次的状态进行递推，
    不依赖任何全局窗口或历史数据缓存（仅保留上一时刻状态）。

    公式：E_t_smoothed = α * E_t_raw + (1-α) * E_{t-1}_smoothed
    """

    def __init__(self, alpha: float):
        """
        Parameters
        ----------
        alpha : float
            EMA 衰减系数，取值 (0, 1]，越接近0越平滑
        """
        if not (0 < alpha <= 1):
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._prev_smoothed: float | None = None

    def step(self, E_raw: float) -> float:
        """
        单步递推。

        Parameters
        ----------
        E_raw : float
            当前时刻的重构误差原始值 E_t_raw

        Returns
        -------
        float
            当前时刻的平滑误差 E_t_smoothed
        """
        if self._prev_smoothed is None:
            # 第一次：E_t_smoothed = E_t_raw（无历史平滑值）
            self._prev_smoothed = E_raw
        else:
            self._prev_smoothed = self.alpha * E_raw + (1 - self.alpha) * self._prev_smoothed
        return self._prev_smoothed

    def reset(self) -> None:
        """重置滤波器状态（重新初始化时调用）。"""
        self._prev_smoothed = None

    @property
    def last_smoothed(self) -> float | None:
        """返回上一时刻的平滑值（用于调试）。"""
        return self._prev_smoothed
