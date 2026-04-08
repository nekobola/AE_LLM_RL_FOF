"""冷启动盲区处理器 (Burn-in Period Handler)

约束：
- 系统前156周（104周训练+52周Z-score积累）处于盲区
- 盲区内强制输出 Final_State = 0.0
- 禁止返回 NaN，确保 PPO 基座的回测连贯性
"""
from __future__ import annotations

import numpy as np


class BurnInHandler:
    """
    冷启动盲区处理器。

    规则：
    1. 周计数器未达到 burn_in_weeks 时，强制输出 0.0
    2. 任何时刻遇到 NaN，强制输出 0.0
    3. 计数器达到后，正常透传上游状态

    156周的构成：
    - Phase1: 104周（AE初始化训练）
    - Phase2: 52周（Z-score基准积累）
    """

    def __init__(self, burn_in_weeks: int = 156):
        """
        Parameters
        ----------
        burn_in_weeks : int
            盲区周数，默认 156 周（3年）
        """
        if burn_in_weeks < 0:
            raise ValueError("burn_in_weeks must be non-negative")

        self.burn_in_weeks = burn_in_weeks
        self._week_counter: int = 0

    def handle(self, state: float) -> float:
        """
        处理状态输出。

        Parameters
        ----------
        state : float
            来自上游的最终状态（已经过 Hard Clip）

        Returns
        -------
        float
            Burn-in 处理后的最终恐慌指数
        """
        self._week_counter += 1

        # 规则1：盲区强制输出0
        if self._week_counter <= self.burn_in_weeks:
            return 0.0

        # 规则2：NaN强制输出0
        if np.isnan(state) or np.isinf(state):
            return 0.0

        return state

    def reset(self) -> None:
        """重置周计数器（系统重启时调用）。"""
        self._week_counter = 0

    @property
    def week_counter(self) -> int:
        """当前周计数器。"""
        return self._week_counter

    @property
    def is_in_burn_in(self) -> bool:
        """是否仍处于 Burn-in 盲区。"""
        return self._week_counter <= self.burn_in_weeks

    @property
    def remaining_burn_in_weeks(self) -> int:
        """距离 Burn-in 结束还剩多少周。"""
        remaining = self.burn_in_weeks - self._week_counter
        return max(0, remaining)

    def __repr__(self) -> str:
        return (
            f"BurnInHandler(burn_in_weeks={self.burn_in_weeks}, "
            f"current_week={self._week_counter}, "
            f"is_in_burn_in={self.is_in_burn_in})"
        )
