"""滚动稳健Z-score计算（严格抗未来）

基准窗口：[t-52, t-1] — 严格禁止包含 t 时刻及未来数据
统计量：Median = median(S_past)，MAD = median(|S_past - Median|)
底层防线：MAD_safe = max(MAD, |Median| × 0.05)
Z-score：E_t_zscore = (E_t_smoothed - Median) / (MAD_safe × 1.4826)
"""
from __future__ import annotations

from collections import deque
from typing import Optional


class RobustZScore:
    """
    滚动稳健 Z-score 计算器（严格抗未来实现）。

    使用固定大小窗口队列，只记录过去52周的 E_smoothed 历史。
    计算当前时刻 Z-score 时，只访问 [t-52, t-1] 窗口，
    绝对不包含 t 时刻及未来的任何数据。

    参数
    ----
    window : int
        滚动窗口大小，默认 52 周（1年）
    mad_floor : float
        MAD 底层物理防线系数，默认 0.05
        即 MAD_safe = max(MAD, |Median| × mad_floor)
    """

    MAD_COEFFICIENT: float = 1.4826  # 正态分布下 MAD→σ 的无偏估计系数

    def __init__(
        self,
        window: int = 52,
        mad_floor: float = 0.05,
    ):
        if window < 1:
            raise ValueError("window must be >= 1")
        if not (0 <= mad_floor <= 1):
            raise ValueError("mad_floor must be in [0, 1]")

        self.window = window
        self.mad_floor = mad_floor
        # 使用 deque 固定窗口大小，只记录历史（不含当前）
        self._history: deque[float] = deque(maxlen=window)

    def step(self, E_smoothed: float) -> float:
        """
        单步计算 Z-score。

        Parameters
        ----------
        E_smoothed : float
            当前时刻 t 的 EMA 平滑误差 E_t_smoothed

        Returns
        -------
        float
            E_t_zscore；若窗口数据不足则返回 0.0（Burning-in 期）
        """
        # -------- 严格抗未来 --------
        # 计算基准窗口 [t-52, t-1]，即 self._history（不包含当前 E_smoothed）
        if len(self._history) < self.window:
            # Burning-in：数据不足，输出0
            self._history.append(E_smoothed)
            return 0.0

        past = list(self._history)  # 复制，不影响原 deque

        # 计算中位数
        sorted_past = sorted(past)
        n = len(sorted_past)
        if n % 2 == 0:
            median = (sorted_past[n // 2 - 1] + sorted_past[n // 2]) / 2
        else:
            median = sorted_past[n // 2]

        # 计算 MAD
        abs_devs = [abs(v - median) for v in past]
        sorted_devs = sorted(abs_devs)
        if n % 2 == 0:
            mad = (sorted_devs[n // 2 - 1] + sorted_devs[n // 2]) / 2
        else:
            mad = sorted_devs[n // 2]

        # 底层物理防线
        mad_safe = max(mad, abs(median) * self.mad_floor)

        # Z-score
        if mad_safe < 1e-12:
            # 防止除零
            zscore = 0.0
        else:
            zscore = (E_smoothed - median) / (mad_safe * self.MAD_COEFFICIENT)

        # append 当前值到历史，供下一时刻使用
        self._history.append(E_smoothed)
        return zscore

    def reset(self) -> None:
        """重置历史窗口。"""
        self._history.clear()

    @property
    def window_size(self) -> int:
        return self.window

    @property
    def current_history_len(self) -> int:
        return len(self._history)
