"""恐慌指数整合输出模块

整合完整流水线：
E_t_raw → EMA滤波 → E_t_smoothed
        → Rolling Z-score → E_t_zscore
        → Hard Clip → E_t_clipped
        → Burn-in处理 → Final_State ∈ [-5.0, 5.0]

用法：
    output = PanicIndexOutput(config)
    final_state = output.step(E_t_raw)
"""
from __future__ import annotations

from typing import Dict, Any, Optional

from src.inference.ema_filter import EMAFilter
from src.inference.robust_zscore import RobustZScore
from src.inference.state_clipper import StateClipper
from src.inference.burn_in_handler import BurnInHandler


class PanicIndexOutput:
    """
    恐慌指数整合输出器。

    串联 EMA滤波 → RobustZScore → StateClipper → BurnInHandler
    四个阶段均为严格无副作用的纯步骤计算（除 history 积累外）。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Parameters
        ----------
        config : Dict[str, Any]
            配置字典，必须包含以下键：
            - inference.ema_alpha: EMA 衰减系数
            - inference.zscore_window: Z-score 滚动窗口
            - inference.mad_safe_floor: MAD 底层防线系数
            - inference.clip_min / clip_max: 截断上下界
            - wfo.burn_in.phase1_weeks + phase2_weeks: Burn-in 周数
        """
        inf_cfg = config["inference"]
        wfo_cfg = config["wfo"]
        burn_in_cfg = wfo_cfg["burn_in"]

        burn_in_weeks = (
            burn_in_cfg.get("phase1_weeks", 104)
            + burn_in_cfg.get("phase2_weeks", 52)
        )

        self.ema = EMAFilter(alpha=inf_cfg["ema_alpha"])
        self.zscore = RobustZScore(
            window=inf_cfg["zscore_window"],
            mad_floor=inf_cfg["mad_safe_floor"],
        )
        self.clipper = StateClipper(
            clip_min=inf_cfg["clip_min"],
            clip_max=inf_cfg["clip_max"],
        )
        self.burn_in = BurnInHandler(burn_in_weeks=burn_in_weeks)

    def step(self, E_raw: float) -> float:
        """
        单步计算最终恐慌指数。

        Parameters
        ----------
        E_raw : float
            当前时刻的重构误差原始值 E_t_raw

        Returns
        -------
        float
            Final_State，已截断且经 Burn-in 处理 ∈ [-5.0, 5.0]
        """
        # 1. EMA 滤波
        E_smoothed = self.ema.step(E_raw)

        # 2. 滚动稳健 Z-score
        E_zscore = self.zscore.step(E_smoothed)

        # 3. Hard Clip
        E_clipped = self.clipper.clip(E_zscore)

        # 4. Burn-in 处理
        final_state = self.burn_in.handle(E_clipped)

        return final_state

    def reset(self) -> None:
        """重置所有子模块状态（系统重启时调用）。"""
        self.ema.reset()
        self.zscore.reset()
        self.burn_in.reset()

    @property
    def is_in_burn_in(self) -> bool:
        """当前是否处于 Burn-in 盲区。"""
        return self.burn_in.is_in_burn_in

    @property
    def remaining_burn_in_weeks(self) -> int:
        """距离 Burn-in 结束还剩多少周。"""
        return self.burn_in.remaining_burn_in_weeks

    def __repr__(self) -> str:
        return (
            f"PanicIndexOutput("
            f"ema_alpha={self.ema.alpha}, "
            f"zscore_window={self.zscore.window_size}, "
            f"clip_range=[{self.clipper.clip_min}, {self.clipper.clip_max}], "
            f"burn_in={self.burn_in})"
        )
