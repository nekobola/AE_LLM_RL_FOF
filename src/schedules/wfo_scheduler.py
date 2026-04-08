"""WFO调度器

协调低频轨（季度重训）和高频轨（周频推断）的执行。
触发机制：配置驱动，非硬编码。
"""
from __future__ import annotations

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from src.training.dual_track.trainer import DualTrackTrainer
from src.inference.weekly_inferrer import WeeklyInferrer
from src.training.burn_in.phase1_init import Phase1Initializer
from src.training.burn_in.phase2_mad_calibrator import Phase2MADCalibrator
from src.models.regime_autoencoder import RegimeAutoEncoder

logger = logging.getLogger(__name__)


class WFOScheduler:
    """
    Walk-Forward Optimization 调度器

    季度触发：低频轨训练器（DualTrackTrainer）
    周频触发：高频轨推断器（WeeklyInferrer）
    冷启动：Phase1(104周) + Phase2(52周) → 并网
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: str = "cpu",
    ):
        self.config = config
        self.device = device
        self.phase1_init = Phase1Initializer(config, device)
        self.phase2_calibrator = None  # 延迟初始化
        self.dual_trainer = DualTrackTrainer(config, device)
        self.weekly_inferrer = WeeklyInferrer(config, device)

        self.burn_in_weeks = (
            config.get("wfo", {}).get("burn_in", {}).get("phase1_weeks", 104)
            + config.get("wfo", {}).get("burn_in", {}).get("phase2_weeks", 52)
        )
        self._phase1_model: Optional[RegimeAutoEncoder] = None
        self._median基准: float = 0.0
        self._mad_safe: float = 0.0

    def run_burn_in(self) -> None:
        """
        执行冷启动（Phase1 + Phase2）。
        仅在系统初始化时调用一次。
        """
        logger.info(f"[WFOScheduler] 开始冷启动（共{self.burn_in_weeks}周盲区）...")

        # Phase1: 初始化训练
        self._phase1_model = self.phase1_init.run()
        logger.info("[WFOScheduler] Phase1 完成")

        # Phase2: MAD标尺铸造
        self.phase2_calibrator = Phase2MADCalibrator(
            phase1_model=self._phase1_model,
            config=self.config,
            device=self.device,
        )
        _, self._median基准, self._mad_safe = self.phase2_calibrator.run()
        logger.info("[WFOScheduler] Phase2 完成，MAD标尺已铸造")

    def trigger_weekly_inference(self, current_date: str) -> float:
        """
        触发周频推断（高频轨）。

        Parameters
        ----------
        current_date : str
            当前周五日期 YYYY-MM-DD

        Returns
        -------
        float
            E_t_raw（送入任务四处理）
        """
        E_t_raw, _ = self.weekly_inferrer.infer(current_date)
        return E_t_raw

    def trigger_quarterly_retrain(self, quarter_end: str) -> Path:
        """
        触发季度重训（低频轨）。

        Parameters
        ----------
        quarter_end : str
            季度末日期 YYYY-MM-DD

        Returns
        -------
        Path
            权重文件路径
        """
        return self.dual_trainer.train_quarter(quarter_end)

    @property
    def median基准(self) -> float:
        return self._median基准

    @property
    def mad_safe(self) -> float:
        return self._mad_safe
