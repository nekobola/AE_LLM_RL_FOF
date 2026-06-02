"""阶段二：MAD标尺铸造 (2021.01 - 2021.12)

冻结阶段一训练好的权重，对2021年进行52周的单步推断。
积攒52个E_t_smoothed，用于生成安全可靠的滚动中位数和MAD基准。
此阶段不训练，仅推断。
"""
from __future__ import annotations

import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple
import logging

from src.models.regime_autoencoder import RegimeAutoEncoder
from src.data_pipeline.track_a.fetcher import TrackAFetcher
from src.features.asset_features import compute_asset_features
from src.features.macro_features import compute_macro_features
from src.features.normalizer import RollingNormalizer
from src.features.reconstruction_error import compute_reconstruction_error

logger = logging.getLogger(__name__)


class Phase2MADCalibrator:
    """
    阶段二：MAD标尺铸造

    时间窗口：2021.01 - 2021.12（52周）
    目标：冻结Phase1权重，积攒52个E_t_smoothed，生成中位数/MAD基准。
    约束：仅推断，不训练。
    """

    def __init__(
        self,
        phase1_model: RegimeAutoEncoder,
        config: Dict[str, Any],
        device: str = "cpu",
    ):
        self.phase1_model = phase1_model
        self.config = config
        self.device = device
        self.ema_alpha = config.get("inference", {}).get("ema_alpha", 0.05)

        # 冻结权重
        self.phase1_model.eval()
        for param in self.phase1_model.parameters():
            param.requires_grad = False

    def run(self) -> Tuple[np.ndarray, float, float]:
        """
        执行阶段二推断，积攒E_t_smoothed序列，计算MAD标尺。

        Returns
        -------
        Tuple[np.ndarray, float, float]
            (E_smoothed序列, 中位数基准, MAD基准)
        """
        logger.info("[Phase2] 开始MAD标尺铸造...")

        # 获取2021年数据
        fetcher = TrackAFetcher(self.config)
        df_weekly = fetcher.fetch_weekly(
            start_date="2021-01-01",
            end_date="2021-12-31",
        )

        asset_feats = compute_asset_features(df_weekly)
        macro_feats = compute_macro_features(df_weekly)
        X = np.concatenate([asset_feats, macro_feats], axis=1)

        normalizer = RollingNormalizer(
            window=self.config["features"]["normalization"]["zscore_window"],
            min_periods=self.config["features"]["normalization"]["min_periods"],
        )
        X_normalized = normalizer.fit_transform(X)

        # EMA递推计算E_t_smoothed
        E_smoothed_series = []
        E_prev = None

        for i in range(len(X_normalized)):
            X_t = X_normalized[i]
            E_raw = compute_reconstruction_error(self.phase1_model, X_t, self.device)

            if E_prev is None:
                E_smoothed = E_raw
            else:
                E_smoothed = self.ema_alpha * E_raw + (1 - self.ema_alpha) * E_prev

            E_smoothed_series.append(E_smoothed)
            E_prev = E_smoothed

        E_smoothed_series = np.array(E_smoothed_series)

        # 计算MAD标尺
        median = np.median(E_smoothed_series)
        mad = np.median(np.abs(E_smoothed_series - median))
        mad_safe = max(mad, abs(median) * 0.05)  # 底层物理防线

        logger.info(
            f"[Phase2] 完成。52个E_t_smoothed样本，"
            f"Median={median:.6f}, MAD={mad:.6f}, MAD_safe={mad_safe:.6f}"
        )

        return E_smoothed_series, median, mad_safe
