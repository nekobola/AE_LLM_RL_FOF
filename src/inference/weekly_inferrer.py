"""高频轨推断器（每周五盘后触发）

数据源：quantchdb / ClickHouse 本地数据库。
功能：提取当周五截面特征 → 加载最新.pth权重 → 计算E_t_raw → 输出至任务四。
"""
from __future__ import annotations

import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime

from src.models.regime_autoencoder import RegimeAutoEncoder
from src.data_pipeline.track_b.fetcher import TrackBFetcher
from src.features.asset_features import compute_asset_features
from src.features.macro_features import compute_macro_features
from src.features.normalizer import RollingNormalizer
from src.features.reconstruction_error import compute_reconstruction_error

logger = logging.getLogger(__name__)


class WeeklyInferrer:
    """
    高频轨推断器

    触发时机：每周五盘后
    数据源：quantchdb / ClickHouse（耦合本地服务器DB）
    输出：E_t_raw（送入任务四EMA滤波）
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: str = "cpu",
    ):
        self.config = config
        self.device = device
        self.weights_path_template = config.get("wfo", {}).get("weights_path", "models/ae_weights_{year}Q{quarter}.pth")

    def infer(self, current_date: str) -> Tuple[float, np.ndarray]:
        """
        执行单周推断。

        Parameters
        ----------
        current_date : str
            当前周五日期，格式 YYYY-MM-DD

        Returns
        -------
        Tuple[float, np.ndarray]
            (E_t_raw, 25维归一化特征向量X_normalized)
        """
        year = int(current_date[:4])
        quarter = (int(current_date[5:7]) - 1) // 3 + 1
        weight_filename = self.weights_path_template.format(year=year, quarter=quarter)
        weight_path = Path(weight_filename)

        logger.info(f"[WeeklyInferrer] {current_date} 周五推断，权重: {weight_filename}")

        # 1. 提取本地数据库特征
        fetcher = TrackBFetcher(self.config)
        df_weekly = fetcher.fetch_weekly(end_date=current_date, lookback_weeks=1)
        if df_weekly.empty:
            raise ValueError(f"无可用数据 for date: {current_date}")

        asset_feats = compute_asset_features(df_weekly)
        macro_feats = compute_macro_features(df_weekly)
        X = np.concatenate([asset_feats, macro_feats], axis=1)

        # 取最新一行
        X_t = X[-1]

        # 2. 滚动归一化
        normalizer = RollingNormalizer(
            window=self.config["features"]["normalization"]["zscore_window"],
            min_periods=self.config["features"]["normalization"]["min_periods"],
        )
        X_normalized = normalizer.fit_transform(X)[-1]

        # 3. 加载权重并计算重构误差
        if not weight_path.exists():
            raise FileNotFoundError(f"权重文件不存在: {weight_path}")

        input_dim = self.config["model"]["regime_autoencoder"]["input_dim"]
        latent_dim = self.config["model"]["regime_autoencoder"]["latent_dim"]
        model = RegimeAutoEncoder(input_dim=input_dim, latent_dim=latent_dim)
        state_dict = torch.load(weight_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        # 冻结
        for param in model.parameters():
            param.requires_grad = False

        # 4. 计算E_t_raw
        E_t_raw = compute_reconstruction_error(model, X_normalized, self.device)
        logger.info(f"[WeeklyInferrer] E_t_raw = {E_t_raw:.6f}")
        return E_t_raw, X_normalized
