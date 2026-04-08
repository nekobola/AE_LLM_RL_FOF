"""阶段一：基座初始化 (2019.01 - 2020.12)

获取104周数据，完成AE网络的第一次初始化训练。
此阶段不向外输出任何状态。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from src.models.regime_autoencoder import RegimeAutoEncoder
from src.data_pipeline.track_a.fetcher import TrackAFetcher
from src.features.asset_features import compute_asset_features
from src.features.macro_features import compute_macro_features
from src.features.normalizer import RollingNormalizer

logger = logging.getLogger(__name__)


class Phase1Initializer:
    """
    阶段一：基座初始化

    时间窗口：2019.01 - 2020.12（104周）
    目标：用随机初始化的权重训练AE，产出基座模型。
    约束：此阶段不向外输出任何状态。
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: str = "cpu",
    ):
        self.config = config
        self.device = device
        self.lookback_weeks = config.get("wfo", {}).get("burn_in", {}).get("phase1_weeks", 104)
        self.epochs = config.get("training", {}).get("phase1_epochs", 50)
        self.lr = config.get("training", {}).get("lr", 1e-3)
        self.batch_size = config.get("training", {}).get("batch_size", 32)

    def run(self) -> RegimeAutoEncoder:
        """
        执行阶段一初始化训练。

        Returns
        -------
        RegimeAutoEncoder
            训练好的AE模型（权重已收敛）
        """
        logger.info("[Phase1] 开始基座初始化训练...")

        # 1. 获取104周数据
        fetcher = TrackAFetcher(self.config)
        df_weekly = fetcher.fetch_weekly(
            start_date="2019-01-01",
            end_date="2020-12-31",
        )

        # 2. 计算25维特征张量
        asset_feats = compute_asset_features(df_weekly)  # shape: (T, 20)
        macro_feats = compute_macro_features(df_weekly)   # shape: (T, 5)
        X = np.concatenate([asset_feats, macro_feats], axis=1)  # shape: (T, 25)

        # 3. 滚动归一化（防穿越窗口[t-252, t-1]，此处全部是历史数据无穿越风险）
        normalizer = RollingNormalizer(
            window=self.config["features"]["normalization"]["zscore_window"],
            min_periods=self.config["features"]["normalization"]["min_periods"],
        )
        X_normalized = normalizer.fit_transform(X)

        # 4. 初始化AE网络（随机权重）
        input_dim = self.config["model"]["regime_autoencoder"]["input_dim"]
        latent_dim = self.config["model"]["regime_autoencoder"]["latent_dim"]
        model = RegimeAutoEncoder(input_dim=input_dim, latent_dim=latent_dim)
        model.to(self.device)

        # 5. 训练收敛
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        X_tensor = torch.from_numpy(X_normalized).float().to(self.device)

        for epoch in range(self.epochs):
            model.train()
            optimizer.zero_grad()
            reconstructed = model(X_tensor)
            loss = criterion(reconstructed, X_tensor)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                logger.info(f"[Phase1] Epoch {epoch+1}/{self.epochs}, Loss={loss.item():.6f}")

        logger.info("[Phase1] 训练完成，权重已收敛")
        return model
