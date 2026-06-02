"""低频轨训练器（每季度末触发）

数据源：AkShare公网数据，与本地交易数据库物理隔离。
功能：拉取过去104周指数特征 → 75%分位数过滤 → 重置权重 → 训练AE → 保存.pth
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime

from src.models.regime_autoencoder import RegimeAutoEncoder
from src.data_pipeline.track_a.fetcher import TrackAFetcher
from src.features.asset_features import compute_asset_features
from src.features.macro_features import compute_macro_features
from src.features.normalizer import RollingNormalizer

logger = logging.getLogger(__name__)


class DualTrackTrainer:
    """
    低频轨训练器

    触发时机：每季度末（3/6/9/12月末）
    数据源：AkShare（物理隔离ClickHouse）
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: str = "cpu",
    ):
        self.config = config
        self.device = device
        wfo_cfg = config.get("wfo", {})
        self.lookback_weeks = wfo_cfg.get("retrain", {}).get("lookback_weeks", 104)
        self.vol_percentile = wfo_cfg.get("burn_in", {}).get("volatility_percentile", 0.75)
        self.epochs = config.get("training", {}).get("retrain_epochs", 30)
        self.lr = config.get("training", {}).get("lr", 1e-3)

    def train_quarter(self, quarter_end: str) -> Path:
        """
        执行单季度重训。

        Parameters
        ----------
        quarter_end : str
            季度末日期，格式 YYYY-MM-DD

        Returns
        -------
        Path
            保存的权重文件路径
        """
        year = int(quarter_end[:4])
        quarter = (int(quarter_end[5:7]) - 1) // 3 + 1
        weight_filename = f"ae_weights_{year}Q{quarter}.pth"
        weights_dir = Path(self.config["wfo"]["weights_path"])
        weights_dir.mkdir(parents=True, exist_ok=True)
        weight_path = weights_dir / weight_filename

        logger.info(f"[DualTrack] 开始{year}Q{quarter}重训...")

        # 1. 获取过去104周数据
        fetcher = TrackAFetcher(self.config)
        df_weekly = fetcher.fetch_weekly(lookback_weeks=self.lookback_weeks)

        # 2. 计算特征
        asset_feats = compute_asset_features(df_weekly)
        macro_feats = compute_macro_features(df_weekly)
        X = np.concatenate([asset_feats, macro_feats], axis=1)

        # 3. 滚动归一化
        normalizer = RollingNormalizer(
            window=self.config["features"]["normalization"]["zscore_window"],
            min_periods=self.config["features"]["normalization"]["min_periods"],
        )
        X_normalized = normalizer.fit_transform(X)

        # 4. 波动率过滤（75%分位数）
        volatility = asset_feats[:, 1]  # volatility_20d 列
        threshold = np.percentile(volatility, self.vol_percentile * 100)
        mask = volatility <= threshold
        X_clean = X_normalized[mask]
        logger.info(f"[DualTrack] 过滤后样本数: {len(X_clean)}/{len(X_normalized)}")

        # 5. 重置权重 + 训练
        input_dim = self.config["model"]["regime_autoencoder"]["input_dim"]
        latent_dim = self.config["model"]["regime_autoencoder"]["latent_dim"]
        model = RegimeAutoEncoder(input_dim=input_dim, latent_dim=latent_dim)
        model.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        X_tensor = torch.from_numpy(X_clean).float().to(self.device)

        for epoch in range(self.epochs):
            model.train()
            optimizer.zero_grad()
            reconstructed = model(X_tensor)
            loss = criterion(reconstructed, X_tensor)
            loss.backward()
            optimizer.step()

        # 6. 保存权重
        torch.save(model.state_dict(), weight_path)
        logger.info(f"[DualTrack] 权重已保存: {weight_path}")
        return weight_path
