"""重构误差计算模块

E_t_raw = ||X_t - Decoder(Encoder(X_t))||_2^2
"""
from __future__ import annotations

import numpy as np
import torch
from typing import Optional

from src.models.regime_autoencoder import RegimeAutoEncoder


def compute_reconstruction_error(
    model: RegimeAutoEncoder,
    X: np.ndarray,
    device: str = "cpu",
) -> float:
    """
    计算重构误差 E_t_raw = ||X_t - Decoder(Encoder(X_t))||_2^2

    Parameters
    ----------
    model : RegimeAutoEncoder
        已加载权重的AE网络（权重冻结，不参与梯度计算）
    X : np.ndarray
        形状为 (25,) 的单样本特征向量
    device : str
        计算设备

    Returns
    -------
    float
        重构误差标量
    """
    model.eval()
    X_tensor = torch.from_numpy(X).float().unsqueeze(0).to(device)
    with torch.no_grad():
        reconstructed = model(X_tensor)
        error = torch.sum((X_tensor - reconstructed) ** 2).item()
    return error


def compute_reconstruction_error_batch(
    model: RegimeAutoEncoder,
    X_batch: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """
    批量计算重构误差

    Parameters
    ----------
    model : RegimeAutoEncoder
    X_batch : np.ndarray
        形状为 (N, 25) 的批量特征矩阵
    device : str

    Returns
    -------
    np.ndarray
        形状为 (N,) 的重构误差向量
    """
    model.eval()
    X_tensor = torch.from_numpy(X_batch).float().to(device)
    with torch.no_grad():
        reconstructed = model(X_tensor)
        errors = torch.sum((X_tensor - reconstructed) ** 2, dim=1).numpy()
    return errors
