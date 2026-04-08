"""
协方差矩阵计算与等权基础权重输出。
基于5维合成收益率序列，计算5x5协方差矩阵，
作为下游Markowitz/PPO的输入。
"""
import numpy as np
import pandas as pd


class CovarianceWeighter:
    """
    基于5维合成收益率，计算5x5协方差矩阵，
    并输出等权基础权重作为下游Markowitz/PPO的输入。
    """

    def compute_covariance(self, returns_5d: pd.DataFrame, annualize: bool = True) -> np.ndarray:
        """
        计算5x5协方差矩阵。

        annualize=True: 日收益→年化（*252）
        """
        cov = returns_5d.cov()
        if annualize:
            cov = cov * 252
        return cov.values  # shape=(5, 5)

    def equal_weight(self) -> np.ndarray:
        """返回5维等权基础权重 [0.2, 0.2, 0.2, 0.2, 0.2]"""
        return np.array([0.2, 0.2, 0.2, 0.2, 0.2])
