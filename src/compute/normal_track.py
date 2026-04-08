import numpy as np
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize


class NormalTrack:
    """
    Normal Track: 基于Markowitz的权益偏向权重求解器。

    目标: 最小化组合方差 (GMV) -> minimize w.T @ Sigma @ w
    约束: sum(w) = 1.0, 各资产上下限
    """

    IDX_BROAD = 0
    IDX_SATELLITE = 1
    IDX_FI = 2
    IDX_SAFE = 3
    IDX_CASH = 4

    def __init__(self, bounds: dict | None = None):
        """
        bounds: dict, 格式 {asset_idx: (min, max)}
               默认使用 config 中的默认值
        """
        self.default_bounds = {
            self.IDX_BROAD:     (0.05, 0.40),
            self.IDX_SATELLITE: (0.05, 0.30),
            self.IDX_FI:        (0.10, 0.60),
            self.IDX_SAFE:      (0.00, 0.20),
            self.IDX_CASH:      (0.00, 0.20),
        }
        self.bounds = bounds or self.default_bounds

    def compute(
        self,
        returns_5d: np.ndarray,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        returns_5d : np.ndarray
            shape = (5, T)，5个资产在T个交易日的日收益率

        Returns
        -------
        np.ndarray
            shape = (5,)，W_Normal，优化后的5维权重向量
        """
        # 1. 协方差收缩 (LedoitWolf)
        cov_estimator = LedoitWolf()
        cov_estimator.fit(returns_5d.T)  # expects (n_samples, n_features)
        Sigma = cov_estimator.covariance_  # shape = (5, 5)

        # 2. GMV 目标函数: minimize w.T @ Sigma @ w
        def gmv_variance(w):
            w = np.array(w)
            return float(w @ Sigma @ w)

        # 3. 约束: sum(w) = 1.0
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        # 4. 边界
        bounds_list = [self.bounds[i] for i in range(5)]

        # 5. 初始猜测 (等权)
        w0 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        # 6. SLSQP 求解
        result = minimize(
            gmv_variance,
            w0,
            method="SLSQP",
            bounds=bounds_list,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )

        if not result.success:
            return w0.copy()

        return result.x  # shape = (5,)
