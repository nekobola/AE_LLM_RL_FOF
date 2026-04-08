import numpy as np


class EventTrack:
    """
    Event Track: 纯防守权重求解器。

    规则:
    - 权益切断: w[0]=w[1]=0 (宽基+卫星=0)
    - 非权益资产使用epsilon保护的倒数波动率加权
    """

    IDX_FI = 2
    IDX_SAFE = 3
    IDX_CASH = 4
    EPSILON = 1e-6

    def __init__(self, epsilon: float = 1e-6):
        self.EPSILON = epsilon

    def compute(
        self,
        returns_5d: np.ndarray,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        returns_5d : np.ndarray
            shape = (5, T)

        Returns
        -------
        np.ndarray
            shape = (5,)，W_Event = [0.0, 0.0, w2, w3, w4]
        """
        sigma_fi = returns_5d[self.IDX_FI].std(ddof=1)
        sigma_safe = returns_5d[self.IDX_SAFE].std(ddof=1)
        sigma_cash = returns_5d[self.IDX_CASH].std(ddof=1)

        iv_fi = 1.0 / (sigma_fi + self.EPSILON)
        iv_safe = 1.0 / (sigma_safe + self.EPSILON)
        iv_cash = 1.0 / (sigma_cash + self.EPSILON)

        total = iv_fi + iv_safe + iv_cash
        w_fi = iv_fi / total
        w_safe = iv_safe / total
        w_cash = iv_cash / total

        w_event = np.array([0.0, 0.0, w_fi, w_safe, w_cash])
        return w_event
