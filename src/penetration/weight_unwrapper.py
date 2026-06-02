import numpy as np


class WeightUnwrapper:
    """
    接收上游5维融合权重，按原合成比例穿透至8只ETF。

    5维权重: W_final^5D = [w1, w2, w3, w4, w5]
    - w1 = 宽基ETF
    - w2 = 卫星ETF (A+B+C等权，故每只 = w2/3)
    - w3 = 固收ETF (利率债+信用债等权，故每只 = w3/2)
    - w4 = 黄金ETF
    - w5 = 货币ETF
    """

    def unwrap(self, w_5d: list[float] | np.ndarray) -> dict[str, float]:
        """
        Parameters
        ----------
        w_5d : list[float] or np.ndarray
            5维权重 [w1, w2, w3, w4, w5]

        Returns
        -------
        dict[str, float]
            各ETF代码及穿透后权重
        """
        w1, w2, w3, w4, w5 = w_5d

        return {
            "宽基ETF": w1,  # 1只
            "卫星ETF_A": w2 / 3.0,  # 3只等权
            "卫星ETF_B": w2 / 3.0,
            "卫星ETF_C": w2 / 3.0,
            "利率债ETF": w3 / 2.0,  # 2只等权
            "信用债ETF": w3 / 2.0,
            "黄金ETF": w4,  # 1只
            "货币ETF": w5,  # 1只
        }
