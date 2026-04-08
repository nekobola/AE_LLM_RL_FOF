import numpy as np
from .normal_track import NormalTrack
from .event_track import EventTrack


class DualTrackEngine:
    """
    异构双轨并发测算引擎。

    输入: 5xT 日收益率矩阵 (来自模块2的合成净值)
    输出: W_Normal (Markowitz进攻权重), W_Event (防御权重)

    资产顺序:
    0=宽基, 1=卫星, 2=固收, 3=避险, 4=现金
    """

    def __init__(self, config: dict | None = None):
        self.normal_track = NormalTrack()
        self.event_track = EventTrack()

    def compute(
        self,
        returns_5d: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        returns_5d : np.ndarray
            shape = (5, T)，5个资产在T个交易日的日收益率

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (W_Normal, W_Event) 两个5维权重向量
        """
        W_Normal = self.normal_track.compute(returns_5d)
        W_Event = self.event_track.compute(returns_5d)
        return W_Normal, W_Event
