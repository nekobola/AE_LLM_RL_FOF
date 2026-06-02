import numpy as np
from .normal_track import NormalTrack
from .event_track import EventTrack


class DualTrackEngine:
    """
    异构双轨并发测算引擎。

    输入: 5xT 日收益率矩阵 (来自模块2的合成净值)
    输出: W_Normal (防御权重), W_Event (进攻权重)

    融合: w_final = alpha * w_event + (1-alpha) * w_normal
    - alpha > 0 (max 0.1): 更多进攻 (w_event权重上升)
    - alpha < 0 (min -0.5): 更多防御 (w_normal权重上升)

    资产顺序:
    0=宽基, 1=卫星, 2=固收, 3=避险, 4=现金
    """

    def __init__(self, config: dict | None = None):
        self.config = config
        self.normal_track = NormalTrack(config=config)
        self.event_track = EventTrack()

    def compute(
        self,
        returns_5d: np.ndarray,
        llm_macro: float = 50.0,
        llm_sentiment: float = 50.0,
        llm_risk: float = 50.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        returns_5d : np.ndarray
            shape = (5, T)，5个资产在T个交易日的日收益率
        llm_macro : float
            LLM宏观信号 [0-100]
        llm_sentiment : float
            LLM情绪信号 [0-100]
        llm_risk : float
            LLM风险信号 [0-100]

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (W_Normal, W_Event) 两个5维权重向量
        """
        W_Normal = self.normal_track.compute(returns_5d)
        W_Event = self.event_track.compute(
            returns_5d,
            llm_macro=llm_macro,
            llm_sentiment=llm_sentiment,
            llm_risk=llm_risk,
        )
        return W_Normal, W_Event
