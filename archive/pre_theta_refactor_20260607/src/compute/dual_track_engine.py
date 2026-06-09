import numpy as np
from .normal_track import NormalTrack
from .event_track import EventTrack
from .event_track_v2 import EventTrackV2
from .event_track_v3 import EventTrackV3
from .event_track_v3_1 import EventTrackV31


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

    V1/V2/V3/V3.1 toggle (按优先级):
      use_v3_1=True → EventTrackV31 (V3 审计修复版)
      use_v3=True   → EventTrackV3  (V3 原版, 含已知设计缺陷)
      use_v2=True   → EventTrackV2  (Signal-Tilted RB, 2D 三角形)
      else          → EventTrack    (三原型 / softmax-blend, V1)
    """

    def __init__(
        self,
        config: dict | None = None,
        use_v2: bool = False,
        use_v3: bool = False,
        use_v3_1: bool = False,
    ):
        self.config = config
        self.use_v2 = use_v2
        self.use_v3 = use_v3
        self.use_v3_1 = use_v3_1
        self.normal_track = NormalTrack(config=config)
        if use_v3_1:
            self.event_track = EventTrackV31()
        elif use_v3:
            self.event_track = EventTrackV3()
        elif use_v2:
            self.event_track = EventTrackV2()
        else:
            self.event_track = EventTrack()

    def compute(
        self,
        returns_5d: np.ndarray,
        llm_macro: float = 50.0,
        llm_sentiment: float = 50.0,
        llm_risk: float = 50.0,
        ae_error: float | None = None,
        tau: float | None = None,
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
        ae_error : float, optional
            AE 重建误差,用于触发 bear-regime 强制防守
        tau : float, optional
            牛熊切换阈值

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (W_Normal, W_Event) 两个5维权重向量
        """
        W_Normal = self.normal_track.compute(returns_5d, ae_error=ae_error, tau=tau)
        W_Event = self.event_track.compute(
            returns_5d,
            llm_macro=llm_macro,
            llm_sentiment=llm_sentiment,
            llm_risk=llm_risk,
            ae_error=ae_error,
            tau=tau,
        )
        return W_Normal, W_Event
