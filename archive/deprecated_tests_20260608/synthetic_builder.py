"""
净值合成器：
将8只ETF收益率合成为5维超级资产。
- V1 (超级宽基) = 宽基ETF日收益率
- V2 (超级卫星) = (卫星A + 卫星B + 卫星C) / 3.0 等权
- V3 (超级固收) = (利率债 + 信用债) / 2.0 等权
- V4 (超级避险) = 黄金ETF日收益率
- V5 (超级现金) = 货币ETF日收益率
"""
import pandas as pd
import numpy as np


class SyntheticBuilder:
    def __init__(self, config: dict):
        self.config = config

    def build(
        self,
        etf_codes: dict[str, str],
        returns_df: pd.DataFrame,
        window: int = 60,
    ) -> pd.DataFrame:
        """
        合成5维资产收益率序列。

        returns_df: 来自quantchdb，8只ETF的日收益率，shape=(T, 8)
        window: 取过去60天

        Returns:
            pd.DataFrame: shape=(window, 5), columns=["V1","V2","V3","V4","V5"]
        """
        df = returns_df.tail(window).copy()

        # V1: 宽基
        v1 = df[etf_codes["宽基"]]

        # V2: 卫星等权
        satellite_codes = [etf_codes["卫星A"], etf_codes["卫星B"], etf_codes["卫星C"]]
        v2 = df[satellite_codes].mean(axis=1)

        # V3: 固收等权
        fi_codes = [etf_codes["固收利率债"], etf_codes["固收信用债"]]
        v3 = df[fi_codes].mean(axis=1)

        # V4: 黄金
        v4 = df[etf_codes["黄金"]]

        # V5: 货币
        v5 = df[etf_codes["现金"]]

        result = pd.DataFrame(
            {"V1": v1, "V2": v2, "V3": v3, "V4": v4, "V5": v5},
            index=df.index,
        )

        return result  # shape=(window, 5)
