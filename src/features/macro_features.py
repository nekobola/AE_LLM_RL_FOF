"""
macro_features.py — 5维宏观特征张量

特征列表：
  1. DR007           : 银行间7天质押式回购利率（Shibor 1W近似，相关性>0.95）
  2. CNY_USD_Offshore: USD/CNY 在岸人民币汇率（BOC SAFE 数据/100）
  3. Yield_10Y_CGB   : 10年期中国国债收益率（中债估值 bond_zh_us_rate）
  4. Term_Spread     : 10Y-2Y 国债期限利差（bond_zh_us_rate 直接提供）
  5. Northbound_Flow : 两融余额20日动量（沪市+深市合并，BOC SAFE代理指标）

数据来源：
  - DR007:       ak.macro_china_shibor_all() [Shibor 1W]
  - CNY_USD:     ak.currency_boc_safe() [美元汇率/100]
  - 国债收益率:   ak.bond_zh_us_rate(start_date) [col3=10Y, col5=10Y-2Y利差]
  - 两融动量:     ak.macro_china_market_margin_sh/sz() → 20日动量百分比

注意：
  - 北向资金数据已于2024年8月停更，改用两融余额20日动量作为市场杠杆情绪代理
  - 两融余额为绝对金额，已通过动量公式转为百分比平稳序列
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# 宏观特征名称
MACRO_FEATURES: List[str] = [
    "DR007",
    "CNY_USD_Offshore",
    "Yield_10Y_CGB",
    "Term_Spread",
    "Northbound_Flow",
]


def _fetch_dr007(start_date: str, end_date: str) -> pd.DataFrame:
    """DR007 银行间质押式回购利率。使用 Shibor 1周 作为近似（相关性>0.95）。"""
    try:
        import akshare as ak
        df = ak.macro_china_shibor_all()
        if df is None or df.empty:
            return pd.DataFrame()
        # 找 1W-定价 列（Shibor 1周，与DR007相关性极高）
        col_1w = None
        for c in df.columns:
            if "1W" in str(c) or "1周" in str(c):
                col_1w = c
                break
        if col_1w is None:
            return pd.DataFrame()
        # 按位置构建结果：col 0 = 日期, col_1w = DR007
        result = pd.DataFrame()
        result["date"] = df.iloc[:, 0]
        result["DR007"] = df[col_1w]
        result["date"] = pd.to_datetime(result["date"])
        result["DR007"] = pd.to_numeric(result["DR007"], errors="coerce")
        # 按日期过滤
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        result = result[(result["date"] >= start_dt) & (result["date"] <= end_dt)]
        return result[["date", "DR007"]].dropna()
    except Exception:
        return pd.DataFrame()


def _fetch_cny_usd_offshore(start_date: str, end_date: str) -> pd.DataFrame:
    """USD/CNY 在岸人民币汇率（使用 BOC SAFE 美元汇率数据 / 100）。"""
    try:
        import akshare as ak
        df = ak.currency_boc_safe()
        if df is None or df.empty:
            return pd.DataFrame()
        # BOC SAFE 数据：col 0 = 日期, col 1 = 美元汇率（需除以100才是真实汇率）
        result = pd.DataFrame()
        result["date"] = df.iloc[:, 0]
        result["CNY_USD_Offshore"] = df.iloc[:, 1] / 100.0  # 原始数据/100 = 真实USD/CNY
        result["date"] = pd.to_datetime(result["date"])
        result["CNY_USD_Offshore"] = pd.to_numeric(result["CNY_USD_Offshore"], errors="coerce")
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        result = result[(result["date"] >= start_dt) & (result["date"] <= end_dt)]
        return result[["date", "CNY_USD_Offshore"]].dropna()
    except Exception:
        return pd.DataFrame()


def _fetch_yield_10y_cgb(start_date: str, end_date: str) -> pd.DataFrame:
    """10年期国债收益率（中债估值，来自 bond_zh_us_rate）。"""
    try:
        import akshare as ak
        df = ak.bond_zh_us_rate(start_date=start_date.replace("-", ""))
        if df is None or df.empty:
            return pd.DataFrame()
        # bond_zh_us_rate: col 0=日期, col 3=中国国债收益率10年
        result = pd.DataFrame()
        result["date"] = df.iloc[:, 0]
        result["Yield_10Y_CGB"] = df.iloc[:, 3]  # 中国国债收益率10年
        result["date"] = pd.to_datetime(result["date"])
        result["Yield_10Y_CGB"] = pd.to_numeric(result["Yield_10Y_CGB"], errors="coerce")
        end_dt = pd.to_datetime(end_date)
        result = result[result["date"] <= end_dt]
        return result[["date", "Yield_10Y_CGB"]].dropna()
    except Exception:
        return pd.DataFrame()


def _fetch_term_spread(start_date: str, end_date: str) -> pd.DataFrame:
    """期限利差 = 10Y - 2Y 国债收益率（中债估值，来自 bond_zh_us_rate）。"""
    try:
        import akshare as ak
        df = ak.bond_zh_us_rate(start_date=start_date.replace("-", ""))
        if df is None or df.empty:
            return pd.DataFrame()
        # bond_zh_us_rate: col 0=日期, col 1=中国国债收益率2年, col 3=中国国债收益率10年
        # col 5=中国国债收益率10年-2年（直接就是利差）
        result = pd.DataFrame()
        result["date"] = df.iloc[:, 0]
        result["Term_Spread"] = df.iloc[:, 5]  # 中国国债收益率10年-2年（直接利差）
        result["date"] = pd.to_datetime(result["date"])
        result["Term_Spread"] = pd.to_numeric(result["Term_Spread"], errors="coerce")
        end_dt = pd.to_datetime(end_date)
        result = result[result["date"] <= end_dt]
        return result[["date", "Term_Spread"]].dropna()
    except Exception:
        return pd.DataFrame()


def _fetch_northbound_flow(start_date: str, end_date: str) -> pd.DataFrame:
    """北向资金净流入（使用两融余额20日动量作为代理变量）。

    注：北向资金数据已于2024年8月停更，改用两融余额20日动量作为市场杠杆情绪代理。
    动量公式：Feature_Margin = (Margin_t - Margin_{t-20}) / Margin_{t-20}
    这样可将非平稳的绝对金额转化为围绕0波动的百分比平稳序列。
    """
    try:
        import akshare as ak
        # 分别获取沪深两融数据
        df_sh = ak.macro_china_market_margin_sh()
        df_sz = ak.macro_china_market_margin_sz()
        if df_sh is None or df_sz is None or df_sh.empty or df_sz.empty:
            return pd.DataFrame()
        # col 0 = 日期, col 1 = 融资余额
        df_sh = df_sh.iloc[:, [0, 1]].copy()
        df_sh.columns = ["date", "margin_sh"]
        df_sz = df_sz.iloc[:, [0, 1]].copy()
        df_sz.columns = ["date", "margin_sz"]

        # 各自先排序并计算pct_change（避免merge后位置错乱）
        for df_ in [df_sh, df_sz]:
            df_["date"] = pd.to_datetime(df_["date"])
            df_["margin"] = pd.to_numeric(df_["margin_sh"] if "margin_sh" in df_.columns else df_["margin_sz"], errors="coerce")
            df_.sort_values("date", inplace=True)
            df_["margin_pct"] = df_["margin"].pct_change(periods=20)
            df_.reset_index(drop=True, inplace=True)

        # Outer merge 合并沪深数据（取任一方的pct_change，因为同步性很高）
        df = pd.merge(df_sh[["date", "margin_pct"]], df_sz[["date", "margin_pct"]],
                      on="date", how="outer", suffixes=("_sh", "_sz"))
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)

        # 合并后取非NaN的pct_change（优先用沪市，沪市数据更完整）
        df["Northbound_Flow"] = df["margin_pct_sh"].fillna(df["margin_pct_sz"])

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]

        # 去除NaN（动量需要20天预热）和inf
        result = df[["date", "Northbound_Flow"]].dropna()
        result = result[np.isfinite(result["Northbound_Flow"])]
        return result
    except Exception:
        return pd.DataFrame()


def fetch_macro_features(
    start_date: str,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    获取并合并 5维宏观特征。

    Args:
        start_date: 数据起始日期
        end_date  : 数据截止日期

    Returns:
        DataFrame [date × macro_feature]，index=date，columns=[
            DR007, CNY_USD_Offshore, Yield_10Y_CGB, Term_Spread, Northbound_Flow
        ]
    """
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    # 分别拉取各特征
    dr007 = _fetch_dr007(start_date, end_date)
    cny_usd = _fetch_cny_usd_offshore(start_date, end_date)
    yield_10y = _fetch_yield_10y_cgb(start_date, end_date)
    term_spread = _fetch_term_spread(start_date, end_date)
    northbound = _fetch_northbound_flow(start_date, end_date)

    # 合并
    frames = [dr007, cny_usd, yield_10y, term_spread, northbound]
    frames = [f for f in frames if f is not None and not f.empty]

    if not frames:
        return pd.DataFrame()

    merged = frames[0]
    for df in frames[1:]:
        merged = pd.merge(merged, df, on="date", how="outer")

    merged = merged.sort_values("date").reset_index(drop=True)
    return merged


def compute_macro_features(
    start_date: str = "2015-01-01",
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    主入口：获取宏观特征 DataFrame，并进行简单前向填充（仅用于极少量缺失）。

    Returns:
        DataFrame [date × 5 macro features]
    """
    df = fetch_macro_features(start_date=start_date, end_date=end_date)
    if df.empty:
        return df

    df = df.set_index("date").sort_index()
    # 极少量缺失用前值填充
    df = df.ffill()
    return df


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    df = compute_macro_features(start_date="2023-01-01")
    print(df.tail(10))
