"""
macro_features.py — 5维宏观特征张量

特征列表：
  1. DR007          : 银行间7天质押式回购利率（盘中走势）
  2. CNY_USD_Offshore: 离岸人民币汇率
  3. Yield_10Y_CGB  : 10年期国债收益率
  4. Term_Spread    : 10Y-1Y 国债期限利差
  5. Northbound_Flow: 北向资金日度净流入

数据来源：
  - AkShare 宏观数据接口
  - 北向资金: stock_board_em_hsgt_north_history（沪深港通历史）
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
    """DR007 银行间质押式回购利率。"""
    try:
        import akshare as ak
        df = ak.currency_bond_zh(start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return pd.DataFrame()
        # 找 DR007 相关列
        dr_col = None
        for c in df.columns:
            if "DR007" in str(c) or "7天" in str(c) or "质押式" in str(c):
                dr_col = c
                break
        if dr_col is None:
            return pd.DataFrame()
        df = df.rename(columns={dr_col: "DR007", "日期": "date"})
        df["date"] = pd.to_datetime(df["date"])
        df["DR007"] = pd.to_numeric(df["DR007"], errors="coerce")
        return df[["date", "DR007"]].dropna()
    except Exception:
        return pd.DataFrame()


def _fetch_cny_usd_offshore(start_date: str, end_date: str) -> pd.DataFrame:
    """离岸人民币汇率 CNH USD。"""
    try:
        import akshare as ak
        df = ak.currency_usdkhq_exchange_rate(start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return pd.DataFrame()
        cny_col = None
        date_col = None
        for c in df.columns:
            lc = str(c).lower()
            if "cny" in lc or "离岸" in str(c) or "usdcnh" in lc:
                cny_col = c
            if "date" in lc or "日期" in str(c) or "时间" in str(c):
                date_col = c
        if cny_col is None or date_col is None:
            return pd.DataFrame()
        df = df.rename(columns={date_col: "date", cny_col: "CNY_USD_Offshore"})
        df["date"] = pd.to_datetime(df["date"])
        df["CNY_USD_Offshore"] = pd.to_numeric(df["CNY_USD_Offshore"], errors="coerce")
        return df[["date", "CNY_USD_Offshore"]].dropna()
    except Exception:
        return pd.DataFrame()


def _fetch_yield_10y_cgb(start_date: str, end_date: str) -> pd.DataFrame:
    """10年期国债收益率。"""
    try:
        import akshare as ak
        df = ak.bond_zh_us_rate(start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return pd.DataFrame()
        # 找中国/国债10年相关列
        col_10y = None
        date_col = None
        for c in df.columns:
            if "10年" in str(c) or "CN10Y" in str(c) or "中国" in str(c):
                col_10y = c
            if "日期" in str(c) or "date" in str(c).lower():
                date_col = c
        if col_10y is None or date_col is None:
            return pd.DataFrame()
        df = df.rename(columns={date_col: "date", col_10y: "Yield_10Y_CGB"})
        df["date"] = pd.to_datetime(df["date"])
        df["Yield_10Y_CGB"] = pd.to_numeric(df["Yield_10Y_CGB"], errors="coerce")
        return df[["date", "Yield_10Y_CGB"]].dropna()
    except Exception:
        return pd.DataFrame()


def _fetch_term_spread(start_date: str, end_date: str) -> pd.DataFrame:
    """期限利差 = 10Y - 1Y 国债收益率。"""
    try:
        import akshare as ak
        df = ak.bond_zh_us_rate(start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return pd.DataFrame()
        col_10y = col_1y = date_col = None
        for c in df.columns:
            if "10年" in str(c) or "CN10Y" in str(c):
                col_10y = c
            if "1年" in str(c) or "CN1Y" in str(c):
                col_1y = c
            if "日期" in str(c) or "date" in str(c).lower():
                date_col = c
        if col_10y is None or col_1y is None or date_col is None:
            return pd.DataFrame()
        df = df.rename(columns={date_col: "date"})
        df["date"] = pd.to_datetime(df["date"])
        df["Yield_10Y"] = pd.to_numeric(df[col_10y], errors="coerce")
        df["Yield_1Y"] = pd.to_numeric(df[col_1y], errors="coerce")
        df["Term_Spread"] = df["Yield_10Y"] - df["Yield_1Y"]
        return df[["date", "Term_Spread"]].dropna()
    except Exception:
        return pd.DataFrame()


def _fetch_northbound_flow(start_date: str, end_date: str) -> pd.DataFrame:
    """北向资金日度净流入。"""
    try:
        import akshare as ak
        df = ak.stock_board_em_hsgt_north_history(
            symbol="沪深港通北向",
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
        )
        if df is None or df.empty:
            return pd.DataFrame()
        date_col = flow_col = None
        for c in df.columns:
            if "日期" in str(c) or "date" in str(c).lower():
                date_col = c
            if "北向" in str(c) or "净买入" in str(c) or "北上" in str(c):
                flow_col = c
        if date_col is None or flow_col is None:
            return pd.DataFrame()
        df = df.rename(columns={date_col: "date", flow_col: "Northbound_Flow"})
        df["date"] = pd.to_datetime(df["date"])
        df["Northbound_Flow"] = pd.to_numeric(df["Northbound_Flow"], errors="coerce")
        return df[["date", "Northbound_Flow"]].dropna()
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
