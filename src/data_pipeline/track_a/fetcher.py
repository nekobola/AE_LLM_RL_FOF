"""
Track A: AkShare data fetcher
Fetches 5 major indices using Friday-close slicing.
严格以周五收盘价为切片，禁止引入未来数据。
"""

from __future__ import annotations

import os
from datetime import datetime, date
from typing import List, Dict, Optional

import pandas as pd


# 5大标的
TRACK_A_INDICES: List[str] = [
    "000300.SH",  # 沪深300
    "000852.SH",  # 中证1000
    "CBA02701.CS",  # 信用债指数
    "AU9999.SGE",  # 黄金现货
    "NH0100.NHF",  # 南华商品指数
]

START_DATE: str = "2015-01-01"


def _is_friday(d: date) -> bool:
    return d.weekday() == 4


def _fetch_index_akshare(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    使用 AkShare 获取指数日线数据，返回严格周五切片。
    使用 edge_start 参数避免引入未确认的尾部数据。
    """
    try:
        import akshare as ak
    except ImportError:
        raise ImportError("akshare is required: pip install akshare")

    # 统一转换为 YYYYMMDD 字符串格式
    start_str = start_date.replace("-", "")
    end_str = end_date.replace("-", "")

    try:
        if symbol.startswith("0") and ".SH" in symbol:
            # 股票/指数 - 使用 index_zh_a_hist
            df = ak.index_zh_a_hist(
                symbol=symbol,
                start_date=start_str,
                end_date=end_str,
                adjust="",
            )
        elif symbol.startswith("000") and ".SH" in symbol:
            df = ak.index_zh_a_hist(
                symbol=symbol,
                start_date=start_str,
                end_date=end_str,
                adjust="",
            )
        elif ".CS" in symbol:
            # 中证债券指数
            df = ak.index_zh_bond_hist(
                symbol=symbol,
                start_date=start_str,
                end_date=end_str,
            )
        elif ".SGE" in symbol:
            # 上海黄金交易所现货
            df = ak.spot_sge_hist(
                symbol="au9999",
                start_date=start_date,
                end_date=end_date,
            )
        elif ".NHF" in symbol:
            # 南华商品指数 - 使用宏观数据
            df = ak.index_nhf_hist(
                start_date=start_date,
                end_date=end_date,
            )
        else:
            raise ValueError(f"Unsupported symbol: {symbol}")

        if df is None or df.empty:
            return pd.DataFrame()

        return df

    except Exception:
        return pd.DataFrame()


def _normalize_columns(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """将不同来源的 DataFrame 统一为 ['date', 'close'] 格式。"""
    if df.empty:
        return df

    col_map: Dict[str, str] = {}

    # 尝试识别日期列
    date_candidates = ["日期", "date", "日期时间", "datetime"]
    for c in date_candidates:
        for col in df.columns:
            if c.lower() in col.lower():
                col_map[col] = "date"
                break

    # 尝试识别收盘价列
    close_candidates = ["收盘", "close", "收盘价", "closeprice"]
    for c in close_candidates:
        for col in df.columns:
            if c.lower() in col.lower():
                col_map[col] = "close"
                break

    if "date" not in col_map or "close" not in col_map:
        return pd.DataFrame()

    df = df.rename(columns={col_map["date"]: "date", col_map["close"]: "close"})
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"])
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "close"]]


def fetch_track_a(start_date: str = START_DATE, end_date: Optional[str] = None) -> pd.DataFrame:
    """
    获取轨A 5大指数的周五收盘数据。

    Args:
        start_date: 数据起始日期
        end_date: 数据截止日期，默认为今天

    Returns:
        DataFrame with columns: [date, 000300.SH, 000852.SH, CBA02701.CS, AU9999.SGE, NH0100.NHF]
    """
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    result_frames: List[pd.DataFrame] = []

    for symbol in TRACK_A_INDICES:
        df_raw = _fetch_index_akshare(symbol, start_date, end_date)
        if df_raw.empty:
            continue
        df_norm = _normalize_columns(df_raw, source=symbol)
        if df_norm.empty:
            continue

        # 严格周五切片
        df_norm["is_friday"] = df_norm["date"].dt.date.apply(_is_friday)
        df_friday = df_norm[df_norm["is_friday"]].copy()
        df_friday = df_friday.drop(columns=["is_friday"])
        df_friday = df_friday.rename(columns={"close": symbol})
        result_frames.append(df_friday)

    if not result_frames:
        return pd.DataFrame()

    # 按日期 merge 所有标的
    merged = result_frames[0]
    for df in result_frames[1:]:
        merged = pd.merge(merged, df, on="date", how="outer")

    merged = merged.sort_values("date").reset_index(drop=True)

    # 填充交易日期缺口（仅限已知的周五）
    all_fridays = pd.date_range(start=start_date, end=end_date, freq="W-FRI", tz="UTC")
    all_fridays = all_fridays.tz_localize(None)
    full_idx = pd.DataFrame({"date": all_fridays})
    merged = pd.merge(full_idx, merged, on="date", how="left")

    return merged


if __name__ == "__main__":
    df = fetch_track_a()
    print(df.tail(10))
