"""
Track B: quantchdb / ClickHouse ETF data fetcher
从本地 ClickHouse 数据库获取 ETF 对应数据，时间范围 2022-01-01 至今。
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import List, Dict, Optional, Union

import pandas as pd


# ClickHouse etf_day 中的 ETF 代码（无交易所后缀）
TRACK_B_ETF_CODES: List[str] = [
    "510300",  # 沪深300ETF
    "159919",  # 中证1000ETF (原512850，数据截止2020-12-08)
    "511010",  # 国债ETF
    "518880",  # 黄金ETF
    "159985",  # 商品/另类ETF (原160217不存在于ClickHouse)
]

# ClickHouse 代码 -> 资产代码映射（用于 compute_asset_features 的列名）
CODE_MAP: Dict[str, str] = {
    "510300": "000300.SH",
    "159919": "000852.SH",
    "511010": "CBA02701.CS",
    "518880": "AU9999.SGE",
    "159985": "NH0100.NHF",
}

START_DATE: str = "2005-01-01"  # ClickHouse 数据从 2005-02-23 开始


def _get_db_config() -> Dict:
    """从环境变量读取 ClickHouse (quantchdb) 配置。"""
    return {
        "host": os.getenv("CHDB_HOST", "localhost"),
        "port": int(os.getenv("CHDB_PORT", 20108)),
        "user": os.getenv("CHDB_USER", "default"),
        "password": os.getenv("CHDB_PASSWORD", ""),
        "database": os.getenv("CHDB_DATABASE", "etf"),
    }


def fetch_track_b(
    start_date: str = START_DATE,
    end_date: Optional[str] = None,
    db_config: Optional[Dict] = None,
    columns: List[str] = ["close"],
) -> Union[pd.DataFrame, pd.Series]:
    """
    从 quantchdb / ClickHouse 获取 ETF 周频收盘价数据。

    Args:
        start_date: 数据起始日期
        end_date: 数据截止日期，默认为今天
        db_config: ClickHouse 配置 dict，若不提供则从环境变量读取
        columns: 要查询的列名列表，支持 "open", "close", "adj_factor"。
                 默认为 ["close"]（向后兼容）。
                 当 columns=["close"] 时返回 Series (index=date, columns=code)
                 当 columns 多于一项时返回 DataFrame (index=date, columns=[code__col,...])

    Returns:
        单列时：Series with columns: [510300, 159919, 511010, 518880, 159985]
               (经CODE_MAP映射为资产代码: 000300.SH, 000852.SH, CBA02701.CS, AU9999.SGE, NH0100.NHF)
        多列时：DataFrame with MultiIndex columns (code x column), index=date
               列名格式: code__open, code__close, code__adj_factor
    """
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    if db_config is None:
        db_config = _get_db_config()

    # 检查是否可连接
    try:
        from quantchdb import ClickHouseDatabase
    except ImportError:
        raise ImportError("quantchdb is required: pip install quantchdb==0.1.11")

    try:
        db = ClickHouseDatabase(config=db_config, terminal_log=False, file_log=False)
    except Exception as e:
        raise ConnectionError(f"Failed to connect to ClickHouse: {e}")

    # 验证 columns
    valid_cols = {"open", "close", "adj_factor"}
    for col in columns:
        if col not in valid_cols:
            raise ValueError(f"Invalid column: {col}. Must be one of {valid_cols}")

    # 构建 IN 查询
    codes_str = ", ".join(f"'{c}'" for c in TRACK_B_ETF_CODES)
    cols_str = ", ".join(columns)
    sql = f"""
        SELECT
            date,
            code,
            {cols_str}
        FROM etf.etf_day
        WHERE date >= '{start_date}'
          AND date <= '{end_date}'
          AND code IN ({codes_str})
        ORDER BY date ASC
    """

    try:
        df = db.fetch(sql)
    except Exception as e:
        raise RuntimeError(f"ClickHouse query failed: {e}")

    if df is None or df.empty:
        return pd.DataFrame() if len(columns) > 1 else pd.Series(dtype=float)

    # 转换类型
    df["date"] = pd.to_datetime(df["date"])
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date"] + columns)

    # 单列时向后兼容：返回 Series (index=date, columns=code)
    if len(columns) == 1:
        col = columns[0]
        pivot = df.pivot_table(index="date", columns="code", values=col)
        pivot = pivot.sort_index()
        # 统一列名（映射为资产代码）
        pivot = pivot.rename(columns=CODE_MAP)
        # 返回 Series，保留原有行为
        return pivot.iloc[:, 0] if pivot.shape[1] == 1 else pivot

    # 多列时：返回 DataFrame (index=date, columns=[code__open, code__close, code__adj_factor])
    records = []
    for _, row in df.iterrows():
        date_val = row["date"]
        code_val = row["code"]
        for col in columns:
            records.append({
                "date": date_val,
                "code": CODE_MAP.get(code_val, code_val),
                "column": col,
                "value": row[col],
            })
    long_df = pd.DataFrame(records)
    wide_df = long_df.pivot_table(
        index="date", columns=["code", "column"], values="value"
    )
    wide_df = wide_df.sort_index()
    # 列名格式: code__column
    wide_df.columns = [f"{code}__{col}" for code, col in wide_df.columns]
    return wide_df


def fetch_track_b_safe(
    start_date: str = START_DATE,
    end_date: Optional[str] = None,
    db_config: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    安全版本：数据获取失败时返回空 DataFrame，不抛出异常。
    """
    try:
        return fetch_track_b(start_date=start_date, end_date=end_date, db_config=db_config)
    except Exception as e:
        import warnings
        warnings.warn(f"Track B data fetch failed: {e}", RuntimeWarning)
        return pd.DataFrame()


if __name__ == "__main__":
    df = fetch_track_b_safe()
    print(df.tail(10))
