"""
Track B: quantchdb / ClickHouse ETF data fetcher
从本地 ClickHouse 数据库获取 ETF 对应数据，时间范围 2022-01-01 至今。
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import List, Dict, Optional

import pandas as pd


# 对应轨A的ETF标的
TRACK_B_ETF_CODES: List[str] = [
    "510300.SH",  # 沪深300ETF
    "512850.SH",  # 中证1000ETF
    "511010.SH",  # 国债ETF
    "518880.SH",  # 黄金ETF
    "160217.SZ",  # 商品ETF
]

START_DATE: str = "2022-01-01"


def _get_db_config() -> Dict:
    """从环境变量读取 ClickHouse 配置。"""
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", 9000)),
        "user": os.getenv("DB_USER", "default"),
        "password": os.getenv("DB_PASSWORD", ""),
        "database": os.getenv("DB_DATABASE", "etf"),
    }


def fetch_track_b(
    start_date: str = START_DATE,
    end_date: Optional[str] = None,
    db_config: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    从 quantchdb / ClickHouse 获取 ETF 周频收盘价数据。

    Args:
        start_date: 数据起始日期
        end_date: 数据截止日期，默认为今天
        db_config: ClickHouse 配置 dict，若不提供则从环境变量读取

    Returns:
        DataFrame with columns: [date, 510300.SH, 512850.SH, 511010.SH, 518880.SH, 160217.SZ]
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

    # 构建 IN 查询
    codes_str = ", ".join(f"'{c}'" for c in TRACK_B_ETF_CODES)
    sql = f"""
        SELECT
            date,
            code,
            close
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
        return pd.DataFrame()

    # 透视表：date × code → close
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"])

    pivot = df.pivot_table(index="date", columns="code", values="close")
    pivot = pivot.sort_index()

    # 统一列名（映射为轨A的标的名称）
    code_map = {
        "510300.SH": "000300.SH",
        "512850.SH": "000852.SH",
        "511010.SH": "CBA02701.CS",
        "518880.SH": "AU9999.SGE",
        "160217.SZ": "NH0100.NHF",
    }
    pivot = pivot.rename(columns=code_map)

    return pivot.reset_index()


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
