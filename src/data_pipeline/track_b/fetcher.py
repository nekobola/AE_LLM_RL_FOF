"""
Track B: quantchdb / ClickHouse ETF data fetcher
从本地 ClickHouse 数据库获取 ETF 对应数据，时间范围 2022-01-01 至今。
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import List, Dict, Optional, Union

import pandas as pd

log = logging.getLogger(__name__)


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


# ── Stage 7c: 8-ETF fetcher ──────────────────────────────────────────
# 8 ETF codes, no CODE_MAP (use raw Symbol)
N_ETF_CODES: List[str] = [
    "511010",  # 国债 ETF
    "518880",  # 黄金 ETF
    "511020",  # 信用债 ETF
    "159985",  # 商品 ETF
    "512100",  # 中证1000 ETF
    "515050",  # 红利低波 ETF
    "159915",  # 创业板 ETF (Phase 2 替换 159919, 行业分散 + 更高 Sharpe)
    "510300",  # 沪深300 ETF
]


def fetch_n_etf(
    start_date: str,
    end_date: Optional[str] = None,
    db_config: Optional[Dict] = None,
    columns: List[str] = ["open", "close", "adj_factor"],
) -> pd.DataFrame:
    """Fetch 8-ETF OHLC + adj_factor from UNION of etf.etf_daily + etf.etf_day.

    - etf.etf_daily: 2021-01 ~ 2025-10-21 (10024 rows for 8 ETFs)
    - etf.etf_day:   2025-10-22 ~ 2026-04-30 (~800 rows for 8 ETFs)
    - Combined: full 5+ years coverage.
    """
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    if db_config is None:
        db_config = _get_db_config()

    try:
        from quantchdb import ClickHouseDatabase
    except ImportError:
        raise ImportError("quantchdb is required")

    db = ClickHouseDatabase(config=db_config, terminal_log=False, file_log=False)

    codes_str = ", ".join(f"'{c}'" for c in N_ETF_CODES)
    col_map_daily = {
        "open": "OpenPrice",
        "close": "ClosePrice",
        "adj_factor": "LatestClosePrice",
    }
    # etf_daily 用 AS 别名统一, etf_day 用原列名 (lower-case open/close/adj_factor)
    cols_str_daily = ", ".join(col_map_daily.get(c, c) for c in columns)
    cols_str_day = ", ".join(columns)  # etf_day columns already match

    # Source 1: etf.etf_daily (2021-01 ~ 2025-10-21, best backfill)
    sql_daily = f"""
        SELECT
            TradingDate AS date,
            Symbol AS code,
            {cols_str_daily}
        FROM etf.etf_daily
        WHERE TradingDate >= '{start_date}'
          AND TradingDate <= '{end_date}'
          AND Symbol IN ({codes_str})
    """
    # Source 2: etf.etf_day (2025-10-22 ~ 2026+, has 2026 data, lower-case cols)
    sql_day = f"""
        SELECT
            date AS date,
            code AS code,
            {cols_str_day}
        FROM etf.etf_day
        WHERE date >= '{start_date}'
          AND date <= '{end_date}'
          AND code IN ({codes_str})
    """
    dfs = []
    for sql, label in [(sql_daily, 'etf_daily'), (sql_day, 'etf_day')]:
        try:
            df_part = db.fetch(sql)
            if df_part is not None and not df_part.empty:
                dfs.append(df_part)
        except Exception as e:
            log.warning(f"fetch_n_etf: {label} query failed: {e}")

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    # Rename etf_daily cols (OpenPrice, ClosePrice, LatestClosePrice) to lower-case to match etf_day
    rename_map = {
        "OpenPrice": "open",
        "ClosePrice": "close",
        "LatestClosePrice": "adj_factor",
    }
    df = df.rename(columns=rename_map)
    # Now cols may be duplicated. Coalesce each col by combining first non-NaN across duplicates
    if df.columns.duplicated().any():
        # Take the first occurrence of each col name (etf_daily 主源, 已在前)
        df = df.T.groupby(lambda x: x).first().T
    for col in ["open", "close", "adj_factor"]:
        if col in df.columns:
            series = df[col]
            if hasattr(series, 'ndim') and series.ndim == 1:
                df[col] = pd.to_numeric(series, errors="coerce")
    drop_subset = ["date", "code"] + [c for c in ["open", "close", "adj_factor"] if c in df.columns]
    df = df.dropna(subset=drop_subset)
    df = df.drop_duplicates(subset=["date", "code"], keep="first")
    df = df.sort_values(["date", "code"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = fetch_track_b_safe()
    print(df.tail(10))
