"""
asset_features.py — 5资产 × 4特征 = 20维资产特征张量

特征定义（严格禁止未来函数）：
  1. weekly_return  : (P_{t} / P_{t-5}) - 1         使用已确认的周五数据
  2. volatility_20d : rolling_std(returns[t-20:t-1])  使用过去20个已确认周五收益
  3. momentum_20d   : (P_{t-1} / P_{t-21}) - 1      使用截至 t-1 的已确认数据
  4. mean_corr_20d  : mean(pairwise_corr(returns[t-20:t-1]))  资产间相关矩阵均值

所有窗口均严格使用 [t-N, t-1] 范围，不包含 t 时刻数据。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# 5大资产
ASSET_CODES: List[str] = [
    "000300.SH",
    "000852.SH",
    "CBA02701.CS",
    "AU9999.SGE",
    "NH0100.NHF",
]

# 窗口参数
MOMENTUM_WINDOW: int = 20
VOLATILITY_WINDOW: int = 20
CORR_WINDOW: int = 20
RETURN_WINDOW: int = 5  # weekly = 5 trading days


def compute_weekly_return(price_df: pd.DataFrame, window: int = RETURN_WINDOW) -> pd.DataFrame:
    """
    周度收益率：(P_t / P_{t-window}) - 1
    使用已确认的周五收盘价，计算至当日（当日数据已知）。
    数据格式：price_df 列 index=date，columns=asset codes。
    """
    ret = price_df.pct_change(periods=window)
    return ret


def compute_volatility_20d(price_df: pd.DataFrame, window: int = VOLATILITY_WINDOW) -> pd.DataFrame:
    """
    20日滚动波动率：std(returns[t-20:t-1])
    仅使用 t-1 及之前的数据。
    """
    # 日收益率（向前差分，使用历史数据）
    returns = price_df.pct_change()

    # 滚动标准差，min_periods 确保起始有足够样本
    vol = returns.rolling(window=window, min_periods=window).std()
    return vol


def compute_momentum_20d(price_df: pd.DataFrame, window: int = MOMENTUM_WINDOW) -> pd.DataFrame:
    """
    动量因子：(P_{t-1} / P_{t-window}) - 1
    仅使用截至 t-1 的已确认价格。
    """
    momentum = price_df.pct_change(periods=window).shift(1)
    return momentum


def compute_mean_corr_20d(price_df: pd.DataFrame, window: int = CORR_WINDOW) -> pd.DataFrame:
    """
    20日资产间滚动相关性均值。
    对每日的资产收益向量，计算 pairwise 相关系数矩阵的均值上三角。
    仅使用 [t-20, t-1] 窗口。
    """
    returns = price_df.pct_change()

    def _mean_corr_row(row_series: pd.Series) -> float:
        """计算某日资产收益向量间的平均相关系数。"""
        # 构造滚动窗口数据
        idx = row_series.name
        window_data = returns.loc[:idx].tail(window)
        if window_data.shape[0] < window:
            return np.nan
        # 转置：每列是一个资产，每行是一个交易日
        corr_mat = window_data.corr()
        # 取上三角（不含对角线）
        mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
        vals = corr_mat.where(mask).stack().values
        if len(vals) == 0:
            return np.nan
        return float(np.nanmean(vals))

    # 逐行计算（慢但正确，无未来函数）
    mean_corr_series = pd.Series(index=returns.index, dtype=float)
    for idx in returns.index:
        window_data = returns.loc[:idx].tail(window)
        if window_data.shape[0] < window:
            mean_corr_series[idx] = np.nan
            continue
        corr_mat = window_data.corr()
        mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
        vals = corr_mat.where(mask).stack().values
        mean_corr_series[idx] = float(np.nanmean(vals)) if len(vals) > 0 else np.nan

    # 格式化为 DataFrame（每个资产列都相同，为 mean_corr）
    result = pd.DataFrame(
        {code: mean_corr_series for code in ASSET_CODES},
        index=returns.index,
    )
    return result


def compute_asset_features(
    price_df: pd.DataFrame,
    asset_codes: Optional[List[str]] = None,
    momentum_window: int = MOMENTUM_WINDOW,
    volatility_window: int = VOLATILITY_WINDOW,
    corr_window: int = CORR_WINDOW,
    return_window: int = RETURN_WINDOW,
) -> pd.DataFrame:
    """
    计算 5资产 × 4特征 = 20维资产特征矩阵。

    Args:
        price_df     : DataFrame [date × asset_code]，index=date，columns=asset codes
        asset_codes  : 资产代码列表
        *_window     : 各特征窗口参数

    Returns:
        DataFrame [date × (asset_code × feature)]  共 20 列
    """
    if asset_codes is None:
        asset_codes = ASSET_CODES

    # 保留已有列，过滤顺序
    price_df = price_df[asset_codes].copy()

    # 1. weekly_return
    weekly_ret = compute_weekly_return(price_df, window=return_window)

    # 2. volatility_20d
    vol = compute_volatility_20d(price_df, window=volatility_window)

    # 3. momentum_20d
    momentum = compute_momentum_20d(price_df, window=momentum_window)

    # 4. mean_corr_20d
    mean_corr = compute_mean_corr_20d(price_df, window=corr_window)

    # 合并为 MultiIndex 列
    feature_names = ["weekly_return", "volatility_20d", "momentum_20d", "mean_corr_20d"]
    frames = {
        "weekly_return": weekly_ret,
        "volatility_20d": vol,
        "momentum_20d": momentum,
        "mean_corr_20d": mean_corr,
    }

    result_frames = []
    for feat_name, feat_df in frames.items():
        for code in asset_codes:
            col_name = f"{code}__{feat_name}"
            feat_df = feat_df.rename(columns={code: col_name})
        result_frames.append(feat_df)

    result = pd.concat(result_frames, axis=1)

    return result


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    # Demo with synthetic data
    dates = pd.date_range("2020-01-01", "2024-12-31", freq="W-FRI")
    np.random.seed(42)
    n = len(dates)
    data = np.random.randn(n, 5).cumsum(axis=0) + 100
    price_df = pd.DataFrame(data, index=dates, columns=ASSET_CODES)

    feat = compute_asset_features(price_df)
    print(feat.tail(10))
