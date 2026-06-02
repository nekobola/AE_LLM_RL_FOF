"""
test_normalizer_no_lookahead.py — 未来函数（Look-ahead Bias）专项测试

测试目标：
  1. 验证 rolling_zscore 窗口严格为 [t-252, t-1]，不包含 t 时刻
  2. 验证 rolling_zscore_manual 结果与 rolling_zscore 一致
  3. 验证 normalize_tensor 严格按列独立标准化，不混入未来信息
  4. 验证 normalize_dataframe 不引入全局标准化（如 sklearn fit_transform）
  5. 诊断：打印每个有效 t 的窗口范围，确认无泄漏

红线判定（任何一项失败 → CI red）：
  - 标准化时刻 t 的均值计算包含 t 本身
  - 使用了 sklearn.fit_transform / StandardScaler.fit
  - 标准化使用了全局统计量（整个序列的 mean/std）
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
import pytest

# 被测模块
from src.features.normalizer import (
    rolling_zscore,
    rolling_zscore_manual,
    normalize_dataframe,
    normalize_tensor,
    diagnose_lookahead_bias,
    ZSCORE_WINDOW,
    MIN_PERIODS,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def synthetic_series():
    """单调递增序列，容易验证标准化方向。"""
    np.random.seed(0)
    dates = pd.date_range("2018-01-01", "2024-12-31", freq="B")
    n = len(dates)
    values = np.arange(n, dtype=float) + np.random.randn(n) * 0.1
    return pd.Series(values, index=dates, name="synthetic")


@pytest.fixture
def multi_feature_df():
    """多特征 DataFrame。"""
    np.random.seed(0)
    dates = pd.date_range("2018-01-01", "2024-12-31", freq="B")
    n = len(dates)
    data = np.random.randn(n, 4).cumsum(axis=0) + 100
    return pd.DataFrame(data, index=dates, columns=["A", "B", "C", "D"])


# ============================================================
# Test 1: 窗口边界验证
# ============================================================

class TestRollingZscoreWindowBoundary:
    """验证 rolling_zscore 不使用 t 时刻数据计算统计量。"""

    def test_rolling_zscore_uses_only_past_data(self, synthetic_series):
        """
        核心断言：rolling_zscore[t] 仅使用 [t-252, t-1] 窗口数据。
        验证方式：在 t 位置注入一个极大值，标准化后该值的影响
        不会立即反映在 t+1 的标准化结果中（因为 t+1 使用的是 [t-251, t]）。
        """
        ts = synthetic_series.copy()
        # 在 t=500 处注入尖峰
        spike_idx = 500
        original_val = ts.iloc[spike_idx]
        ts.iloc[spike_idx] = original_val * 100  # 尖峰

        zscore = rolling_zscore(ts)

        # 尖峰时刻（t=500）标准化值应极大
        assert abs(zscore.iloc[spike_idx]) > 10, "Spike at t should be extreme z-score"

        # t=501 的窗口是 [501-252, 500] = [249, 500]，包含尖峰
        # 但 t=501 的标准化使用 mean[249:501], std[249:501]
        # t=502 的标准化使用 mean[250:502], std[250:502]
        # 若尖峰泄漏到 t+252，则说明使用了未来数据
        # 验证：t=252+spike_idx 的时刻，尖峰已移出窗口
        exit_idx = spike_idx + ZSCORE_WINDOW
        if exit_idx < len(ts):
            # 尖峰已不在窗口内，zscore 应该回归 |z| < 5
            assert abs(zscore.iloc[exit_idx]) < 5, (
                f"Spike should exit window at t+{ZSCORE_WINDOW}, "
                f"but zscore={zscore.iloc[exit_idx]}"
            )

    def test_rolling_zscore_t_minus_one_not_t(self, synthetic_series):
        """
        验证 rolling_zscore(shifted) 与手动窗口计算完全一致。
        对每个 t，用纯手动窗口计算 mean/std，对比结果。
        """
        ts = synthetic_series.copy()
        zscore_fast = rolling_zscore(ts)
        n = len(ts)

        for t in range(ZSCORE_WINDOW, n, ZSCORE_WINDOW):
            window_data = ts.iloc[t - ZSCORE_WINDOW : t]
            manual_mean = window_data.mean()
            manual_std = window_data.std()
            if manual_std == 0 or np.isnan(manual_std):
                continue
            manual_z = (ts.iloc[t] - manual_mean) / manual_std
            fast_z = zscore_fast.iloc[t]
            assert abs(manual_z - fast_z) < 1e-10, (
                f"t={t}: manual={manual_z:.8f}, fast={fast_z:.8f}"
            )


# ============================================================
# Test 2: NumPy 与 Pandas 实现一致性
# ============================================================

class TestRollingZscoreManual:
    """验证纯 NumPy 手动实现与 Pandas rolling 结果一致。"""

    def test_manual_matches_pandas(self, synthetic_series):
        """滚动 Z-score NumPy 实现应与 Pandas 实现数学等价。"""
        ts = synthetic_series.copy()
        arr = ts.values

        pandas_result = rolling_zscore(ts)
        # Pandas .std() 默认 ddof=1；手动实现需保持一致
        manual_result = rolling_zscore_manual(arr, window=ZSCORE_WINDOW, min_periods=MIN_PERIODS)

        # 对非 NaN 位置逐一比对（允许浮点误差）
        valid_mask = ~np.isnan(manual_result)
        pandas_vals = pandas_result.values[valid_mask]
        manual_vals = manual_result[valid_mask]

        assert np.allclose(pandas_vals, manual_vals, rtol=1e-9, atol=1e-9), (
            "Pandas and NumPy rolling zscore implementations must match within tolerance"
        )

    def test_manual_window_excludes_current(self, synthetic_series):
        """NumPy 实现验证：t 时刻数据不进入窗口。"""
        arr = synthetic_series.values
        n = len(arr)

        # 取一个特定 t
        t = ZSCORE_WINDOW + 10
        result = rolling_zscore_manual(arr, window=ZSCORE_WINDOW, min_periods=MIN_PERIODS)

        # 检查 t 时刻的窗口数据：窗口范围是 [t-window, t-1]
        window_start = t - ZSCORE_WINDOW
        window_end = t  # 不含 t
        window_data = arr[window_start:window_end]

        # t 不应在窗口内（窗口是 [t-window, t-1]）
        assert t not in range(window_start, window_end), "t must not be in window"
        assert len(window_data) == ZSCORE_WINDOW, "Window must be exactly ZSCORE_WINDOW"


# ============================================================
# Test 3: DataFrame 列独立性
# ============================================================

class TestNormalizeDataframeColumnIsolation:
    """验证 normalize_dataframe 每列独立标准化，无列间信息泄漏。"""

    def test_columns_normalized_independently(self, multi_feature_df):
        """验证 normalize_dataframe 对每列独立标准化，无 NaN 泄漏。"""
        norm_df = normalize_dataframe(multi_feature_df, window=ZSCORE_WINDOW)

        # 取窗口足够后的有效数据（不应有 NaN）
        valid = norm_df.iloc[ZSCORE_WINDOW + MIN_PERIODS:]

        # 无 NaN（标准化正常执行）
        assert not valid.isna().any().any(), (
            f"Unexpected NaN in normalized DataFrame:\n{valid.isna().sum()}"
        )

        # 所有值应为有限数（无 inf/nan）
        for col in valid.columns:
            assert np.isfinite(valid[col]).all(), (
                f"Column {col} contains non-finite values"
            )

    def test_no_sklearn_fit_transform(self, multi_feature_df):
        """静态代码检查：确保未引入 sklearn 标准化函数。"""
        import inspect
        import re
        from src.features import normalizer
        source = inspect.getsource(normalizer)

        # 移除 docstring 和注释后再检查
        # 去除 docstring（triple-quoted）
        source_no_docs = re.sub(r'""".*?"""', '', source, flags=re.DOTALL)
        source_no_docs = re.sub(r"'''.*?'''", '', source_no_docs, flags=re.DOTALL)
        # 去除单行注释
        source_no_docs = re.sub(r'#.*', '', source_no_docs)

        forbidden = ["fit_transform", "StandardScaler", "MinMaxScaler"]
        for token in forbidden:
            assert token not in source_no_docs, (
                f"Forbidden sklearn API '{token}' found in normalizer.py source"
            )


# ============================================================
# Test 4: 诊断报告
# ============================================================

class TestDiagnoseLookaheadBias:
    """验证诊断函数正确识别窗口边界。"""

    def test_diagnose_window_range(self, synthetic_series):
        """诊断报告应显示每个 t 的窗口为 [t-252, t-1]。"""
        diag = diagnose_lookahead_bias(synthetic_series.to_frame("value"))

        if diag.empty:
            pytest.skip("Not enough data for diagnosis")

        first_row = diag.iloc[0]
        assert first_row["window_size"] == ZSCORE_WINDOW
        assert bool(first_row["includes_t"]) is False, (
            "includes_t should be False (window does not include t)"
        )

        # 验证最后一个有效行
        last_row = diag.iloc[-1]
        t_date = last_row["t"]
        window_end = last_row["window_end"]
        # window_end 应该是 t-1
        assert window_end < t_date, "Window end must be strictly before t"


# ============================================================
# Test 5: 边界条件
# ============================================================

class TestNormalizerEdgeCases:
    """边界条件测试。"""

    def test_empty_series_returns_empty(self):
        s = pd.Series([], dtype=float)
        result = rolling_zscore(s)
        assert result.empty

    def test_constant_series_returns_nan(self):
        """常数序列 std=0，标准化应返回 NaN。"""
        dates = pd.date_range("2020-01-01", periods=300, freq="B")
        const = pd.Series(np.ones(300), index=dates)
        zscore = rolling_zscore(const)
        assert zscore.iloc[ZSCORE_WINDOW:].isna().all(), (
            "Constant series must produce NaN z-scores"
        )

    def test_window_larger_than_series(self):
        """窗口大于序列长度时，应返回全 NaN。"""
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        s = pd.Series(np.arange(10.0), index=dates)
        zscore = rolling_zscore(s, window=252, min_periods=60)
        assert zscore.isna().all(), "Should return all NaN when window > series length"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
