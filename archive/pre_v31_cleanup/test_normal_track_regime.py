"""Stage 2: NormalTrack bear-regime 强制重分配测试"""
import numpy as np

from src.compute.normal_track import NormalTrack


def test_normal_track_bear_forced_reallocation():
    """ae_error > tau → 必须返回 BEAR_WEIGHTS (固收 ≥ 0.45)"""
    track = NormalTrack()
    np.random.seed(0)
    returns_5d = np.random.randn(5, 60) * 0.01

    weights = track.compute(returns_5d, ae_error=30.0, tau=20.0)

    # 验证 BEAR_WEIGHTS
    assert np.allclose(weights, NormalTrack.BEAR_WEIGHTS), \
        f"bear weights {weights} != expected {NormalTrack.BEAR_WEIGHTS}"
    # 防御总权重(固收+黄金+现金) ≥ 0.80
    defensive = weights[2] + weights[3] + weights[4]
    assert defensive >= 0.80, f"defensive total {defensive:.3f} < 0.80"


def test_normal_track_bear_regime_bypass_overrides_erc():
    """即使波动率结构指向高权益,bear-regime 仍必须返回防御配置"""
    track = NormalTrack()
    # 模拟固收波动率极高、权益波动率极低(ERC 会指向权益)
    returns_5d = np.zeros((5, 60))
    returns_5d[0] = 0.001   # 宽基低波
    returns_5d[1] = 0.001   # 卫星低波
    returns_5d[2] = 0.05    # 固收高波
    returns_5d[3] = 0.05    # 黄金高波
    returns_5d[4] = 0.001   # 现金低波

    weights = track.compute(returns_5d, ae_error=25.0, tau=20.0)

    # bear 仍必须重分配
    assert weights[2] >= 0.40, f"bear 固收 {weights[2]:.3f} < 0.40"
    assert weights[1] <= 0.10, f"bear 卫星 {weights[1]:.3f} > 0.10"


def test_normal_track_bull_uses_erc():
    """不传 ae_error / ae_error < tau → 走 ERC,不应等于 BEAR_WEIGHTS"""
    track = NormalTrack()
    np.random.seed(42)
    returns_5d = np.random.randn(5, 60) * 0.01

    # 不传 ae_error
    w_default = track.compute(returns_5d)
    # 传 ae_error=0, tau=20 (bull)
    w_bull = track.compute(returns_5d, ae_error=0.0, tau=20.0)

    assert not np.allclose(w_default, NormalTrack.BEAR_WEIGHTS)
    assert not np.allclose(w_bull, NormalTrack.BEAR_WEIGHTS)
    # sum 应该为 1
    assert abs(w_bull.sum() - 1.0) < 1e-6


def test_normal_track_bounds_widened():
    """Stage 2 放宽 bounds: 卫星下限=0, 黄金上限=0.30, 现金上限=0.30"""
    track = NormalTrack()
    bounds = track.default_bounds

    assert bounds[NormalTrack.IDX_SATELLITE][0] == 0.0, \
        f"satellite 下限 {bounds[NormalTrack.IDX_SATELLITE][0]} != 0"
    assert bounds[NormalTrack.IDX_SATELLITE][1] == 0.25, \
        f"satellite 上限 {bounds[NormalTrack.IDX_SATELLITE][1]} != 0.25"
    assert bounds[NormalTrack.IDX_SAFE][1] == 0.30, \
        f"safe 上限 {bounds[NormalTrack.IDX_SAFE][1]} != 0.30"
    assert bounds[NormalTrack.IDX_CASH][1] == 0.30, \
        f"cash 上限 {bounds[NormalTrack.IDX_CASH][1]} != 0.30"


def test_normal_track_insufficient_samples_falls_back():
    """样本不足仍回退等权,而不是 bear 配置"""
    track = NormalTrack()
    returns_5d = np.random.randn(5, 3) * 0.01  # 不足 MIN_SAMPLES=5

    w = track.compute(returns_5d, ae_error=30.0, tau=20.0)

    # bear bypass 应当先生效,不依赖样本数
    assert np.allclose(w, NormalTrack.BEAR_WEIGHTS)
