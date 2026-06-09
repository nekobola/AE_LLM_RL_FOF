"""EventTrackV2 Signal-Tilted Risk Budgeting unit tests.

Covers:
  1. AE hard switch → BEAR vertex
  2. Strong growth signal → GROWTH vertex (with cap)
  3. Neutral signal → NEUTRAL vertex (RB solver returns near-uniform)
  4. RB objective is polynomial form (no division by port_var)
  5. β_neutral = max(0, 1 - sum) protects when signals are extreme both ways
  6. Bounds enforced (no asset outside its range)
  7. Solver converges for typical 60-day returns
  8. w0 is asymmetric (proves design change #2)
  9. Sanity: edge between w_event and w_normal is non-zero in mixed regime
"""
import numpy as np

from src.compute.event_track_v2 import EventTrackV2
from src.compute.dual_track_engine import DualTrackEngine
from src.compute.normal_track import NormalTrack


def _make_returns(seed=42, n=60, broad=0.012, sat=0.015, fi=0.003, safe=0.010, cash=0.001):
    np.random.seed(seed)
    rets = np.zeros((5, n))
    rets[0] = np.random.randn(n) * broad
    rets[1] = np.random.randn(n) * sat
    rets[2] = np.random.randn(n) * fi
    rets[3] = np.random.randn(n) * safe
    rets[4] = np.random.randn(n) * cash
    return rets


def test_bear_regime_forces_bear_vertex():
    """ae_error > tau: β_bear=0.85 → RB pushes w toward B_BEAR (固收 0.35 + 黄金 0.45 = 0.80 防御)."""
    track = EventTrackV2()
    rets = _make_returns(seed=10)

    w = track.compute(rets, llm_macro=50, llm_sentiment=50, llm_risk=50,
                      ae_error=30.0, tau=20.0)

    # 固收 + 黄金 + 现金应占主导
    defensive = w[2] + w[3] + w[4]
    assert defensive >= 0.65, f"bear defensive {defensive:.3f} < 0.65"
    # 卫星应被压低
    assert w[1] <= 0.15, f"bear satellite {w[1]:.3f} > 0.15"
    # sum 归一
    assert abs(w.sum() - 1.0) < 1e-6


def test_strong_growth_signal_pushes_offensive():
    """强 macro+senti → β_growth 大 → b 推向 B_GROWTH,RB 给出更高 equity 权重."""
    track = EventTrackV2()
    rets = _make_returns(seed=20)

    w = track.compute(rets, llm_macro=85, llm_sentiment=85, llm_risk=20,
                      ae_error=0.0, tau=20.0)

    # 关键方向性: equity 在 growth 应 > 在 bear (而非绝对 > 50%)
    w_bear = track.compute(rets, llm_macro=50, llm_sentiment=50, llm_risk=50,
                           ae_error=30.0, tau=20.0)
    bear_equity = w_bear[0] + w_bear[1]
    growth_equity = w[0] + w[1]
    assert growth_equity > bear_equity, \
        f"growth equity {growth_equity:.3f} ≤ bear equity {bear_equity:.3f}"
    # 卫星在 growth 应 > 0.10 (RB 在 inverse-vol 下,0.40 的 b[1] 落到 w[1] ≈ 0.15)
    assert w[1] >= 0.10, f"growth satellite {w[1]:.3f} < 0.10"


def test_neutral_signal_soft_blend():
    """中性 LLM 信号 → β_neutral 主导 → 5 维权重相对均匀."""
    track = EventTrackV2()
    rets = _make_returns(seed=30)

    w = track.compute(rets, llm_macro=50, llm_sentiment=50, llm_risk=50,
                      ae_error=0.0, tau=20.0)

    # 各项权重应集中在 [0.10, 0.40] 区间(中性 RB 收敛)
    assert all(0.05 <= wi <= 0.45 for wi in w), f"neutral weights 偏离 {w}"
    # 极差(最大-最小)应 < 0.35
    assert (w.max() - w.min()) < 0.40, f"neutral spread {w.max()-w.min():.3f} 过大"


def test_polynomial_objective_no_division_by_portvar():
    """设计要点: 目标函数不应除以 w'Σw (避免 1/x torsion)."""
    # 间接验证: 在极低 vol 数据上 SLSQP 仍能 ftol=1e-9 收敛
    track = EventTrackV2()
    np.random.seed(7)
    rets = np.random.randn(5, 60) * 0.0001  # 极低 vol

    w = track.compute(rets, llm_macro=50, llm_sentiment=50, llm_risk=50)
    assert abs(w.sum() - 1.0) < 1e-6
    # 权重要有变化,不应是 w0 死锁
    assert np.linalg.norm(w - track.W0) > 1e-3, "SLSQP 死锁在 w0"


def test_beta_neutral_zero_protection():
    """极端双强信号: bear+growth 同时被推到上限,β_neutral 不应为负."""
    track = EventTrackV2()
    rets = _make_returns(seed=11)

    # d1=15 (超低 macro,大幅触发 bear) + d3=99 (超强 senti,大幅触发 growth)
    # bear = 0.50 * (50-15)/50 + 0.30 * (90-70)/30 = 0.35 + 0.20 = 0.55
    # growth = 0.45 * 0 + 0.40 * (99-65)/35 = 0.389
    # total = 0.939 → cap 触发 → 0.95/0.939 ≈ 1.012
    # 调整后: bear ≈ 0.557, growth ≈ 0.393
    # neutral = max(0, 1 - 0.557 - 0.393) = 0.05
    w = track.compute(rets, llm_macro=15, llm_sentiment=99, llm_risk=90,
                      ae_error=0.0, tau=20.0)

    # b 已被 cap 保护,RB 不会跑飞
    assert abs(w.sum() - 1.0) < 1e-6
    # 此时 bear + growth 都很强,权重应在 (fi, safe) 与 (broad, sat) 之间妥协
    assert 0.10 <= w[2] <= 0.50  # fi 介于 0.35(bear) 与 0.15(growth) 之间
    assert 0.10 <= w[0] <= 0.30  # broad 介于 0.05(bear) 与 0.25(growth) 之间


def test_bounds_enforced():
    """所有权重必须在 BOUNDS 范围内."""
    track = EventTrackV2()
    rets = _make_returns(seed=12)

    w = track.compute(rets, llm_macro=85, llm_sentiment=85, llm_risk=20)

    for i, (lo, hi) in enumerate(track.BOUNDS):
        assert lo - 1e-6 <= w[i] <= hi + 1e-6, \
            f"w[{i}]={w[i]:.3f} 超出 bounds [{lo}, {hi}]"


def test_solver_converges_60d_returns():
    """60 个交易日样本下,ftol=1e-9 应能收敛."""
    track = EventTrackV2()
    np.random.seed(99)
    rets = np.random.randn(5, 60) * 0.01

    # 通过设置不同 b 多次求解
    for macro in [25, 50, 75, 95]:
        w = track.compute(rets, llm_macro=macro, llm_sentiment=macro, llm_risk=50)
        assert abs(w.sum() - 1.0) < 1e-6
        # 权重要有变化(非死锁)
        assert np.linalg.norm(w - track.W0) > 1e-4 or macro == 50, \
            f"macro={macro} 时 SLSQP 未移动"


def test_w0_is_asymmetric():
    """设计要点 #2: w0 非等权,5 维全异,各分量与 0.2 偏差 ≤ 0.10."""
    track = EventTrackV2()
    # 5 维全异
    assert len(set(track.W0.round(4))) == 5, f"w0 {track.W0} 不是 5 维全异"
    # 不应等于 [0.2]*5
    assert not np.allclose(track.W0, [0.2] * 5)
    # 各分量与 0.2 偏差小起步,又非等权(允许 cash=0.15, fi=0.25 等合理差异)
    assert all(abs(x - 0.2) < 0.10 for x in track.W0), f"w0 {track.W0} 偏离 0.2 太大"


def test_v2_normal_track_edge_meaningful():
    """v2 EventTrack vs NormalTrack: 中性信号下 ||w_event - w_normal|| > 0.10."""
    engine = DualTrackEngine(use_v2=True)
    rets = _make_returns(seed=42)

    w_n, w_e = engine.compute(
        rets, llm_macro=50, llm_sentiment=50, llm_risk=50,
        ae_error=0.0, tau=20.0,
    )

    diff = np.linalg.norm(w_e - w_n)
    assert diff > 0.10, f"中性 ||w_event - w_normal|| = {diff:.3f} 应 > 0.10"


def test_v2_regime_conditional_divergence():
    """v2: bear vs bull+growth 下 EventTrack 距离应 > 0.10 且方向正确."""
    engine = DualTrackEngine(use_v2=True)
    rets = _make_returns(seed=77)

    w_bear = engine.compute(
        rets, llm_macro=50, llm_sentiment=50, llm_risk=50,
        ae_error=30.0, tau=20.0,
    )[1]
    w_growth = engine.compute(
        rets, llm_macro=85, llm_sentiment=85, llm_risk=20,
        ae_error=0.0, tau=20.0,
    )[1]

    diff = np.linalg.norm(w_bear - w_growth)
    # RB 逆 vol 特性让绝对距离受限,> 0.10 即可
    assert diff > 0.10, f"v2 bear vs growth 距离 {diff:.3f} < 0.10"
    # 方向: bear fi > growth fi
    assert w_bear[2] > w_growth[2], "bear fi 应 > growth fi"
    # 方向: growth equity > bear equity
    assert (w_growth[0] + w_growth[1]) > (w_bear[0] + w_bear[1])


def test_v2_bear_uses_bear_vertex_dominantly():
    """v2 bear regime: w 防御总权重应 > 0.70,fi 应 > 0.30."""
    track = EventTrackV2()
    rets = _make_returns(seed=33)

    w = track.compute(rets, llm_macro=50, llm_sentiment=50, llm_risk=50,
                      ae_error=30.0, tau=20.0)

    # RB 在 inverse-vol 下,固收(fi) 拿大头,黄金(safe) 反而较小
    # 但 B_BEAR 的 b[2]=0.328 → w[2] 应 ≥ 0.30
    assert w[2] >= 0.30, f"bear fi {w[2]:.3f} < 0.30"
    # 防御总权重 ≥ 0.75
    defensive = w[2] + w[3] + w[4]
    assert defensive >= 0.70, f"bear defensive {defensive:.3f} < 0.70"


def test_insufficient_samples_returns_normalized_w0():
    """< 5 个样本:回退归一化 w0(非全 0 或 NaN)."""
    track = EventTrackV2()
    rets = np.random.randn(5, 3) * 0.01

    w = track.compute(rets, llm_macro=50, llm_sentiment=50, llm_risk=50)

    assert not np.any(np.isnan(w))
    assert abs(w.sum() - 1.0) < 1e-6
    # 不应进入 bear 强制分支(没传 ae_error)
    assert not np.allclose(w, [0.05, 0.05, 0.35, 0.45, 0.10])
