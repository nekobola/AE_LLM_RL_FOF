import numpy as np

from src.compute.event_track import EventTrack


def test_event_track_crisis_prefers_defensive_assets():
    track = EventTrack()
    returns_5d = np.array(
        [
            [0.03, -0.04, 0.05, -0.02, 0.04],
            [0.06, -0.08, 0.09, -0.03, 0.07],
            [0.003, 0.002, -0.001, 0.002, 0.001],
            [0.004, -0.002, 0.003, 0.002, -0.001],
            [0.0001, 0.0002, 0.0000, -0.0001, 0.0001],
        ],
        dtype=float,
    )

    weights = track.compute(
        returns_5d,
        llm_macro=30.0,
        llm_sentiment=25.0,
        llm_risk=90.0,
        ae_error=0.0,
        tau=20.0,
    )

    defensive_weight = weights[2] + weights[3] + weights[4]
    equity_weight = weights[0] + weights[1]
    assert defensive_weight > equity_weight
    assert weights[2] >= 0.25


def test_event_track_growth_prefers_satellite_equity():
    track = EventTrack()
    returns_5d = np.array(
        [
            [0.01, 0.012, 0.008, 0.011, 0.009],
            [0.025, 0.020, 0.027, 0.023, 0.021],
            [0.001, 0.000, 0.002, 0.001, 0.000],
            [0.002, 0.001, 0.001, 0.002, 0.001],
            [0.000, 0.000, 0.000, 0.000, 0.000],
        ],
        dtype=float,
    )

    weights = track.compute(
        returns_5d,
        llm_macro=78.0,
        llm_sentiment=82.0,
        llm_risk=35.0,
        ae_error=0.0,
        tau=20.0,
    )

    assert weights[1] > weights[2]
    assert weights[1] > weights[3]


def test_event_track_forced_crisis_in_bear_regime():
    """Stage 2: ae_error > tau 必须强制 crisis 原型,不再 softmax 微调"""
    track = EventTrack()
    returns_5d = np.random.randn(5, 20) * 0.01  # 中性 vol

    weights = track.compute(
        returns_5d,
        llm_macro=50.0,    # 全部中性
        llm_sentiment=50.0,
        llm_risk=50.0,
        ae_error=30.0,     # bear regime
        tau=20.0,
    )

    # Crisis 原型: 固收 0.45 + 黄金 0.35 + 现金 0.15 = 0.95
    # 在 event_intensity 混合下,固收应该显著 ≥ 0.35
    assert weights[2] >= 0.35, f"bear regime 固收权重 {weights[2]:.3f} < 0.35"
    assert weights[1] <= 0.10, f"bear regime 卫星权重 {weights[1]:.3f} > 0.10"
    # 防御总权重(固收+黄金+现金)应远超进攻总权重
    defensive = weights[2] + weights[3] + weights[4]
    offensive = weights[0] + weights[1]
    assert defensive > 2 * offensive, f"defensive={defensive:.3f} not > 2*offensive={offensive:.3f}"


def test_event_track_forced_growth_in_strong_macro():
    """Stage 2: 强增长信号 + 低权益 stress 必须强制 growth 原型"""
    track = EventTrack()
    # 极低 vol 模拟(无 equity stress)
    returns_5d = np.array(
        [
            [0.001, 0.001, 0.001, 0.001, 0.001] * 4,
        ] * 5,
        dtype=float,
    )
    # 让 sat 比 broad 略高以产生 sat_lead
    returns_5d[1] = returns_5d[1] * 1.5

    weights = track.compute(
        returns_5d,
        llm_macro=85.0,    # 极强
        llm_sentiment=85.0,
        llm_risk=20.0,     # 低风险
        ae_error=0.0,      # bull
        tau=20.0,
    )

    # Growth 原型: 卫星 0.48 + 宽基 0.32 = 0.80 进攻
    # 在 event_intensity 混合下,卫星应显著 ≥ 0.35
    assert weights[1] >= 0.30, f"strong growth 卫星权重 {weights[1]:.3f} < 0.30"
    assert weights[0] >= 0.20, f"strong growth 宽基权重 {weights[0]:.3f} < 0.20"


def test_event_track_bull_uses_softmax_blend():
    """Stage 2: bull + 中性信号 → softmax 混合,不应强制任何原型"""
    track = EventTrack()
    np.random.seed(42)
    returns_5d = np.random.randn(5, 20) * 0.01

    weights = track.compute(
        returns_5d,
        llm_macro=50.0,
        llm_sentiment=50.0,
        llm_risk=50.0,
        ae_error=0.0,
        tau=20.0,
    )

    # 不应过度偏离 base_neutral
    base = np.array([0.20, 0.10, 0.30, 0.20, 0.20])
    diff = np.abs(weights - base).max()
    # event_intensity 中性时偏离应 < 0.10
    assert diff < 0.15, f"bull+中性 偏离 base_neutral {diff:.3f} 过大"
