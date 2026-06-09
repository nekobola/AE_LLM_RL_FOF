"""Stage 2: 验证两条 track 在不同 regime 下产生显著不同的权重(PPO fusion 价值)"""
import numpy as np

from src.compute.dual_track_engine import DualTrackEngine
from src.compute.normal_track import NormalTrack
from src.compute.event_track import EventTrack


def _make_returns(seed=42, broad_vol=0.012, sat_vol=0.015, fi_vol=0.003, safe_vol=0.010, cash_vol=0.001):
    """构造 5 资产各 60 日收益率矩阵"""
    np.random.seed(seed)
    rets = np.zeros((5, 60))
    rets[0] = np.random.randn(60) * broad_vol
    rets[1] = np.random.randn(60) * sat_vol
    rets[2] = np.random.randn(60) * fi_vol
    rets[3] = np.random.randn(60) * safe_vol
    rets[4] = np.random.randn(60) * cash_vol
    return rets


def test_bear_regime_forces_strong_divergence():
    """Bear regime: NormalTrack=BEAR_WEIGHTS, EventTrack=crisis 原型 → 两者接近 (均偏防御)
    关键是 W_Event 固收显著 > 0.35 (满足 PPO 切换价值)"""
    engine = DualTrackEngine()
    rets = _make_returns(seed=10)

    w_normal, w_event = engine.compute(
        rets,
        llm_macro=50.0,
        llm_sentiment=50.0,
        llm_risk=50.0,
        ae_error=30.0,
        tau=20.0,
    )

    # EventTrack 在 bear 必须出防御配置
    assert w_event[2] >= 0.35, f"bear w_event_fi {w_event[2]:.3f} < 0.35"
    assert w_event[3] >= 0.20, f"bear w_event_safe {w_event[3]:.3f} < 0.20"
    assert w_event[1] <= 0.10, f"bear w_event_satellite {w_event[1]:.3f} > 0.10"

    # NormalTrack 固收 = 0.50 (BEAR_WEIGHTS)
    assert w_normal[2] == 0.50, f"bear w_normal_fi {w_normal[2]:.3f} != 0.50"


def test_bull_strong_growth_forces_offensive_event():
    """Bull + 强增长信号: EventTrack 强制 growth 原型,卫星 ≥ 0.30"""
    engine = DualTrackEngine()
    rets = _make_returns(seed=20)

    w_normal, w_event = engine.compute(
        rets,
        llm_macro=85.0,
        llm_sentiment=85.0,
        llm_risk=20.0,
        ae_error=0.0,
        tau=20.0,
    )

    # EventTrack 应偏进攻
    assert w_event[1] >= 0.30, f"growth w_event_satellite {w_event[1]:.3f} < 0.30"
    assert w_event[0] >= 0.20, f"growth w_event_broad {w_event[0]:.3f} < 0.20"
    # EventTrack 与 NormalTrack 显著不同
    diff = np.linalg.norm(w_event - w_normal)
    assert diff > 0.20, f"event-normal 距离 {diff:.3f} 太小,无 fusion 价值"


def test_bull_weak_signal_uses_softmax_blend():
    """Bull + 中性信号: EventTrack 走 softmax,应该接近 base_neutral (温和)"""
    engine = DualTrackEngine()
    rets = _make_returns(seed=30)

    w_normal, w_event = engine.compute(
        rets,
        llm_macro=50.0,
        llm_sentiment=50.0,
        llm_risk=50.0,
        ae_error=0.0,
        tau=20.0,
    )

    base = np.array([0.20, 0.10, 0.30, 0.20, 0.20])
    diff_from_base = np.abs(w_event - base).max()
    # 在中性信号下偏离应 < 0.15
    assert diff_from_base < 0.15, f"中性 EventTrack 偏离 base {diff_from_base:.3f} 过大"


def test_fusion_alpha_gradient_exists():
    """Stage 2 核心目标: 不同 regime 下 (w_event - w_normal) 差异显著,
    使 PPO 在 α∈[0,1] 上有明确梯度可以爬"""
    engine = DualTrackEngine()
    rets = _make_returns(seed=42)

    # 1) bear 状态
    w_n_bear, w_e_bear = engine.compute(
        rets, llm_macro=50.0, llm_sentiment=50.0, llm_risk=50.0,
        ae_error=30.0, tau=20.0,
    )
    # 2) bull + 强增长
    w_n_bull, w_e_bull = engine.compute(
        rets, llm_macro=85.0, llm_sentiment=85.0, llm_risk=20.0,
        ae_error=0.0, tau=20.0,
    )

    # bear 周的 edge (event - normal) 与 bull 周的 edge 应该方向相反
    # bear: 两者都偏防御,edge 小但符号不一定
    # bull: 两者差异大,edge 大
    # 关键是: 同一周给定相同 returns,两条 track 给出的权重差必须 > 0.15
    bear_diff = np.linalg.norm(w_e_bear - w_n_bear)
    bull_diff = np.linalg.norm(w_e_bull - w_n_bull)

    assert bear_diff > 0.10, f"bear ||w_event - w_normal|| = {bear_diff:.3f} 应 > 0.10"
    assert bull_diff > 0.20, f"bull ||w_event - w_normal|| = {bull_diff:.3f} 应 > 0.20"


def test_event_track_bear_vs_bull_strongly_different():
    """EventTrack 在 bear vs bull 下的输出应显著不同 (regime 敏感度)"""
    engine = DualTrackEngine()
    rets = _make_returns(seed=99)

    w_bear = engine.compute(
        rets, llm_macro=50.0, llm_sentiment=50.0, llm_risk=50.0,
        ae_error=30.0, tau=20.0,
    )[1]

    w_bull_growth = engine.compute(
        rets, llm_macro=85.0, llm_sentiment=85.0, llm_risk=20.0,
        ae_error=0.0, tau=20.0,
    )[1]

    diff = np.linalg.norm(w_bear - w_bull_growth)
    assert diff > 0.50, f"EventTrack bear vs growth 距离 {diff:.3f} 太小"
    # bear 时卫星很低, growth 时卫星很高
    assert w_bear[1] < 0.10, f"bear 卫星 {w_bear[1]:.3f} 应 < 0.10"
    assert w_bull_growth[1] > 0.30, f"growth 卫星 {w_bull_growth[1]:.3f} 应 > 0.30"
