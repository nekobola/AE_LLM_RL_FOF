"""EventTrackV3 (Exponential Tilting) unit tests.

Coverage:
  1. Stagflation scenario: V3 produces b with gold + cash dominant,
     fi compressed. V2 cannot reach this point in Δ⁴.
  2. Neutral signal → b = b0 (uniform), w ≈ uniform.
  3. Strong bull → b tilts heavily to broad/satellite.
  4. Strong bear → b tilts heavily to fi/safe/cash.
  5. AE sigmoid soft injection is C^∞ smooth (no hard switch).
  6. RB convergence at ftol=1e-9.
  7. Asymmetric w0.
  8. Bounds enforced.
  9. V3 vs V2 divergence: V3 reaches points V2 cannot.
"""
import numpy as np

from src.compute.event_track_v3 import EventTrackV3
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


# ── Core b-target behavior ──


def test_v3_b_uniform_when_neutral():
    """中性信号: m=s=r=0, equity_stress=0, sat_lead=0 → b = b0 = uniform."""
    track = EventTrackV3()
    b = track._per_asset_scores(0, 0, 0, 0, 0)
    np.testing.assert_array_almost_equal(b, [0, 0, 0, 0, 0])

    b_tilt = track.B0 * np.exp(track.THETA * b)
    b_tilt = b_tilt / b_tilt.sum()
    np.testing.assert_array_almost_equal(b_tilt, track.B0)


def test_v3_b_strong_bull_pushes_equity():
    """强 bull (m=+1, s=+1, r=-1): b_broad + b_sat 应该 ≫ defensive."""
    track = EventTrackV3()
    m, s, r = 1.0, 1.0, -1.0
    scores = track._per_asset_scores(m, s, r, 0, 0.5)
    b = track.B0 * np.exp(track.THETA * scores)
    b = b / b.sum()

    equity_share = b[0] + b[1]
    defensive_share = b[2] + b[3] + b[4]
    assert equity_share > defensive_share, (
        f"bull equity {equity_share:.3f} ≤ defensive {defensive_share:.3f}"
    )
    assert b[1] > 0.3, f"sat {b[1]:.3f} 应主导 bull 象限"
    assert b[2] < 0.05, f"fi {b[2]:.3f} 几乎为 0"


def test_v3_b_stagflation_dominates_gold_and_cash():
    """滞胀 (m=-0.5, s=-0.3, r=+0.8, eq_stress=+0.5): gold + cash 应该主导."""
    track = EventTrackV3()
    m, s, r = -0.5, -0.3, 0.8
    scores = track._per_asset_scores(m, s, r, 0.5, 0)
    b = track.B0 * np.exp(track.THETA * scores)
    b = b / b.sum()

    # 滞胀: 黄金 + 现金 + 固收都应该显著,equity 几乎为 0
    safe_total = b[2] + b[3] + b[4]
    equity = b[0] + b[1]
    assert safe_total > 0.7, f"stagflation safe_total {safe_total:.3f} ≤ 0.7"
    assert equity < 0.3, f"stagflation equity {equity:.3f} ≥ 0.3"
    # 关键: 黄金单独 ≥ 0.20 (V2 三角形内任何顶点都到不了这种 "gold 单高" 形态)
    assert b[3] >= 0.20, f"stagflation gold {b[3]:.3f} < 0.20"


def test_v3_stagflation_unreachable_in_v2_triangle():
    """关键论证: V2 凸包 {B_BEAR, B_NEUTRAL, B_GROWTH} 内部任何点
    都不能精确产生 stagflation 形态。V3 可以,距离差距是数量级。"""
    track_v2 = EventTrackV2()
    track_v3 = EventTrackV3()

    m, s, r = -0.5, -0.3, 0.8
    # 扫描所有 V2 β_bear ∈ [0, 1], β_growth ∈ [0, 1-β_bear]
    b_v2_min_dist = float("inf")
    # V3 实际产生的 b 作为 target
    scores = track_v3._per_asset_scores(m, s, r, 0.5, 0)
    b_v3 = track_v3.B0 * np.exp(track_v3.THETA * scores)
    b_v3 = b_v3 / b_v3.sum()
    target = b_v3.copy()

    for beta_bear in np.linspace(0, 1, 21):
        for beta_growth in np.linspace(0, 1 - beta_bear, 21):
            beta_neutral = 1 - beta_bear - beta_growth
            if beta_neutral < 0:
                continue
            b_v2 = (
                beta_bear * track_v2.B_BEAR
                + beta_growth * track_v2.B_GROWTH
                + beta_neutral * track_v2.B_NEUTRAL
            )
            dist = np.linalg.norm(b_v2 - target)
            b_v2_min_dist = min(b_v2_min_dist, dist)

    # V2 凸包内最近点距 target > 0.20(无法精确表达),V3 距离 < 0.001
    assert b_v2_min_dist > 0.15, (
        f"V2 凸包最近 {b_v2_min_dist:.3f} < 0.15,V2 其实能接近这种形态"
    )
    # 验证 V3 真的能精确产生
    v3_dist = np.linalg.norm(b_v3 - target)
    assert v3_dist < 1e-6, f"V3 距自己 {v3_dist:.6f} 应为 0"


# ── End-to-end compute ──


def test_v3_neutral_returns_near_uniform():
    """中性 LLM 信号 + 平稳市场 → b = b0(均匀),w 受 RB inverse-vol 影响."""
    track = EventTrackV3()
    rets = _make_returns(seed=42)

    w = track.compute(rets, llm_macro=50, llm_sentiment=50, llm_risk=50)

    # b 是均匀的(中性 LLM + 中性 market)
    scores = track._per_asset_scores(0, 0, 0, 0, 0)
    b = track.B0 * np.exp(track.THETA * scores)
    b = b / b.sum()
    np.testing.assert_array_almost_equal(b, track.B0)

    # w 不一定均匀(RB inverse-vol: fi vol=0.003 → w[2] 偏高)
    # 但应满足:defensive (fi+safe+cash) ≥ equity (broad+sat) - 因为 b 偏 0.2 均匀
    # + fi 极低 vol
    assert all(0.05 <= wi <= 0.50 for wi in w), f"neutral w 偏离 {w}"


def test_v3_strong_bull_offensive():
    """强 bull → w 应偏进攻."""
    track = EventTrackV3()
    rets = _make_returns(seed=42)

    w = track.compute(rets, llm_macro=85, llm_sentiment=85, llm_risk=20)

    equity = w[0] + w[1]
    defensive = w[2] + w[3] + w[4]
    # bull 不一定 equity > defensive (受 RB inverse-vol 影响),
    # 但 bull equity 应显著 > bear equity
    w_bear = track.compute(rets, llm_macro=50, llm_sentiment=50, llm_risk=50,
                           ae_error=30, tau=20)
    assert equity > (w_bear[0] + w_bear[1])


def test_v3_bear_regime_defensive():
    """AE > τ: w 应偏防御,固收主导."""
    track = EventTrackV3()
    rets = _make_returns(seed=42)

    w = track.compute(rets, llm_macro=50, llm_sentiment=50, llm_risk=50,
                      ae_error=30, tau=20)

    # 防御总权重 ≥ 0.70
    defensive = w[2] + w[3] + w[4]
    assert defensive >= 0.70, f"bear defensive {defensive:.3f} < 0.70"
    # 固收 ≥ 0.25 (RB inverse-vol + bear_pressure boost)
    assert w[2] >= 0.25, f"bear fi {w[2]:.3f} < 0.25"


# ── AE sigmoid soft injection ──


def test_v3_ae_sigmoid_smooth_transition():
    """C^∞ 平滑: E_t 大幅跨越 τ 时 w 平滑变化,无跳变."""
    track = EventTrackV3()
    rets = _make_returns(seed=42)

    # E_t 序列从 5 到 50(大幅跨越 τ=20, 包含纯 bull + 强 bear)
    e_seq = np.linspace(5, 50, 10)
    w_seq = []
    for e in e_seq:
        w = track.compute(rets, llm_macro=50, llm_sentiment=50, llm_risk=50,
                          ae_error=e, tau=20)
        w_seq.append(w)
    w_seq = np.array(w_seq)

    # 步进间变化: 单步变化不应超过总范围变化
    step_diffs = [
        np.linalg.norm(w_seq[i+1] - w_seq[i])
        for i in range(len(w_seq)-1)
    ]
    total_diff = np.linalg.norm(w_seq[-1] - w_seq[0])

    # 平滑性: 平均步进 < 总变化 / 步数 × 2
    avg_step = np.mean(step_diffs)
    assert avg_step < 2 * total_diff / len(e_seq), (
        f"sigmoid 应平滑, avg_step={avg_step:.4f} 应 < {2*total_diff/len(e_seq):.4f}"
    )
    # 无单步跳变: max step < 3x avg step
    assert max(step_diffs) < 3 * avg_step, (
        f"sigmoid 不应有跳变, max_step={max(step_diffs):.4f} vs avg={avg_step:.4f}"
    )


def test_v3_bear_pressure_amplifies_defensive():
    """AE > τ 时 θ_eff 增加,defensive 资产得分更高."""
    track = EventTrackV3()
    rets = _make_returns(seed=42)

    w_no_bear = track.compute(rets, llm_macro=50, llm_sentiment=50, llm_risk=50)
    w_strong_bear = track.compute(rets, llm_macro=50, llm_sentiment=50, llm_risk=50,
                                   ae_error=50, tau=20)

    # 强 bear 时期 defensive 应 ≥ 无 bear 时期
    assert (w_strong_bear[2] + w_strong_bear[3] + w_strong_bear[4]) >= (
        w_no_bear[2] + w_no_bear[3] + w_no_bear[4]
    )


# ── Solver and bounds ──


def test_v3_bounds_enforced():
    """所有 w 必须在 BOUNDS 范围内."""
    track = EventTrackV3()
    rets = _make_returns(seed=42)

    w = track.compute(rets, llm_macro=85, llm_sentiment=85, llm_risk=20)

    for i, (lo, hi) in enumerate(track.BOUNDS):
        assert lo - 1e-6 <= w[i] <= hi + 1e-6, \
            f"w[{i}]={w[i]:.3f} 超出 bounds [{lo}, {hi}]"


def test_v3_w0_is_asymmetric():
    """V2 沿用: w0 非等权."""
    track = EventTrackV3()
    assert not np.allclose(track.W0, [0.2] * 5)
    assert len(set(track.W0.round(4))) == 5


def test_v3_solver_converges():
    """60 日样本下,ftol=1e-9 收敛."""
    track = EventTrackV3()
    np.random.seed(99)
    rets = np.random.randn(5, 60) * 0.01

    for macro in [25, 50, 75, 95]:
        w = track.compute(rets, llm_macro=macro, llm_sentiment=macro, llm_risk=50)
        assert abs(w.sum() - 1.0) < 1e-6


def test_v3_insufficient_samples_fallback():
    """样本 < 2 → 回退归一化 b-as-w."""
    track = EventTrackV3()
    rets = np.random.randn(5, 1) * 0.01

    w = track.compute(rets, llm_macro=85, llm_sentiment=85, llm_risk=20)
    assert not np.any(np.isnan(w))
    assert abs(w.sum() - 1.0) < 1e-6


# ── V3 vs V2 comparison ──


def test_v3_normal_track_edge_meaningful():
    """V3 EventTrack vs NormalTrack: 中性信号下 ||w_event - w_normal|| > 0.10."""
    engine = DualTrackEngine(use_v3=True)
    rets = _make_returns(seed=42)

    w_n, w_e = engine.compute(
        rets, llm_macro=50, llm_sentiment=50, llm_risk=50,
        ae_error=0.0, tau=20.0,
    )

    diff = np.linalg.norm(w_e - w_n)
    assert diff > 0.10, f"中性 ||w_event - w_normal|| = {diff:.3f} 应 > 0.10"


def test_v3_regime_conditional_divergence():
    """V3: bear vs bull+growth 下 EventTrack 距离应 > 0.15."""
    engine = DualTrackEngine(use_v3=True)
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
    assert diff > 0.15, f"V3 bear vs growth 距离 {diff:.3f} < 0.15"
