"""EventTrackV3.1 (audit fixes) unit tests.

V3.1 fixes three structural flaws in V3:
  1. Scale mismatch: matrix-normalized scores (W·f instead of hardcoded additive)
  2. Gold tragedy: scores bounded + AE shifter enforces V_DEFENSE in bear
  3. AE gain paradox: AE is now a SHIFTER (convex combo), not a GAIN multiplier

Tests:
  A. Score range bound: s_i in [-1, 1] for all f in [-1, 1]^5
  B. Gold tragedy regression: crisis → b_gold ≥ b_fi (was 0.5x in V3)
  C. AE shifter dominance: bear_pressure=1 forces s toward V_DEFENSE
  D. Bull-end + AE-bear: defense forced (no bull amplification)
  E. Crisis LLM signals + no AE: b_gold ≈ b_fi (tied), neither dominates
  F. Stagflation expressivity (V3.1 covers what V2 cannot)
  G. End-to-end pipeline: V3.1 in bear matches V2's b_bear shape
  H. RB convergence + bounds + asymmetric w0
"""
import numpy as np

from src.compute.event_track_v3_1 import EventTrackV31
from src.compute.event_track_v2 import EventTrackV2
from src.compute.v31_engine import V31Engine
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


# ── A. Score range bound ──


def test_v31_all_scores_bounded_in_unit_range():
    """Fix 1 verification: W·f for any f in [-1,1]^5 → s in [-1,1]."""
    track = EventTrackV31()
    # Sample 1000 random f vectors
    np.random.seed(0)
    fs = np.random.uniform(-1, 1, size=(1000, 5))
    for f in fs:
        s = track.W @ f
        assert all(-1.01 <= si <= 1.01 for si in s), f"s out of [-1, 1]: {s} for f={f}"


def test_v31_gold_crisis_reaches_max():
    """V3.1 matrix: pure crisis (r=1, m=-1) → s_gold = 1."""
    track = EventTrackV31()
    f = np.array([-1.0, 0.0, 1.0, 0.0, 0.0])  # m=-1, r=1
    s = track.W @ f
    # Gold row: -1/3*m + 2/3*r = 1/3 + 2/3 = 1
    assert abs(s[3] - 1.0) < 1e-6, f"s_gold {s[3]} != 1.0 in crisis"


def test_v31_fi_crisis_reaches_max():
    """V3.1 matrix: pure crisis (m=-1, s=-1, eq=1) → s_fi = 1."""
    track = EventTrackV31()
    f = np.array([-1.0, -1.0, 0.0, 1.0, 0.0])  # m=-1, s=-1, eq=1
    s = track.W @ f
    # Fi row: -1/3*m + -1/3*s + 1/3*eq = 1/3 + 1/3 + 1/3 = 1
    assert abs(s[2] - 1.0) < 1e-6, f"s_fi {s[2]} != 1.0 in crisis"


# ── B. Gold tragedy regression ──


def test_v31_gold_dominates_in_bear_regime():
    """V3.1 fixes gold tragedy: 在 AE 强 bear 时, gold 应该 ≥ fi."""
    track = EventTrackV31()
    rets = _make_returns(seed=42)

    # 强 bear (AE 远高于 τ)
    w = track.compute(rets, llm_macro=50, llm_sentiment=50, llm_risk=50,
                      ae_error=80, tau=20)  # bear_pressure ≈ 1.0

    # b: gold 至少应与 fi 相当
    # 注: RB inverse-vol 可能会调整,但 gold 应保持高
    assert w[3] >= w[2] - 0.05, f"bear w_gold {w[3]:.3f} 远小于 w_fi {w[2]:.3f}"


def test_v31_crisis_signals_gold_geq_fi():
    """V3 关键回归: 纯 crisis LLM 信号(无 AE 强注入),s_fi = s_gold = 1 (并列)."""
    track = EventTrackV31()
    rets = _make_returns(seed=42)

    # m=-1, s=-1, r=+1, equity_stress≈0.5(由 rets 算出)
    w = track.compute(rets, llm_macro=0, llm_sentiment=0, llm_risk=100,
                      ae_error=21, tau=20)  # 弱 bear (just above tau)

    # 弱 bear 时, s_final ≈ s(LLM),其中 s_fi 和 s_gold 接近 1
    # b: gold 至少 ≥ fi - 0.05(RB inverse-vol 微差)
    assert w[3] >= w[2] - 0.05, f"crisis w_gold {w[3]:.3f} < w_fi {w[2]:.3f}"


def test_v31_no_v3_gold_tragedy():
    """直接对比 V3 vs V3.1: 在同一危机信号下, V3 gold < fi, V3.1 gold ≥ fi."""
    track_v3 = EventTrackV3 = None
    from src.compute.event_track_v3 import EventTrackV3
    track_v3 = EventTrackV3()
    track_v31 = EventTrackV31()
    rets = _make_returns(seed=42)

    # Crisis LLM: m=0(50), s=0(50), r=100 — 应该触发 bear
    w_v3 = track_v3.compute(rets, llm_macro=50, llm_sentiment=50, llm_risk=100,
                            ae_error=21, tau=20)
    w_v31 = track_v31.compute(rets, llm_macro=50, llm_sentiment=50, llm_risk=100,
                              ae_error=21, tau=20)

    # V3 在此情景下应该是 gold < fi (V3 黄金悲剧)
    # V3.1 应该 gold >= fi
    print(f"V3: gold={w_v3[3]:.3f}, fi={w_v3[2]:.3f}, gold/fi = {w_v3[3]/w_v3[2]:.2f}")
    print(f"V3.1: gold={w_v31[3]:.3f}, fi={w_v31[2]:.3f}, gold/fi = {w_v31[3]/w_v31[2]:.2f}")

    # V3.1 gold 至少应该 ≥ V3 gold(修复不应让 gold 减少)
    # 更严格: V3.1 gold/fi > V3 gold/fi
    ratio_v3 = w_v3[3] / max(w_v3[2], 1e-6)
    ratio_v31 = w_v31[3] / max(w_v31[2], 1e-6)
    assert ratio_v31 > ratio_v3, f"V3.1 gold/fi {ratio_v31:.2f} ≤ V3 {ratio_v3:.2f}"


# ── C. AE shifter dominance ──


def test_v31_bear_pressure_forces_defense():
    """Fix 2: bear_pressure=1 强制 s → V_DEFENSE."""
    track = EventTrackV31()
    rets = _make_returns(seed=42)

    # 强 bear (AE >> τ) → bear_pressure → 1
    w = track.compute(rets, llm_macro=50, llm_sentiment=50, llm_risk=50,
                      ae_error=80, tau=20)

    # 此时 s ≈ V_DEFENSE = [-1, -1, 0.5, 1, 0.5]
    # b: gold 主导(0.40+), fi/cash 中等(0.24), broad/sat 极低(0.05)
    # 验证 weight 模式
    assert w[3] > 0.30, f"AE 强 bear gold {w[3]:.3f} 应 ≥ 0.30"
    assert w[0] < 0.10, f"AE 强 bear broad {w[0]:.3f} 应 < 0.10"
    assert w[1] < 0.10, f"AE 强 bear sat {w[1]:.3f} 应 < 0.10"


# ── D. Bull-end + AE-bear divergence ──


def test_v31_bull_end_ae_bear_forces_defense():
    """V3 关键 bug 修复: 牛市末端 + AE 警报 → 强制防御, 不再强化 bull."""
    track = EventTrackV31()
    rets = _make_returns(seed=42)

    # LLM 信号强烈看多(2015 中, 2007 中)
    # m=85, s=85, r=15 → 强烈 risk-on
    # 但 AE 触发 bear
    w = track.compute(rets, llm_macro=85, llm_sentiment=85, llm_risk=15,
                      ae_error=80, tau=20)

    # AE 强 bear (bear_pressure → 1) 应压过 LLM 看多
    # 期望 w 接近 V_DEFENSE 行为
    assert w[3] > 0.20, f"AE bear should override LLM bull, gold {w[3]:.3f} 太小"
    assert w[0] < 0.20, f"AE bear should suppress equity, broad {w[0]:.3f} 太大"


def test_v31_no_ae_no_shifter():
    """无 AE 信号时, V3.1 退化为纯 LLM 驱动, 不应有 V_DEFENSE 强制."""
    track = EventTrackV31()
    rets = _make_returns(seed=42)

    w_no_ae = track.compute(rets, llm_macro=85, llm_sentiment=85, llm_risk=15)
    w_with_ae = track.compute(rets, llm_macro=85, llm_sentiment=85, llm_risk=15,
                              ae_error=80, tau=20)

    # 无 AE: equity 主导; 有 AE: 防御
    eq_no_ae = w_no_ae[0] + w_no_ae[1]
    eq_with_ae = w_with_ae[0] + w_with_ae[1]
    def_with_ae = w_with_ae[2] + w_with_ae[3] + w_with_ae[4]

    assert eq_no_ae > 0.30, f"无 AE bull equity {eq_no_ae:.3f} 太小"
    assert def_with_ae > eq_with_ae, f"有 AE 应防御主导, 但 eq={eq_with_ae:.3f}, def={def_with_ae:.3f}"


# ── E. Crisis LLM signals alone ──


def test_v31_crisis_llm_ties_gold_and_fi():
    """无 AE 时, 纯 crisis LLM 信号: gold 和 fi 并列 (s 都 = 1), 后续由 RB 微调."""
    track = EventTrackV31()
    f = np.array([-1.0, -1.0, 1.0, 1.0, 0.0])  # 极端 crisis
    s = track.W @ f
    # 期望: s_fi = 1, s_gold = 1, s_cash = 1 (三者并列)
    assert abs(s[2] - 1.0) < 1e-6
    assert abs(s[3] - 1.0) < 1e-6
    assert abs(s[4] - 1.0) < 1e-6


# ── F. Stagflation expressivity ──


def test_v31_stagflation_against_v2():
    """V3.1 仍能表达 V2 凸包外的 stagflation (黄金+现金双高)."""
    track_v2 = EventTrackV2()
    track_v31 = EventTrackV31()

    # 滞胀 LLM: m=25, s=35, r=90 (低 macro, 中 sentiment, 高 risk)
    m, s, r = -0.5, -0.3, 0.8
    f = np.array([m, s, r, 0.5, 0.0])
    scores = track_v31.W @ f
    b_v31 = track_v31.B0 * np.exp(track_v31.THETA * scores)
    b_v31 = b_v31 / b_v31.sum()

    # V3.1 在滞胀下: gold 应该显著(>0.20), cash 应该中等(>0.15)
    assert b_v31[3] >= 0.20, f"stagflation gold {b_v31[3]:.3f} < 0.20"
    assert b_v31[4] >= 0.10, f"stagflation cash {b_v31[4]:.3f} < 0.10"

    # V2 凸包内最接近点距离 > V3.1 距离
    target = b_v31.copy()
    b_v2_min = float("inf")
    for beta_bear in np.linspace(0, 1, 11):
        for beta_growth in np.linspace(0, 1 - beta_bear, 11):
            beta_neutral = 1 - beta_bear - beta_growth
            if beta_neutral < 0:
                continue
            b_v2 = (
                beta_bear * track_v2.B_BEAR
                + beta_growth * track_v2.B_GROWTH
                + beta_neutral * track_v2.B_NEUTRAL
            )
            b_v2_min = min(b_v2_min, np.linalg.norm(b_v2 - target))

    assert b_v2_min > 0.10, f"V2 凸包最近 {b_v2_min:.3f} < 0.10, 实际能表达"


# ── G. End-to-end regime match V2 ──


def test_v31_bear_matches_v2_bear_shape():
    """V3.1 强 bear 时, 权重模式应接近 V2 bear 顶点(gi, gold, cash 主导)."""
    track_v2 = EventTrackV2()
    track_v31 = EventTrackV31()
    rets = _make_returns(seed=42)

    # V3.1 强 bear (AE → 1)
    w_v31 = track_v31.compute(rets, llm_macro=50, llm_sentiment=50, llm_risk=50,
                              ae_error=80, tau=20)

    # V2 bear 顶点
    b_v2 = track_v2.B_BEAR  # (0.05, 0.05, 0.45, 0.35, 0.10)

    # 模式相似: defensive 占主导, equity 极低
    assert (w_v31[2] + w_v31[3] + w_v31[4]) > 0.70
    assert (w_v31[0] + w_v31[1]) < 0.30


# ── H. Solver and bounds ──


def test_v31_bounds_enforced():
    track = EventTrackV31()
    rets = _make_returns(seed=42)

    w = track.compute(rets, llm_macro=85, llm_sentiment=85, llm_risk=15)
    for i, (lo, hi) in enumerate(track.BOUNDS):
        assert lo - 1e-6 <= w[i] <= hi + 1e-6


def test_v31_bounds_asymmetric():
    """V3.1 bounds are intentionally asymmetric (gold has room to dominate in bear)."""
    track = EventTrackV31()
    # Bounds should not all be uniform — each asset has a distinct role
    assert not all(b == track.BOUNDS[0] for b in track.BOUNDS)
    # Gold and fi upper bounds should be set to allow bear-defensive pattern
    # (gold needs room above 0.30 to break the V3 tragedy)
    assert track.BOUNDS[3][1] >= 0.35, "gold upper bound should be ≥ 0.35"
    # Cash should be tight (cap ≤ 0.25) to prevent over-cash stalling
    assert track.BOUNDS[4][1] <= 0.25, "cash upper bound should be ≤ 0.25"


def test_v31_theta_stage6():
    """Stage 6: THETA reduced 1.0 -> 0.7 to lift weekly std and edge std."""
    track = EventTrackV31()
    assert track.THETA == 0.7, f"Stage 6 THETA should be 0.7, got {track.THETA}"
    # Compare V3.1 (0.7) vs hypothetical V3.1 (1.0) on the same bear signal
    rets = np.random.RandomState(0).randn(5, 60) * np.array([0.012, 0.015, 0.003, 0.010, 0.001])[:, None]
    w_th07 = track.compute(rets, llm_macro=15, llm_sentiment=15, llm_risk=85,
                           ae_error=80, tau=20)
    track.THETA = 1.0
    w_th10 = track.compute(rets, llm_macro=15, llm_sentiment=15, llm_risk=85,
                           ae_error=80, tau=20)
    track.THETA = 0.7  # restore
    # Lower THETA -> less extreme b distribution -> gold weight should be lower
    # (because exp(s*0.7) is more uniform than exp(s*1.0))
    # This is the design: lift weekly std by reducing the extremes
    print(f"  THETA=0.7: gold={w_th07[3]:.3f}, fi={w_th07[2]:.3f}")
    print(f"  THETA=1.0: gold={w_th10[3]:.3f}, fi={w_th10[2]:.3f}")
    # THETA=0.7 should have more balanced (less extreme) weights
    diff_07 = abs(w_th07[3] - w_th07[2])
    diff_10 = abs(w_th10[3] - w_th10[2])
    assert diff_07 < diff_10, f"THETA=0.7 should be more balanced: {diff_07} < {diff_10}"


def test_v31_solver_converges():
    track = EventTrackV31()
    np.random.seed(99)
    rets = np.random.randn(5, 60) * 0.01

    for macro in [25, 50, 75, 95]:
        w = track.compute(rets, llm_macro=macro, llm_sentiment=macro, llm_risk=50)
        assert abs(w.sum() - 1.0) < 1e-6


def test_v31_insufficient_samples_fallback():
    track = EventTrackV31()
    rets = np.random.randn(5, 1) * 0.01

    w = track.compute(rets, llm_macro=85, llm_sentiment=85, llm_risk=20)
    assert not np.any(np.isnan(w))
    assert abs(w.sum() - 1.0) < 1e-6


def test_v31_v_defense_norm_unit():
    """V_DEFENSE 各项应在 [-1, 1]."""
    track = EventTrackV31()
    for v in track.V_DEFENSE:
        assert -1.0 <= v <= 1.0, f"V_DEFENSE entry {v} 超出 [-1, 1]"


def test_v31_three_regimes_smoothly_separated():
    """三个 regime (bull, neutral, bear) 应产生显著不同的权重."""
    engine = V31Engine()
    rets = _make_returns(seed=77)

    w_bull = engine.compute(
        rets, llm_macro=85, llm_sentiment=85, llm_risk=15,
        ae_error=5, tau=20,
    )
    w_neutral = engine.compute(
        rets, llm_macro=50, llm_sentiment=50, llm_risk=50,
        ae_error=10, tau=20,
    )
    w_bear = engine.compute(
        rets, llm_macro=15, llm_sentiment=15, llm_risk=85,
        ae_error=60, tau=20,
    )

    diff_bull_bear = np.linalg.norm(w_bull - w_bear)
    diff_neutral_bull = np.linalg.norm(w_neutral - w_bull)
    diff_neutral_bear = np.linalg.norm(w_neutral - w_bear)

    assert diff_bull_bear > 0.20, f"bull vs bear 距离 {diff_bull_bear:.3f} 太小"
    assert diff_neutral_bull > 0.10
    assert diff_neutral_bear > 0.10
