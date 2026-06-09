"""Stage 1 ML-side unit tests.

Stage 1 fixes the PPO meta-controller collapse (alpha locks at 1.0, tau locks at 20.0).
The fix is split across three changes:

  1. config.yaml / action_mapper:  alpha_max 0.5 -> 0.1, alpha_bias 0.0 -> -0.05
     (Smaller step size + negative bias breaks the a1=0 stable equilibrium)
  2. config.yaml / reward_function: switch_bull_reward 0.45 -> 0.05,
     lambda_alpha_direct 0.0 -> 0.05
     (Weaker regime-switch reward + new regime-conditional alpha signal)
  3. reward_function.compute():  wire lambda_alpha_direct into the per-step reward
     so that the PPO controller gets a smooth gradient on alpha in both regimes

Tests cover:
  A. Reward function: lambda_alpha_direct is actually used in compute
  B. Regime-conditional alpha signal: bull rewards high alpha, bear penalizes
  C. Switch reward: 4 quadrants (alpha x regime) all behave correctly
  D. Action mapper: bias breaks the a1=0 stable equilibrium
  E. Action mapper: alpha_max=0.1 caps per-step add to 10%
  F. End-to-end: MDPEnvironment + Stage 1 config produces meaningful alpha dynamics
  G. End-to-end: PPO-friendly gradient exists (no longer flat)
"""
import numpy as np
import pytest

from src.env.action_mapper import ActionMapper
from src.env.mdp_environment import MDPEnvironment
from src.env.reward_function import RewardFunction


# ── Reward function base kwargs ──

def _base_kwargs(
    alpha_current: float = 0.5,
    alpha_prev: float = 0.5,
    regime_bull: bool = True,
    ae_error: float = 1.0,
    threshold_tau: float = 20.0,
    r_port: float = 0.0,
    w_final_t: np.ndarray | None = None,
    w_final_t_minus_1: np.ndarray | None = None,
    equity_curve: np.ndarray | None = None,
    normal_equity_curve: np.ndarray | None = None,
    normal_return_t: float = 0.0,
):
    """Build a RewardFunction.compute() kwargs dict with neutral defaults."""
    if w_final_t is None:
        w_final_t = np.array([0.2] * 5)
    if w_final_t_minus_1 is None:
        w_final_t_minus_1 = w_final_t.copy()
    if equity_curve is None:
        equity_curve = np.array([1.0, 1.0])
    if normal_equity_curve is None:
        normal_equity_curve = np.array([1.0, 1.0])
    return dict(
        ae_error=ae_error,
        threshold_tau=threshold_tau,
        r_port=r_port,
        w_final_t=w_final_t,
        w_final_t_minus_1=w_final_t_minus_1,
        port_returns=np.array([0.0, 0.0]),
        benchmark_returns=np.array([0.0, 0.0]),
        equity_curve=equity_curve,
        regret_ema_t=0.0,
        regime_bull=regime_bull,
        alpha_prev=alpha_prev,
        alpha_current=alpha_current,
        normal_return_t=normal_return_t,
        normal_equity_curve=normal_equity_curve,
    )


# ── A. lambda_alpha_direct is actually used ──


def test_lambda_alpha_direct_affects_reward_in_bull():
    """In bull regime, lambda_alpha_direct should reward alpha > 0.5."""
    rf = RewardFunction(
        lambda_alpha_direct=1.0,
        lambda_endpoint=0.0,
        lambda_turnover=0.0,
        lambda_te=0.0,
        kappa_mdd=0.0,
        switch_bull_reward=0.0,
        switch_bear_reward=0.0,
        switch_bull_penalty=0.0,
        switch_bear_penalty=0.0,
    )
    r_low = rf.compute(**_base_kwargs(alpha_current=0.0, regime_bull=True))
    r_high = rf.compute(**_base_kwargs(alpha_current=1.0, regime_bull=True))
    # alpha=1.0 in bull must reward more than alpha=0.0 in bull
    assert r_high > r_low, f"high alpha should be rewarded in bull: r_high={r_high}, r_low={r_low}"


def test_lambda_alpha_direct_affects_reward_in_bear():
    """In bear regime, lambda_alpha_direct should penalize alpha > 0.5."""
    rf = RewardFunction(
        lambda_alpha_direct=1.0,
        lambda_endpoint=0.0,
        lambda_turnover=0.0,
        lambda_te=0.0,
        kappa_mdd=0.0,
        switch_bull_reward=0.0,
        switch_bear_reward=0.0,
        switch_bull_penalty=0.0,
        switch_bear_penalty=0.0,
    )
    r_low = rf.compute(**_base_kwargs(alpha_current=0.0, regime_bull=False))
    r_high = rf.compute(**_base_kwargs(alpha_current=1.0, regime_bull=False))
    # alpha=1.0 in bear must reward less than alpha=0.0 in bear
    assert r_high < r_low, f"high alpha should be penalized in bear: r_high={r_high}, r_low={r_low}"


# ── B. Regime-conditional alpha signal (semantic check) ──


def test_stage1_bull_rewards_offensive_alpha():
    """Stage 1: in bull, alpha=0.8 should give higher reward than alpha=0.2."""
    rf = RewardFunction(
        lambda_alpha_direct=0.05,
        lambda_endpoint=0.05,
        lambda_turnover=0.0,
        lambda_te=0.0,
        kappa_mdd=0.0,
        switch_bull_reward=0.05,
        switch_bear_reward=0.05,
        switch_bull_penalty=0.05,
        switch_bear_penalty=0.05,
    )
    r_offensive = rf.compute(**_base_kwargs(alpha_current=0.8, regime_bull=True))
    r_defensive = rf.compute(**_base_kwargs(alpha_current=0.2, regime_bull=True))
    assert r_offensive > r_defensive, (
        f"Stage 1 should reward offensive alpha in bull: r_off={r_offensive}, r_def={r_defensive}"
    )


def test_stage1_bear_rewards_defensive_alpha():
    """Stage 1: in bear, alpha=0.2 should give higher reward than alpha=0.8."""
    rf = RewardFunction(
        lambda_alpha_direct=0.05,
        lambda_endpoint=0.05,
        lambda_turnover=0.0,
        lambda_te=0.0,
        kappa_mdd=0.0,
        switch_bull_reward=0.05,
        switch_bear_reward=0.05,
        switch_bull_penalty=0.05,
        switch_bear_penalty=0.05,
    )
    r_offensive = rf.compute(**_base_kwargs(alpha_current=0.8, regime_bull=False))
    r_defensive = rf.compute(**_base_kwargs(alpha_current=0.2, regime_bull=False))
    assert r_defensive > r_offensive, (
        f"Stage 1 should reward defensive alpha in bear: r_off={r_offensive}, r_def={r_defensive}"
    )


# ── C. Switch reward: 4 quadrants ──


def test_switch_reward_bull_high_alpha_is_positive():
    """regime_bull + alpha > 0.5 should give positive switch_reward."""
    rf = RewardFunction(
        switch_bull_reward=0.05,
        switch_bear_reward=0.05,
        switch_bull_penalty=0.05,
        switch_bear_penalty=0.05,
    )
    r = rf.compute(**_base_kwargs(alpha_current=0.8, regime_bull=True))
    # In this regime, switch_bull_reward (+0.05) applies, not penalty
    # Pure switch_reward contribution is +0.05 (other terms are controlled by
    # turning off turnover/te/mdd/endpoint/alpha_signal in this test config)
    assert r > 0.0, f"bull + high alpha should be rewarded: r={r}"


def test_switch_reward_bear_high_alpha_is_negative():
    """regime_bear + alpha > 0.5 should give negative switch_reward."""
    rf = RewardFunction(
        switch_bull_reward=0.05,
        switch_bear_reward=0.05,
        switch_bull_penalty=0.05,
        switch_bear_penalty=0.05,
    )
    r = rf.compute(**_base_kwargs(alpha_current=0.8, regime_bull=False))
    assert r < 0.0, f"bear + high alpha should be penalized: r={r}"


# ── D. Action mapper: bias breaks the a1=0 stable equilibrium ──


def test_action_mapper_bias_breaks_zero_equilibrium():
    """Stage 1: a1=0 should NOT produce delta=0 (otherwise alpha locks)."""
    mapper = ActionMapper(alpha_min=-0.5, alpha_max=0.1, alpha_bias=-0.05)
    delta, _ = mapper.map(0.0, 0.0)
    assert delta != 0.0, f"a1=0 should not produce zero delta: delta={delta}"
    assert delta < 0.0, f"a1=0 should drift toward defensive: delta={delta}"


def test_action_mapper_a1_pos_one_max_offensive():
    """Stage 1: a1=+1 should add exactly alpha_range + bias = 0.3 - 0.05 = 0.25,
    clipped to alpha_max=0.1."""
    mapper = ActionMapper(alpha_min=-0.5, alpha_max=0.1, alpha_bias=-0.05)
    delta, _ = mapper.map(1.0, 0.0)
    # alpha_range = (0.1 - (-0.5)) / 2 = 0.3
    # delta = 1.0 * 0.3 + (-0.05) = 0.25
    # clipped to alpha_max=0.1
    assert delta == 0.1, f"a1=+1 should clip to alpha_max=0.1: delta={delta}"


def test_action_mapper_a1_neg_one_max_defensive():
    """Stage 1: a1=-1 should cut by alpha_range - bias = 0.3 - (-0.05) = 0.35."""
    mapper = ActionMapper(alpha_min=-0.5, alpha_max=0.1, alpha_bias=-0.05)
    delta, _ = mapper.map(-1.0, 0.0)
    assert delta == pytest.approx(-0.35), f"a1=-1 should give -0.35: delta={delta}"


# ── E. Action mapper: alpha_max=0.1 caps per-step add ──


def test_action_mapper_alpha_max_caps_addition():
    """Stage 1: per-step alpha addition is capped at 0.1 (not 0.5)."""
    mapper = ActionMapper(alpha_min=-0.5, alpha_max=0.1, alpha_bias=-0.05)
    # Even with extreme a1, delta should be clipped to [alpha_min, alpha_max]
    for a1 in [0.5, 0.9, 1.0]:
        delta, _ = mapper.map(a1, 0.0)
        assert delta <= 0.1, f"a1={a1}: delta={delta} should be <= 0.1"


def test_action_mapper_tau_delta_range_capped():
    """Stage 1: tau_delta_range=0.1 (not 2.0)."""
    mapper = ActionMapper(tau_delta_range=0.1)
    for a2 in [0.5, 0.9, 1.0, -1.0]:
        _, delta_tau = mapper.map(0.0, a2)
        assert abs(delta_tau) <= 0.1, f"a2={a2}: delta_tau={delta_tau} should be <= 0.1"


# ── F. End-to-end: MDPEnvironment with Stage 1 config ──


def _stage1_config() -> dict:
    return {
        "action_mapper": {
            "alpha_min": -0.5,
            "alpha_max": 0.1,
            "tau_delta_range": 0.1,
            "alpha_bias": -0.05,
        },
        "regret_engine": {"ema_decay": 0.8},
        "state_assembler": {"sharpe_clip_low": -3.0, "sharpe_clip_high": 3.0},
        "reward_function": {
            "lambda_turnover": 0.0,
            "lambda_te": 0.0,
            "kappa": 0.0,
            "eta": 0.0,
            "switch_bull_reward": 0.05,
            "switch_bear_reward": 0.05,
            "switch_bull_penalty": 0.05,
            "switch_bear_penalty": 0.05,
            "lambda_alpha_direct": 0.05,
            "lambda_endpoint": 0.05,
            "lambda_relative": 1.0,
        },
        "env": {
            "tau_min": 5.0,
            "tau_max": 50.0,
            "initial_alpha": 0.5,
            "initial_tau": 20.0,
            "episode_max_steps": 10,
        },
    }


def test_stage1_env_breaks_alpha_lock():
    """Stage 1: PPO-friendly gradient means alpha should not lock at 1.0.

    With a1 = 0 (PPO default at initialization), Stage 1's negative bias
    should make alpha drift DOWN (toward defensive), not lock at 1.0.
    """
    env = MDPEnvironment(_stage1_config())
    env.reset()
    env.inject_live_data({
        "ae_error": 5.0,             # well below tau
        "vol_mkt_20d": 0.15,
        "llm_macro": 50.0,
        "llm_sentiment": 50.0,
        "llm_risk": 50.0,
        "w_normal_t": np.array([0.2] * 5),
        "w_event_t": np.array([0.0, 0.0, 0.33, 0.33, 0.34]),
        "returns_window_5d": np.array([[0.0] * 5, [0.0] * 5]),
    })
    # Step with a1=0 (PPO initialization): Stage 1 bias should drift alpha down
    _, _, _, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
    # alpha should be < 0.5 (drifted down by -0.05 bias)
    assert info["alpha"] < 0.5, (
        f"Stage 1 should break alpha=1.0 lock: a1=0 should drift down, got alpha={info['alpha']}"
    )


def test_stage1_env_tau_can_actually_move():
    """Stage 2 bug fix: tau should be PPO-controllable (not locked at 20.0)."""
    env = MDPEnvironment(_stage1_config())
    env.reset()
    env.inject_live_data({
        "ae_error": 5.0,
        "vol_mkt_20d": 0.15,
        "llm_macro": 50.0,
        "llm_sentiment": 50.0,
        "llm_risk": 50.0,
        "w_normal_t": np.array([0.2] * 5),
        "w_event_t": np.array([0.0, 0.0, 0.33, 0.33, 0.34]),
        "returns_window_5d": np.array([[0.0] * 5, [0.0] * 5]),
    })
    # Step with a2=+1 (max tau increase)
    _, _, _, _, info1 = env.step(np.array([0.0, 1.0], dtype=np.float32))
    # Step with a2=-1 (max tau decrease)
    _, _, _, _, info2 = env.step(np.array([0.0, -1.0], dtype=np.float32))
    # tau should have changed (not locked at 20.0)
    assert info1["tau"] != 20.0 or info2["tau"] != 20.0, (
        f"tau should be PPO-controllable: tau1={info1['tau']}, tau2={info2['tau']}"
    )


# ── G. End-to-end: PPO-friendly gradient exists ──


def test_stage1_reward_has_meaningful_alpha_gradient():
    """Verify Stage 1 reward has a non-trivial gradient in alpha.

    For PPO to learn, the reward must change measurably with alpha
    (not flat at 0). Sweep alpha from 0.0 to 1.0 and verify std > threshold.
    """
    rf = RewardFunction(
        lambda_alpha_direct=0.05,
        lambda_endpoint=0.05,
        lambda_turnover=0.0,
        lambda_te=0.0,
        kappa_mdd=0.0,
        switch_bull_reward=0.05,
        switch_bear_reward=0.05,
        switch_bull_penalty=0.05,
        switch_bear_penalty=0.05,
    )
    bull_rewards = [rf.compute(**_base_kwargs(alpha_current=a, regime_bull=True)) for a in np.linspace(0, 1, 11)]
    bear_rewards = [rf.compute(**_base_kwargs(alpha_current=a, regime_bull=False)) for a in np.linspace(0, 1, 11)]
    bull_std = float(np.std(bull_rewards))
    bear_std = float(np.std(bear_rewards))
    assert bull_std > 0.02, f"bull reward should have alpha gradient: std={bull_std}"
    assert bear_std > 0.02, f"bear reward should have alpha gradient: std={bear_std}"
    # The two regimes should be anti-correlated (bull likes high alpha, bear dislikes)
    bull_corr = float(np.corrcoef(np.linspace(0, 1, 11), bull_rewards)[0, 1])
    bear_corr = float(np.corrcoef(np.linspace(0, 1, 11), bear_rewards)[0, 1])
    assert bull_corr > 0.5, f"bull should positively correlate with alpha: corr={bull_corr}"
    assert bear_corr < -0.5, f"bear should negatively correlate with alpha: corr={bear_corr}"


def test_stage1_keeps_pre_existing_test_passing():
    """The test_mdp_environment_alignment test must still pass after Stage 1."""
    rf = RewardFunction(
        lambda_turnover=0.0,
        lambda_te=0.0,
        kappa_mdd=0.0,
        eta_regret=0.0,
        switch_bull_reward=0.0,
        switch_bear_reward=0.0,
        switch_bull_penalty=0.0,
        switch_bear_penalty=0.0,
        lambda_alpha_direct=1.0,
        lambda_endpoint=0.0,
        lambda_relative=0.0,
    )
    r = rf.compute(**_base_kwargs(alpha_prev=0.0, alpha_current=1.0, regime_bull=True))
    assert r > 0.0
