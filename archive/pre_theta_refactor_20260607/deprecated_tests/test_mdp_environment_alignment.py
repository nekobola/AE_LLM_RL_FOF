import numpy as np

from src.env.mdp_environment import MDPEnvironment
from src.env.reward_function import RewardFunction


def make_config() -> dict:
    return {
        "action_mapper": {
            "alpha_min": -0.5,
            "alpha_max": 0.5,
            "tau_delta_range": 1.0,
            "alpha_bias": 0.0,
        },
        "regret_engine": {"ema_decay": 0.8},
        "state_assembler": {
            "sharpe_clip_low": -3.0,
            "sharpe_clip_high": 3.0,
        },
        "reward_function": {
            "lambda_turnover": 0.0,
            "lambda_te": 0.0,
            "kappa": 0.0,
            "eta": 0.0,
            "switch_bull_reward": 0.0,
            "switch_bear_reward": 0.0,
            "switch_bull_penalty": 0.0,
            "switch_bear_penalty": 0.0,
            "lambda_alpha_direct": 0.0,
            "lambda_endpoint": 0.0,
            "lambda_relative": 1.0,
        },
        "env": {
            "tau_min": 0.0,
            "tau_max": 50.0,
            "initial_alpha": 0.5,
            "initial_tau": 20.0,
            "episode_max_steps": 10,
        },
    }


def test_reward_function_uses_current_alpha_not_previous_alpha():
    reward_fn = RewardFunction(
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

    reward = reward_fn.compute(
        ae_error=1.0,
        threshold_tau=2.0,
        r_port=0.0,
        w_final_t=np.array([0.2] * 5),
        w_final_t_minus_1=np.array([0.2] * 5),
        port_returns=np.array([0.0, 0.0]),
        benchmark_returns=np.array([0.0, 0.0]),
        equity_curve=np.array([1.0, 1.0]),
        regret_ema_t=0.0,
        regime_bull=True,
        alpha_prev=0.0,
        alpha_current=1.0,
        normal_return_t=0.0,
        normal_equity_curve=np.array([1.0, 1.0]),
    )

    assert reward > 0.0


def test_environment_reward_depends_on_fused_weights():
    env = MDPEnvironment(make_config())
    env.reset()

    live_data = {
        "ae_error": 30.0,
        "vol_mkt_20d": 0.15,
        "llm_macro": 50.0,
        "llm_sentiment": 50.0,
        "llm_risk": 80.0,
        "w_normal_t": np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        "w_event_t": np.array([0.0, 1.0, 0.0, 0.0, 0.0]),
        "returns_window_5d": np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.1, 0.0, 0.0, 0.0],
            ]
        ),
    }
    env.inject_live_data(live_data)
    _, reward_high_alpha, _, _, info_high = env.step(np.array([1.0, 0.0], dtype=np.float32))

    env = MDPEnvironment(make_config())
    env.reset()
    env.inject_live_data(live_data)
    _, reward_low_alpha, _, _, info_low = env.step(np.array([-1.0, 0.0], dtype=np.float32))

    assert info_high["r_port"] > info_low["r_port"]
    assert reward_high_alpha > reward_low_alpha


def test_reward_is_independent_of_regret_penalty_now():
    reward_fn = RewardFunction(
        lambda_turnover=0.0,
        lambda_te=0.0,
        kappa_mdd=0.0,
        eta_regret=100.0,
        switch_bull_reward=0.0,
        switch_bear_reward=0.0,
        switch_bull_penalty=0.0,
        switch_bear_penalty=0.0,
        lambda_alpha_direct=0.0,
        lambda_endpoint=0.0,
        lambda_relative=1.0,
    )

    kwargs = dict(
        ae_error=30.0,
        threshold_tau=20.0,
        r_port=0.05,
        w_final_t=np.array([0.2] * 5),
        w_final_t_minus_1=np.array([0.2] * 5),
        port_returns=np.array([0.01, 0.02]),
        benchmark_returns=np.array([0.00, 0.01]),
        equity_curve=np.array([1.0, 1.05]),
        regime_bull=False,
        alpha_prev=0.5,
        alpha_current=0.5,
        normal_return_t=0.01,
        normal_equity_curve=np.array([1.0, 1.01]),
    )

    reward_low_regret = reward_fn.compute(regret_ema_t=0.0, **kwargs)
    reward_high_regret = reward_fn.compute(regret_ema_t=999.0, **kwargs)

    assert reward_low_regret == reward_high_regret
