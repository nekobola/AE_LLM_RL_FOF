"""Environment Module

PPO MDP environment for AE-LLM-RL-FOF.
Gymnasium-compatible with custom regime-conditional reward.
"""
from src.env.action_mapper import ActionMapper
from src.env.metrics_utils import (
    calculate_tracking_error,
    calculate_current_drawdown,
    calculate_current_drawdown_incremental,
    calculate_sharpe_ratio,
)
from src.env.regret_engine import RegretEngine
from src.env.reward_function import RewardFunction
from src.env.state_assembler import StateAssembler, StateTuple
from src.env.mdp_environment import MDPEnvironment

__all__ = [
    "ActionMapper",
    "calculate_tracking_error",
    "calculate_current_drawdown",
    "calculate_current_drawdown_incremental",
    "calculate_sharpe_ratio",
    "RegretEngine",
    "RewardFunction",
    "StateAssembler",
    "StateTuple",
    "MDPEnvironment",
]
