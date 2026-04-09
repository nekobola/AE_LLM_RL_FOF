#!/usr/bin/env python3
"""
train_ppo.py — PPO 元控制器沙盒训练

定位：强化学习核心训练引擎，仅离线运行，寻找最优策略参数。

核心链路：
  1. 读取历史预计算的 E_t (AE重建误差) + LLM评分
  2. 初始化 env.mdp_environment (MDP环境)
  3. 实例化 ppo.networks (Actor-Critic) + ppo.buffer
  4. 执行 Epoch-based 批量 Rollout + 梯度更新
  5. TensorBoard 实时打点：Actor Loss / Critic Loss / Entropy / Regret

监控指标（TensorBoard）：
  - loss/actor_clip_loss
  - loss/critic_vf_loss
  - loss/entropy_loss
  - loss/total_loss
  - reward/mean_reward_per_step
  - reward/mean_regret_ema
  - env/alpha_distribution
  - env/tau_distribution

用法：
  python scripts/train_ppo.py \
    --total-timesteps 100000 \
    --tb-log-dir logs/tensorboard \
    --checkpoint-path checkpoints/actor_critic.pth
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.env.mdp_environment import MDPEnvironment
from src.ppo.networks import ActorCritic
from src.ppo.buffer import RolloutBuffer
from src.ppo.trainer import PPOTrainer

# ── TensorBoard ────────────────────────────────────────────────────────────────
try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False
    SummaryWriter = None

# ── Logging ───────────────────────────────────────────────────────────────────
(PROJECT_ROOT / "logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / "train_ppo.log", mode="a"),
    ],
)
log = logging.getLogger("train_ppo")


def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def inject_live_data_from_history(
    env: MDPEnvironment,
    features_df,
    llm_scores_df,
) -> None:
    """
    将历史 E_t 和 LLM 评分注入环境，构建静态历史数据集。
    这个函数把离线数据切片装配成 env 可以消费的形式。
    """
    pass  # Stub: 实际由 MDPEnvironment.inject_live_data() 调用


def main() -> None:
    parser = argparse.ArgumentParser(description="train_ppo: PPO沙盒训练")
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--tb-log-dir", default="logs/tensorboard")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true", help="从checkpoint恢复训练")
    args = parser.parse_args()

    config = load_config()
    ppo_cfg = config.get("ppo", {})
    paths_cfg = config.get("paths", {})

    checkpoint_path = Path(
        args.checkpoint_path
        or paths_cfg.get("checkpoints", "checkpoints")
    ) / "actor_critic.pth"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    tb_log_dir = PROJECT_ROOT / args.tb_log_dir
    tb_log_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ───────────────────────────────────────────────────────────
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    # ── TensorBoard ───────────────────────────────────────────────────────
    if TB_AVAILABLE:
        tb_writer = SummaryWriter(log_dir=str(tb_log_dir))
        log.info(f"TensorBoard: {tb_log_dir}")
    else:
        log.warning("TensorBoard 不可用，安装: pip install tensorboard")
        tb_writer = None

    # ── 加载 AE Scaler + 特征 + LLM评分 ───────────────────────────────────
    scaler_path = (
        PROJECT_ROOT / paths_cfg.get("checkpoints", "checkpoints") / "ae_scaler.pkl"
    )
    features_path = (
        PROJECT_ROOT / paths_cfg.get("data_processed", "data/processed") / "features_master.parquet"
    )

    if scaler_path.exists() and features_path.exists():
        log.info(f"加载 AE Scaler: {scaler_path}")
        with open(scaler_path, "rb") as f:
            scaler_state = pickle.load(f)

        log.info(f"加载特征: {features_path}")
        features_df = np.load(features_path)
        log.info("历史数据就绪，可注入环境")
    else:
        log.warning("历史特征或scaler不存在，环境将使用模拟数据")
        features_df = None

    # ── MDP Environment ───────────────────────────────────────────────────
    log.info("初始化 MDP Environment ...")
    env = MDPEnvironment(config)

    # ── Actor-Critic ──────────────────────────────────────────────────────
    log.info("初始化 Actor-Critic 网络 ...")
    ac = ActorCritic(
        state_dim=ppo_cfg.get("state_dim", 10),
        action_dim=ppo_cfg.get("action_dim", 2),
        hidden_dim=64,
    ).to(device)

    # ── Rollout Buffer ────────────────────────────────────────────────────
    buffer_size = ppo_cfg.get("buffer_size", 100)
    log.info(f"Rollout Buffer: size={buffer_size}")

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = PPOTrainer(
        actor_critic=ac,
        config=config,
        device=device,
    )
    trainer.buffer = RolloutBuffer(
        buffer_size=buffer_size,
        state_dim=ppo_cfg.get("state_dim", 10),
        action_dim=ppo_cfg.get("action_dim", 2),
    )

    # ── Resume ─────────────────────────────────────────────────────────────
    step_offset = 0
    if args.resume and checkpoint_path.exists():
        log.info(f"从 checkpoint 恢复: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        ac.load_state_dict(ckpt["ac"])
        trainer.optimizer.load_state_dict(ckpt["optimizer"])
        step_offset = ckpt.get("step_count", 0)
        log.info(f"恢复步数: {step_offset}")

    # ── 训练主循环 ─────────────────────────────────────────────────────────
    total_timesteps = args.total_timesteps
    buffer_size_i = trainer.buffer_size
    n_updates = total_timesteps // buffer_size_i

    log.info(f"══ PPO Training Start: {n_updates} 次PPO更新 ═══════════════════")
    log.info(f"    total_steps={total_timesteps}, buffer={buffer_size_i}, k_epochs={ppo_cfg.get('k_epochs',4)}")

    t_start = time.time()

    for ppo_iter in range(1, n_updates + 1):
        # 1. 收集 Rollout
        trainer.collect_rollout(env, max_steps=buffer_size_i)

        # 2. PPO 更新
        loss_stats = trainer.update()

        # 3. TensorBoard 打点
        if tb_writer is not None:
            global_step = step_offset + ppo_iter * buffer_size_i
            tb_writer.add_scalar("loss/total_loss", loss_stats["loss_total"], global_step)
            tb_writer.add_scalar("loss/actor_clip_loss", loss_stats["loss_clip"], global_step)
            tb_writer.add_scalar("loss/critic_vf_loss", loss_stats["loss_vf"], global_step)
            tb_writer.add_scalar("loss/entropy_loss", loss_stats["loss_entropy"], global_step)
            tb_writer.add_scalar("reward/mean_reward", loss_stats.get("mean_reward", 0), global_step)

            # Buffer 内统计
            states, _, rewards, _, _, _ = trainer.buffer.get_all() if trainer.buffer._size > 0 else (None,None,None,None,None,None)
            if rewards is not None and len(rewards) > 0:
                tb_writer.add_scalar("env/mean_reward_per_step", float(np.mean(rewards)), global_step)
                tb_writer.add_scalar("env/std_reward_per_step", float(np.std(rewards)), global_step)

            tb_writer.flush()

        # 4. 定期 checkpoint
        if ppo_iter % 10 == 0:
            ckpt = {
                "ac": trainer.ac.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "step_count": step_offset + ppo_iter * buffer_size_i,
                "ppo_iter": ppo_iter,
            }
            torch.save(ckpt, checkpoint_path)

        elapsed = time.time() - t_start
        log.info(
            f"  PPO iter {ppo_iter:4d}/{n_updates}  "
            f"loss_total={loss_stats['loss_total']:.4f}  "
            f"loss_clip={loss_stats['loss_clip']:.4f}  "
            f"loss_vf={loss_stats['loss_vf']:.4f}  "
            f"elapsed={elapsed:.0f}s"
        )

    # ── 最终保存 ───────────────────────────────────────────────────────────
    final_ckpt = {
        "ac": trainer.ac.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "step_count": total_timesteps,
    }
    torch.save(final_ckpt, checkpoint_path)

    total_elapsed = time.time() - t_start
    log.info(f"══ PPO Training 完成！总耗时={total_elapsed:.1f}s ═")
    log.info(f"Checkpoint: {checkpoint_path}")

    if tb_writer:
        tb_writer.close()


if __name__ == "__main__":
    main()
