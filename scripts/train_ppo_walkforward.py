"""Walk-forward PPO 训练 (Stage 7c 8-ETF pipeline).

核心思路 (C 选项):
  - 对每个 quarter_end Q, 在 [Q - lookback_weeks, Q - 1] 数据上训练一个新的 PPO
  - 保存为 checkpoints/walkforward/actor_critic_q{N}_{year}.pth
  - WFO 在 Q 季度内使用该 PPO, 下一季度切换到更新的 PPO
  - 真正 out-of-sample: 测试 PPO 在 Q 时没看过 Q 之后的数据

训练数据:
  - features: 5-dim market features (from features_master.parquet)
  - 8 ETF weekly returns: 真实 ClickHouse 业绩 (与生产 WFO 一致)
  - 训练 reward = dot(w_event_8d, etf_returns_8d)
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_ppo import (
    inject_live_data_from_history,
    load_config,
)
from src.env.mdp_environment import MDPEnvironment
from src.models.regime_autoencoder import RegimeAutoEncoder
from src.ppo.buffer import RolloutBuffer
from src.ppo.networks import ActorCritic
from src.ppo.trainer import PPOTrainer

(PROJECT_ROOT / "logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / "train_ppo_walkforward.log", mode="a"),
    ],
)
log = logging.getLogger("train_ppo_wf")


def get_quarter_ends(start_date: str, end_date: str) -> list[tuple[int, int, pd.Timestamp]]:
    """返回 (year, quarter, last_friday) 列表, 在 [start_date, end_date] 范围内."""
    fridays = pd.bdate_range(start=start_date, end=end_date, freq="W-FRI")
    quarter_ends = []
    seen = set()
    for f in fridays:
        q = (f.year, (f.month - 1) // 3 + 1)
        if q not in seen and f.month % 3 == 0 and f.day >= 25:
            quarter_ends.append((f.year, q[1], pd.Timestamp(f)))
            seen.add(q)
    return quarter_ends


def load_features(features_path: Path) -> Optional[pd.DataFrame]:
    if not features_path.exists():
        return None
    df = pd.read_parquet(features_path)
    return df


def load_llm_scores(llm_db_path: Path) -> Optional[pd.DataFrame]:
    if not llm_db_path.exists():
        return None
    conn = sqlite3.connect(llm_db_path)
    df = pd.read_sql(
        "SELECT week_end, concept, d1, d2, d3 FROM llm_scores WHERE error IS NULL",
        conn,
    )
    conn.close()
    return df


def load_ae_model(
    ae_path: Path,
    features_df: pd.DataFrame,
    config: dict,
    device: str,
):
    if not ae_path.exists():
        return None
    model_cfg = config.get("model", {}).get("regime_autoencoder", {})
    model = RegimeAutoEncoder(
        input_dim=features_df.shape[1],
        latent_dim=model_cfg.get("latent_dim", 6),
        hidden_dim=model_cfg.get("hidden_dim", 16),
    ).to(device)
    ckpt = torch.load(ae_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt.get("model_state", ckpt))
    model.eval()
    return model


def train_one_quarter(
    features_slice: pd.DataFrame,
    llm_slice: Optional[pd.DataFrame],
    ae_model,
    scaler_state,
    config: dict,
    device: str,
    quarter_label: str,
    timesteps: int,
    init_checkpoint: Optional[Path] = None,
) -> Path:
    """Train PPO on a slice of data. Returns checkpoint path.

    Warm-start: 如果 init_checkpoint 给定, 从该 checkpoint 加载 ac 权重继续训练 (优化器重置).
    Cold-start: 如果 init_checkpoint 为 None, 随机初始化.
    """
    # Build env
    env = MDPEnvironment(config)

    # Build network
    ppo_cfg = config.get("ppo", {})
    state_dim = ppo_cfg.get("state_dim", 9)
    action_dim = ppo_cfg.get("action_dim", 1)
    ac = ActorCritic(state_dim=state_dim, action_dim=action_dim, hidden_dim=64).to(device)

    # Warm-start: load ac weights from init_checkpoint
    if init_checkpoint is not None and init_checkpoint.exists():
        log.info(f"  [{quarter_label}] warm-start from {init_checkpoint.name}")
        prev = torch.load(init_checkpoint, map_location=device, weights_only=False)
        ac.load_state_dict(prev["ac"])
    else:
        log.info(f"  [{quarter_label}] cold-start (随机初始化)")

    # Build buffer + trainer (优化器总是新 init, 避免 Adam moments 跨季度漂移)
    buffer_size = ppo_cfg.get("buffer_size", 100)
    buffer = RolloutBuffer(buffer_size=buffer_size, state_dim=state_dim, action_dim=action_dim)
    trainer = PPOTrainer(actor_critic=ac, config=config, device=device, buffer=buffer)

    # Build live_data list
    if len(features_slice) < 30:
        log.warning(f"  [{quarter_label}] 数据不足 ({len(features_slice)} days), 跳过")
        return None

    log.info(f"  [{quarter_label}] 准备 live_data (features={features_slice.shape}, llm={None if llm_slice is None else len(llm_slice)})...")
    live_data_list = inject_live_data_from_history(
        env=env,
        features_df=features_slice,
        llm_scores_df=llm_slice,
        ae_model=ae_model,
        scaler_state=scaler_state,
        device=device,
        config=config,
    )
    if not live_data_list:
        log.warning(f"  [{quarter_label}] live_data 为空, 跳过")
        return None

    n_updates = timesteps // buffer_size
    log.info(f"  [{quarter_label}] 训练 PPO: {n_updates} updates × {buffer_size} steps = {timesteps} total")

    data_idx = 0
    t_start = time.time()
    for ppo_iter in range(n_updates):
        data_idx = trainer.collect_rollout_manual(
            env=env,
            live_data_list=live_data_list,
            data_start_idx=data_idx,
        )
        loss_stats = trainer.update()
        if (ppo_iter + 1) % max(1, n_updates // 10) == 0:
            elapsed = time.time() - t_start
            log.info(
                f"  [{quarter_label}] PPO iter {ppo_iter+1}/{n_updates}  "
                f"loss_total={loss_stats['loss_total']:.4f}  elapsed={elapsed:.0f}s"
            )

    # Save checkpoint
    ckpt_dir = PROJECT_ROOT / "checkpoints" / "walkforward"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"actor_critic_{quarter_label}.pth"
    torch.save(
        {
            "ac": ac.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "step_count": n_updates * buffer_size,
            "quarter_label": quarter_label,
        },
        ckpt_path,
    )
    log.info(f"  [{quarter_label}] saved: {ckpt_path}")
    return ckpt_path


def main() -> None:
    parser = argparse.ArgumentParser(description="train_ppo_walkforward: 季度重训 PPO")
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2026-04-30")
    parser.add_argument("--lookback-weeks", type=int, default=104, help="训练窗口 (weeks)")
    parser.add_argument("--timesteps-per-quarter", type=int, default=20000)
    parser.add_argument("--burn-in-weeks", type=int, default=104, help="跳过前 N 周 (训练数据不足)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default="checkpoints/walkforward")
    parser.add_argument(
        "--init-checkpoint",
        default="checkpoints/actor_critic_oos.pth",
        help="warm-start 起始 ckpt (Q1 用这个, Q2+ 自动用上一季度 ckpt)",
    )
    parser.add_argument(
        "--no-warm-start", action="store_true",
        help="禁用 warm-start, 每个季度都随机初始化 (cold-start)",
    )
    parser.add_argument(
        "--init-end-date", default="2022-12-30",
        help="init PPO 训练结束日, 该日期之前的 quarter 全部 skip (避免 pre-WFO 数据浪费训练时间)",
    )
    args = parser.parse_args()

    config = load_config()
    paths_cfg = config.get("paths", {})

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"══ train_ppo_walkforward 启动 ══")
    log.info(f"  start_date={args.start_date}, end_date={args.end_date}")
    log.info(f"  lookback_weeks={args.lookback_weeks}, burn_in_weeks={args.burn_in_weeks}")
    log.info(f"  timesteps/quarter={args.timesteps_per_quarter}, device={device}")

    # Load features
    features_path = PROJECT_ROOT / paths_cfg.get("data_processed", "data/processed") / "features_master.parquet"
    features_df = load_features(features_path)
    if features_df is None:
        log.error(f"features_master 不存在: {features_path}")
        sys.exit(1)
    log.info(f"features_master: {features_df.shape}, {features_df.index.min().date()} ~ {features_df.index.max().date()}")

    # Load LLM scores
    llm_db_path = PROJECT_ROOT / "data" / "llm_cache" / "llm_scores.db"
    llm_df = load_llm_scores(llm_db_path)
    log.info(f"LLM scores: {None if llm_df is None else len(llm_df)}")

    # Load AE model
    ae_path = PROJECT_ROOT / paths_cfg.get("checkpoints", "checkpoints") / "ae_weights.pth"
    scaler_path = PROJECT_ROOT / paths_cfg.get("checkpoints", "checkpoints") / "ae_scaler.pkl"
    scaler_state = None
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler_state = pickle.load(f)
    ae_model = load_ae_model(ae_path, features_df, config, device)
    log.info(f"AE model: {'loaded' if ae_model is not None else 'missing'}")

    # Get quarter ends
    quarter_ends = get_quarter_ends(args.start_date, args.end_date)
    log.info(f"Quarter ends: {len(quarter_ends)}")
    for year, q, friday in quarter_ends:
        log.info(f"  Q{q} {year}: {friday.date()}")

    # Iterate: for each quarter, train PPO on [Q - lookback, Q - 1]
    burn_in_end = pd.Timestamp(args.start_date) + pd.DateOffset(weeks=args.burn_in_weeks)
    init_end = pd.Timestamp(args.init_end_date)
    ckpt_paths = []
    # 第一个季度用 init-checkpoint, 之后每个季度用上一季度的 ckpt (warm-start)
    prev_ckpt: Optional[Path] = None
    init_ckpt = None if args.no_warm_start else Path(args.init_checkpoint)
    if init_ckpt and init_ckpt.exists():
        log.info(f"  Warm-start 起始 ckpt: {init_ckpt}")
    else:
        log.info(f"  Warm-start 关闭 (init_ckpt={init_ckpt})")

    for year, q, last_friday in quarter_ends:
        # Skip quarters before init_end_date (pre-WFO, init PPO 已训练过)
        if last_friday <= init_end:
            log.info(f"  Q{q} {year} ({last_friday.date()}) <= init_end_date, skip")
            continue
        if last_friday <= burn_in_end:
            log.info(f"  Q{q} {year} ({last_friday.date()}) <= burn-in end, skip")
            continue
        quarter_label = f"q{q}_{year}"
        train_end = last_friday - pd.DateOffset(days=1)
        train_start = train_end - pd.DateOffset(weeks=args.lookback_weeks)

        # Slice features_df
        feat_mask = (features_df.index >= train_start) & (features_df.index <= train_end)
        features_slice = features_df[feat_mask]
        # Slice llm
        llm_slice = None
        if llm_df is not None:
            llm_df_copy = llm_df.copy()
            llm_df_copy["week_end"] = pd.to_datetime(llm_df_copy["week_end"])
            llm_mask = (llm_df_copy["week_end"] >= train_start) & (llm_df_copy["week_end"] <= train_end)
            llm_slice = llm_df_copy[llm_mask]

        if len(features_slice) < 50:
            log.warning(f"  [{quarter_label}] features 不足 ({len(features_slice)}), skip")
            continue

        # 决定这个季度的 init_checkpoint
        if args.no_warm_start:
            quarter_init = None  # cold-start
        elif prev_ckpt is None:
            quarter_init = init_ckpt  # Q1 用外部 init
        else:
            quarter_init = prev_ckpt  # Q2+ 用上一季度 ckpt

        # 第一个 quarter (prev_ckpt is None) 不训练, 直接 copy init_ckpt
        # 原因: 它的 train window [t-104w, t-1w] 包含 t (test period), 会有 look-ahead
        # 解决: Q1 用 init_ckpt (= 回测期前 PPO), Q2+ 再 walk-forward warm-start
        if prev_ckpt is None and not args.no_warm_start and quarter_init is not None and quarter_init.exists():
            ckpt_dir = PROJECT_ROOT / "checkpoints" / "walkforward"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"actor_critic_{quarter_label}.pth"
            import shutil
            shutil.copy(quarter_init, ckpt_path)
            log.info(
                f"  ╔══ Q{q} {year}: 直接 copy {quarter_init.name} → {ckpt_path.name} "
                f"(无训练, 避免 look-ahead) ══"
            )
            ckpt_paths.append((quarter_label, last_friday, ckpt_path))
            prev_ckpt = ckpt_path
            continue

        log.info(
            f"  ╔══ Q{q} {year}: train on {train_start.date()} ~ {train_end.date()} "
            f"({len(features_slice)} days) ══"
        )
        ckpt = train_one_quarter(
            features_slice=features_slice,
            llm_slice=llm_slice,
            ae_model=ae_model,
            scaler_state=scaler_state,
            config=config,
            device=device,
            quarter_label=quarter_label,
            timesteps=args.timesteps_per_quarter,
            init_checkpoint=quarter_init,
        )
        if ckpt is not None:
            ckpt_paths.append((quarter_label, last_friday, ckpt))
            prev_ckpt = ckpt  # 下个季度的 init

    log.info(f"══ walk-forward 训练完成: {len(ckpt_paths)} 个 checkpoints ══")
    for label, friday, path in ckpt_paths:
        log.info(f"  {label} ({friday.date()}): {path}")


if __name__ == "__main__":
    main()
