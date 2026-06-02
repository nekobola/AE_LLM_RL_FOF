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
import pandas as pd
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
    with open(PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def inject_live_data_from_history(
    env: MDPEnvironment,
    features_df,
    llm_scores_df,
    ae_model,
    scaler_state,
    device: str,
    config: dict,
) -> list[dict]:
    """
    将历史 E_t 和 LLM 评分注入环境，构建静态历史数据集。
    
    Returns
    -------
    list[dict]
        每个时间步的 live_data 字典列表
    """
    import pandas as pd
    from src.features.reconstruction_error import compute_reconstruction_error
    
    live_data_list = []
    
    # 对齐 features 和 llm_scores 的日期
    # llm_scores_df 是周频(week_end, 10概念/周)，先按周聚合(均值)再向前填充到日频
    if llm_scores_df is not None and not llm_scores_df.empty:
        llm_scores_df = llm_scores_df.set_index("week_end")
        llm_scores_df.index = pd.to_datetime(llm_scores_df.index)
        llm_scores_df = llm_scores_df.sort_index()
        
        # 宽基指数集合
        wide_base = {"沪深300", "中证1000"}
        
        # 分别聚合宽基和satellite
        # d1/d2: 全市场平均; d3: 宽基均值*0.7 + satellite最大*0.3
        def smart_agg(group):
            d1 = group["d1"].mean()
            d2 = group["d2"].mean()
            
            wide = group[group["concept"].isin(wide_base)]
            satellite = group[~group["concept"].isin(wide_base)]
            
            wide_d3_mean = wide["d3"].mean() if len(wide) > 0 else 50.0
            satellite_d3_max = satellite["d3"].max() if len(satellite) > 0 else 50.0
            
            d3 = wide_d3_mean * 0.7 + satellite_d3_max * 0.3
            
            return pd.Series({"d1": d1, "d2": d2, "d3": d3})
        
        llm_agg = llm_scores_df.groupby(level="week_end").apply(smart_agg)
        
        # 向前填充周频评分到日频
        llm_scores_df = llm_agg.reindex(features_df.index, method="ffill")
    
    # 计算 AE 重建误差（如果模型可用）
    if ae_model is not None and scaler_state is not None:
        X = features_df.values
        ae_errors = compute_reconstruction_error_batch(ae_model, X, device)
    else:
        ae_errors = np.zeros(len(features_df))
    
    # 计算市场波动率（20日滚动）
    # 使用 features 中的资产波动率列（如果有的话）
    vol_col = None
    for col in features_df.columns:
        if "vol" in col.lower() or "volatility" in col.lower():
            vol_col = col
            break
    
    if vol_col is not None:
        vol_series = features_df[vol_col].rolling(20).std() * np.sqrt(252)
    else:
        vol_series = pd.Series(0.15, index=features_df.index)
    
    vol_series = vol_series.fillna(0.15)
    
    # ── 提取资产收益率列 ─────────────────────────────────────────────────
    # features_df 列格式: {code}__{feature_name}
    asset_codes = config.get("data_pipeline", {}).get("asset_codes", [
        "000300.SH", "000852.SH", "CBA02701.CS", "AU9999.SGE", "NH0100.NHF"
    ])
    weekly_return_cols = [f"{code}__weekly_return" for code in asset_codes]
    available_return_cols = [c for c in weekly_return_cols if c in features_df.columns]
    
    # 提取收益率矩阵 (T x N)
    if available_return_cols:
        returns_matrix = features_df[available_return_cols].fillna(0.0).values
    else:
        returns_matrix = np.zeros((len(features_df), 5))

    # ── DualTrackEngine: 预计算每日的 (w_normal_t, w_event_t) ──────────────
    from src.compute.dual_track_engine import DualTrackEngine
    dual_engine = DualTrackEngine(config)
    w_normal_series = []
    w_event_series = []
    for i in range(len(returns_matrix)):
        # DualTrackEngine.compute() expects (5, T)，取最近30天
        lookback = min(30, i)
        ret_5d = returns_matrix[max(0, i-lookback):i+1].T  # shape (5, T)
        # 提取 LLM 信号（用于 EventTrack 进攻决策）
        llm_macro = llm_sentiment = llm_risk = 50.0
        if llm_scores_df is not None and not llm_scores_df.empty:
            try:
                row_llm = llm_scores_df.iloc[i]
                llm_macro = float(row_llm.d1) if not pd.isna(row_llm.d1) else 50.0
                llm_sentiment = float(row_llm.d2) if not pd.isna(row_llm.d2) else 50.0
                llm_risk = float(row_llm.d3) if not pd.isna(row_llm.d3) else 50.0
            except Exception:
                pass
        if ret_5d.shape[1] < 5:
            # 数据不足，用等权填充
            w_normal_series.append(np.array([0.2]*5))
            w_event_series.append(np.array([0.2, 0.2, 0.2, 0.2, 0.2]))  # 进攻轨默认等权
        else:
            w_n, w_e = dual_engine.compute(
                ret_5d,
                llm_macro=llm_macro,
                llm_sentiment=llm_sentiment,
                llm_risk=llm_risk,
            )
            w_normal_series.append(w_n)
            w_event_series.append(w_e)
    
    # 组合收益：等权平均
    port_returns_series = returns_matrix.mean(axis=1)
    
    # Benchmark: 第一只资产 (沪深300)
    if available_return_cols:
        bench_returns = returns_matrix[:, 0]
    else:
        bench_returns = np.zeros(len(features_df))
    
    # Equity curve: 累积净值
    equity_curve = np.cumprod(1.0 + port_returns_series)
    equity_curve = np.insert(equity_curve, 0, 1.0)  # 初始净值 1.0
    
    # 滚动 Sharpe (20日窗口)
    window = 20
    rs_mean = pd.Series(port_returns_series).rolling(window).mean()
    rs_std = pd.Series(port_returns_series).rolling(window).std()
    sharpe_series = rs_mean / (rs_std + 1e-9) * np.sqrt(252)
    sharpe_series = sharpe_series.fillna(0.0)

    # 滚动最大回撤
    def rolling_max_drawdown(equity: np.ndarray) -> float:
        """计算单期最大回撤"""
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        return float(np.min(drawdown))
    
    # 构建 live_data 列表
    dates = features_df.index.tolist()
    for i, date in enumerate(dates):
        # LLM 评分
        if llm_scores_df is not None and date in llm_scores_df.index:
            row = llm_scores_df.loc[date]
            llm_macro = row.get("d1", 50.0) if "d1" in row else 50.0
            llm_sentiment = row.get("d2", 50.0) if "d2" in row else 50.0
            llm_risk = row.get("d3", 50.0) if "d3" in row else 50.0
        else:
            llm_macro = 50.0
            llm_sentiment = 50.0
            llm_risk = 50.0

        # 市场波动率
        vol = float(vol_series.iloc[i]) if i < len(vol_series) else 0.15
        vol = max(vol, 0.01)

        # 当前组合收益率
        r_port = float(port_returns_series[i]) if i < len(port_returns_series) else 0.0

        # 当前 Sharpe（从20日滚动窗口）
        sharpe = float(sharpe_series.iloc[i]) if i < len(sharpe_series) else 0.0

        # 当前 MDD（从 equity curve 计算到当前时点）
        current_equity = equity_curve[:i+2]  # include initial 1.0
        mdd = rolling_max_drawdown(current_equity) if len(current_equity) > 1 else 0.0

        # 历史收益率窗口（用于 reward 计算的 TE）
        hist_ret_window = port_returns_series[max(0, i-19):i+1] if i > 0 else np.array([0.0])
        hist_bench_window = bench_returns[max(0, i-19):i+1] if i > 0 else np.array([0.0])
        
        # 累积 equity curve 到当前
        equity_to_date = equity_curve[:i+2].tolist()
        
        # 2×5 收益窗口：[t-1, t]，用于RegretEngine计算
        n_assets = returns_matrix.shape[1]
        if i >= 1 and returns_matrix.shape[1] == n_assets:
            ret_prev = returns_matrix[i - 1].tolist()   # t-1
            ret_curr = returns_matrix[i].tolist()      # t
        else:
            ret_prev = [0.0] * n_assets
            ret_curr = [0.0] * n_assets
        returns_window_5d = [ret_prev, ret_curr]
        
        live_data = {
            "ae_error": float(ae_errors[i]) if i < len(ae_errors) else 0.0,
            "vol_mkt_20d": vol,
            "llm_macro": llm_macro,
            "llm_sentiment": llm_sentiment,
            "llm_risk": llm_risk,
            "port_sharpe_20d": sharpe,
            "port_mdd_current": abs(mdd),  # MDD 是负值，取绝对值
            "r_port": r_port,
            "w_normal_t": w_normal_series[i],
            "w_event_t": w_event_series[i],
            "port_returns": hist_ret_window,
            "benchmark_returns": hist_bench_window,
            "equity_curve": equity_to_date,
            "returns_window_5d": returns_window_5d,  # [t-1, t]，shape (2, 5)
        }
        live_data_list.append(live_data)
    
    return live_data_list


def compute_reconstruction_error_batch(
    model,
    X_batch: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """批量计算重构误差"""
    import torch
    model.eval()
    X_tensor = torch.from_numpy(X_batch).float().to(device)
    with torch.no_grad():
        reconstructed = model(X_tensor)
        errors = torch.sum((X_tensor - reconstructed) ** 2, dim=1).numpy()
    return errors


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
        or (PROJECT_ROOT / paths_cfg.get("checkpoints", "checkpoints"))
    )
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "actor_critic.pth"
    # Ensure parent dir exists
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_path.resolve()

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
        tb_writer: SummaryWriter = SummaryWriter(log_dir=str(tb_log_dir))  # type: ignore[assignment]
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
        features_df = pd.read_parquet(features_path)
        log.info(f"特征矩阵: {features_df.shape}")
        log.info("历史数据就绪，可注入环境")
    else:
        log.warning("历史特征或scaler不存在，环境将使用模拟数据")
        features_df = None
        scaler_state = None
    
    # ── 加载 AE 模型（用于计算 ae_error）───────────────────────────────────
    ae_model = None
    if features_df is not None:
        ae_path = PROJECT_ROOT / paths_cfg.get("checkpoints", "checkpoints") / "ae_weights.pth"
        if ae_path.exists():
            from src.models.regime_autoencoder import RegimeAutoEncoder
            model_cfg = config.get("model", {}).get("regime_autoencoder", {})
            ae_model = RegimeAutoEncoder(
                input_dim=features_df.shape[1],
                latent_dim=model_cfg.get("latent_dim", 6),
                hidden_dim=model_cfg.get("hidden_dim", 16),
            ).to(device)
            ckpt = torch.load(ae_path, map_location=device, weights_only=True)
            ae_model.load_state_dict(ckpt.get("model_state", ckpt))
            ae_model.eval()
            log.info(f"AE 模型加载: {ae_path}")
        else:
            log.warning("AE 权重不存在，跳过 ae_error 计算")
    
    # ── 加载 LLM 评分历史 ─────────────────────────────────────────────────
    llm_scores_df = None
    llm_db_path = PROJECT_ROOT / "data" / "llm_cache" / "llm_scores.db"
    if llm_db_path.exists():
        import sqlite3
        conn = sqlite3.connect(llm_db_path)
        llm_scores_df = pd.read_sql(
            "SELECT week_end, concept, d1, d2, d3 FROM llm_scores WHERE error IS NULL",
            conn
        )
        conn.close()
        log.info(f"LLM 评分历史: {len(llm_scores_df)} 条")
    else:
        log.warning("LLM scores 数据库不存在，使用默认评分")

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
    rollout_buffer = RolloutBuffer(
        buffer_size=buffer_size,
        state_dim=ppo_cfg.get("state_dim", 10),
        action_dim=ppo_cfg.get("action_dim", 2),
    )
    trainer = PPOTrainer(
        actor_critic=ac,
        config=config,
        device=device,
        buffer=rollout_buffer,
    )

    # ── Resume ─────────────────────────────────────────────────────────────
    step_offset = 0
    if args.resume and checkpoint_path.exists():
        log.info(f"从 checkpoint 恢复: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        ac.load_state_dict(ckpt["ac"])
        trainer.optimizer.load_state_dict(ckpt["optimizer"])
        step_offset = ckpt.get("step_count", 0)
        # 清理旧buffer数据，避免resume时残留数据污染
        trainer.buffer.clear()
        log.info(f"恢复步数: {step_offset}，buffer已清空")

    # ── 训练主循环 ─────────────────────────────────────────────────────────
    total_timesteps = args.total_timesteps
    buffer_size_i = trainer.buffer_size
    n_updates = total_timesteps // buffer_size_i

    log.info(f"══ PPO Training Start: {n_updates} 次PPO更新 ═══════════════════")
    log.info(f"    total_steps={total_timesteps}, buffer={buffer_size_i}, k_epochs={ppo_cfg.get('k_epochs',4)}")

    t_start = time.time()

    # ── 准备历史数据用于注入 ───────────────────────────────────────────────
    data_idx = 0  # 即使features_df为None也初始化，避免后续引用错误
    if features_df is not None:
        live_data_list = inject_live_data_from_history(
            env, features_df, llm_scores_df, ae_model, scaler_state, device, config
        )
        log.info(f"已准备 {len(live_data_list)} 步历史数据用于注入")
    else:
        live_data_list = None
        log.warning("无历史数据，训练将失败（step() 要求注入数据）")

    try:
        for ppo_iter in range(1, n_updates + 1):
            # ── 收集 Rollout（使用真实数据注入）───────────────────────────────
            data_idx = trainer.collect_rollout_manual(
                env=env,
                live_data_list=live_data_list,
                data_start_idx=data_idx,
            )

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
    finally:
        # ── 资源清理 ────────────────────────────────────────────────────────
        if tb_writer is not None:
            tb_writer.close()
        log.info("TB Writer 已关闭")

    # ── 最终保存 ───────────────────────────────────────────────────────────
    final_ckpt = {
        "ac": trainer.ac.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "step_count": step_offset + n_updates * buffer_size_i,
    }
    torch.save(final_ckpt, checkpoint_path)

    total_elapsed = time.time() - t_start
    log.info(f"══ PPO Training 完成！总耗时={total_elapsed:.1f}s ═")
    log.info(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
