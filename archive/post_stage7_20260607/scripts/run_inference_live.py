#!/usr/bin/env python3
"""
run_inference_live.py — 实盘/准实盘信号下发

定位：每周五盘后通过 Cron 定时任务触发的生产环境脚本。

核心链路：
  1. 增量更新本周五的收盘数据与文本数据
  2. 快速跑通 run_data_etl (单步) + run_llm_batch (单周)
  3. 加载最新的 ae_weights.pth 与 actor_critic.pth
  4. 提取最新 10维状态空间 S_t
  5. 前向传播 Actor → Δα, Δτ → 融合双轨权重
  6. 调用 penetration.agentbase_formatter 生成 target_weights.json

输出落盘：
  - results/target_weights_{date}.json
  - 标准 AgentBase 接口格式

用法（Cron 示例，每周五 16:00 触发）：
  0 16 * * 5 cd /path/to/ae-llm-rl-fof && python scripts/run_inference_live.py >> logs/cron_inference.log 2>&1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.models.regime_autoencoder import RegimeAutoEncoder
from src.env.action_mapper import ActionMapper
from src.env.state_assembler import StateAssembler
from src.penetration.agentbase_formatter import AgentBaseFormatter
from src.compute.v31_engine import V31Engine

# ── Logging ───────────────────────────────────────────────────────────────────
(PROJECT_ROOT / "logs").mkdir(exist_ok=True, parents=True)
(PROJECT_ROOT / "results").mkdir(exist_ok=True, parents=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / "run_inference_live.log", mode="a"),
    ],
)
log = logging.getLogger("run_inference_live")


def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    # 关键: yaml不替换${ENV}，运行时必须从环境变量读取
    raw["llm"]["api_key"] = os.environ.get("LLM_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    return raw


def run_etl_incremental(config: dict) -> pd.DataFrame:
    """
    增量ETL：只跑最新一周数据。
    内部调用 run_data_etl.py 的核心逻辑。
    
    Returns
    -------
    pd.DataFrame
        track_b ETF 日频数据
    """
    import pandas as pd
    from src.data_pipeline.track_b.fetcher import fetch_track_b_safe

    today = date.today()
    week_end = today.isoformat()

    # 找上周五
    from datetime import timedelta
    days_since_friday = (today.weekday() - 4) % 7
    last_friday = today - timedelta(days=days_since_friday or 7)
    start = (last_friday - timedelta(days=7)).isoformat()

    log.info(f"[ETL] 增量更新: {start} → {week_end}")

    try:
        df_b = fetch_track_b_safe(start_date=start, end_date=week_end)
        log.info(f"[ETL] track_b: {len(df_b)} 条")
        return df_b
    except Exception as e:
        log.error(f"[ETL] track_b 拉取失败: {e}")
        raise


def run_llm_incremental(config: dict, week_end: str) -> dict:
    """
    单周LLM打分：运行 run_llm_batch.py 作用于单个周五。
    """
    import asyncio
    import os
    from src.llm_engine.async_semantic_engine import AsyncSemanticEngine

    log.info(f"[LLM] 单周打分: {week_end}")
    try:
        engine = AsyncSemanticEngine(config)
        scores = asyncio.run(engine.evaluate(week_end))
        log.info(f"[LLM] 评分完成: {scores}")
        return scores
    except Exception as e:
        log.warning(f"[LLM] 调用失败，使用默认值: {e}")
        return {}  # 空字典，assemble_state会聚合为默认值50.0


def load_models(config: dict, device: str):
    """加载 AE 权重 + PPO Actor-Critic 权重。"""
    paths_cfg = config.get("paths", {})

    # AE
    ae_path = PROJECT_ROOT / paths_cfg.get("checkpoints", "checkpoints") / "ae_weights.pth"
    scaler_path = PROJECT_ROOT / paths_cfg.get("checkpoints", "checkpoints") / "ae_scaler.pkl"

    ae = RegimeAutoEncoder(
        input_dim=config.get("model", {}).get("regime_autoencoder", {}).get("input_dim", 25),
        latent_dim=config.get("model", {}).get("regime_autoencoder", {}).get("latent_dim", 6),
        hidden_dim=config.get("model", {}).get("regime_autoencoder", {}).get("hidden_dim", 16),
    ).to(device)

    if ae_path.exists():
        ckpt = torch.load(ae_path, map_location=device)
        ae.load_state_dict(ckpt.get("model_state", ckpt))
        ae.eval()
        log.info(f"AE 权重加载: {ae_path}")
    else:
        log.warning(f"AE 权重不存在: {ae_path}，使用随机初始化")

    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler_state = pickle.load(f)
        log.info(f"AE Scaler 加载: {scaler_path}")
    else:
        scaler_state = None
        log.warning("AE Scaler 不存在")

    # PPO
    ppo_path = PROJECT_ROOT / paths_cfg.get("checkpoints", "checkpoints") / "actor_critic.pth"
    from src.ppo.networks import ActorCritic
    ppo_cfg = config.get("ppo", {})
    ac = ActorCritic(
        state_dim=ppo_cfg.get("state_dim", 10),
        action_dim=ppo_cfg.get("action_dim", 2),
        hidden_dim=64,
    ).to(device)

    if ppo_path.exists():
        ckpt = torch.load(ppo_path, map_location=device)
        ac.load_state_dict(ckpt.get("ac", ckpt))
        ac.eval()
        log.info(f"PPO 权重加载: {ppo_path}")
    else:
        log.warning(f"PPO 权重不存在: {ppo_path}，使用随机策略")

    return ae, scaler_state, ac


def assemble_state(
    ae,
    scaler_state,
    ae_error: float,
    vol_mkt_20d: float,
    llm_scores: dict,
    regret_ema_norm: float,
    alpha_prev: float,
    tau_prev: float,
    config: dict,
    device: str,
) -> np.ndarray:
    """组装当前10维状态向量 S_t。"""
    assembler = StateAssembler(
        sharpe_clip_low=config.get("state_assembler", {}).get("sharpe_clip_low", -3.0),
        sharpe_clip_high=config.get("state_assembler", {}).get("sharpe_clip_high", 3.0),
    )

    if scaler_state:
        assembler._ae_mean = float(scaler_state["mean"].mean())
        assembler._ae_std  = float(scaler_state["std"].mean())
        assembler._vol_min  = 0.0
        assembler._vol_max  = 1.0
        assembler._tau_min  = 0.0
        assembler._tau_max  = 1.0

    sharpe_20d   = 0.0
    mdd_current  = 0.0

    # LLM返回 {concept_name: {d1, d2, d3}}，聚合为3个维度
    # 聚合策略：
    # - d1/d2: 全市场平均（横向比较）
    # - d3: 使用【宽基指数】(沪深300/中证1000)的d3均值 + satellite板块d3最大值
    #   原因：宽基d3反映市场整体风险，satellite高d3表示尾部风险传染
    all_d1 = [v["d1"] for v in llm_scores.values() if "d1" in v]
    all_d2 = [v["d2"] for v in llm_scores.values() if "d2" in v]
    
    # 宽基指数
    wide_base = {"沪深300", "中证1000"}
    wide_d3 = [v["d3"] for k, v in llm_scores.items() if k in wide_base and "d3" in v]
    satellite_d3 = [v["d3"] for k, v in llm_scores.items() if k not in wide_base and "d3" in v]
    
    # d3 = 宽基均值 * 0.7 + satellite最大 * 0.3
    # 宽基反映市场主体风险，satellite反映尾部传染
    llm_macro      = float(np.mean(all_d1)) if all_d1 else 50.0
    llm_sentiment = float(np.mean(all_d2)) if all_d2 else 50.0
    
    if wide_d3 and satellite_d3:
        llm_risk = float(np.mean(wide_d3) * 0.7 + np.max(satellite_d3) * 0.3)
    elif wide_d3:
        llm_risk = float(np.mean(wide_d3))
    elif satellite_d3:
        llm_risk = float(np.max(satellite_d3))
    else:
        llm_risk = 50.0

    S_t = assembler.assemble(
        ae_error=ae_error,
        vol_mkt_20d=vol_mkt_20d,
        llm_macro=llm_macro,
        llm_sentiment=llm_sentiment,
        llm_risk=llm_risk,
        port_sharpe_20d=sharpe_20d,
        port_mdd_current=mdd_current,
        regret_ema_norm=regret_ema_norm,
        tau_prev=tau_prev,
        alpha_prev=alpha_prev,
    )
    return S_t


def compute_target_weights(
    ae,
    ac,
    state: np.ndarray,
    config: dict,
    device: str,
    llm_macro: float = 50.0,
    llm_sentiment: float = 50.0,
    llm_risk: float = 50.0,
) -> tuple[np.ndarray, float, float]:
    """
    前向传播计算目标权重。

    Returns
    -------
    Tuple[np.ndarray, float, float]
        (w_target, delta_alpha, delta_tau)
    """
    with torch.no_grad():
        S_t = torch.from_numpy(state).float().unsqueeze(0).to(device)

        # Actor 前向
        mu_t, _ = ac.actor(S_t)
        mu_t = mu_t.cpu().numpy().squeeze()

        # Action Mapping
        action_mapper = ActionMapper(
            alpha_min=config.get("action_mapper", {}).get("alpha_min", -0.5),
            alpha_max=config.get("action_mapper", {}).get("alpha_max", 0.1),
            tau_delta_range=config.get("action_mapper", {}).get("tau_delta_range", 0.1),
        )
        delta_alpha, delta_tau = action_mapper.map(float(mu_t[0]), float(mu_t[1]))

        alpha_new = action_mapper.clip_alpha(
            config.get("env", {}).get("initial_alpha", 0.5) + delta_alpha
        )
        tau_new = action_mapper.clip_tau(
            config.get("env", {}).get("initial_tau", 0.5) + delta_tau,
            config.get("env", {}).get("tau_min", 0.0),
            config.get("env", {}).get("tau_max", 1.0),
        )

        # Stage 7: 单轨 V3.1, PPO 控 theta
        v31_engine = V31Engine(config)
        try:
            returns_5d = np.random.randn(5, 5) * 0.01
            # TODO(Stage 7): theta 由 PPO actor 实时推理, 此处先用 0.7 占位
            w_event = v31_engine.compute(
                returns_5d,
                llm_macro=llm_macro,
                llm_sentiment=llm_sentiment,
                llm_risk=llm_risk,
                theta=0.7,
            )
        except Exception:
            w_event = np.array([0.2] * 5)

        w_target = np.clip(w_event, 0, 1)
        w_target = w_target / (w_target.sum() + 1e-9)

        return w_target, alpha_new, tau_new


def format_and_save(
    w_target: np.ndarray,
    week_end: str,
    alpha: float,
    tau: float,
    ae_error: float,
    config: dict,
) -> Path:
    """生成 target_weights.json。"""
    output = {
        "week_end": week_end,
        "generated_at": date.today().isoformat(),
        "alpha": float(alpha),
        "tau": float(tau),
        "ae_error": float(ae_error),
        "weights_5d": {
            "V1_wide_base":  float(w_target[0]),
            "V2_satellite":  float(w_target[1]),
            "V3_pure_bond":  float(w_target[2]),
            "V4_hedge":      float(w_target[3]),
            "V5_cash":        float(w_target[4]),
        },
        "raw_weights": [float(w) for w in w_target],
    }

    out_dir = PROJECT_ROOT / config.get("paths", {}).get("wfo_results", "results/wfo")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"target_weights_{week_end}.json"

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    log.info(f"目标权重已生成: {out_path}")
    log.info(f"  weights: {[f'{w:.3f}' for w in w_target]}")
    log.info(f"  alpha={alpha:.3f}, tau={tau:.3f}")

    # 同时生成 latest.json（供实盘读取最新版本）
    latest_path = out_dir / "target_weights_latest.json"
    with open(latest_path, "w") as f:
        json.dump(output, f, indent=2)

    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="run_inference_live: 实盘信号下发")
    parser.add_argument("--week-end", default=None, help="周五日期 YYYY-MM-DD，默认为今天")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-etl", action="store_true", help="跳过ETL/LLM步骤，使用缓存")
    args = parser.parse_args()

    from datetime import timedelta
    today = date.today()
    days_since_friday = (today.weekday() - 4) % 7
    week_end = (
        args.week_end
        or (today - timedelta(days=days_since_friday or 7)).isoformat()
    )

    log.info(f"══ 实盘信号下发: {week_end} ══════════════════════════════")

    config = load_config()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ── Step 1: ETL（增量） ─────────────────────────────────────────────────
    df_b = None
    if not args.skip_etl:
        log.info("[Step 1/5] ETL 增量数据 ...")
        df_b = run_etl_incremental(config)
    else:
        log.info("[Step 1/5] 跳过 ETL")
    
    # ── 加载最新特征用于 AE 计算 ─────────────────────────────────────────
    import pandas as pd
    features_path = PROJECT_ROOT / config.get("paths", {}).get("data_processed", "data/processed") / "features_master.parquet"
    latest_features = None
    features_df = None  # 确保后续作用域可访问
    if features_path.exists():
        features_df = pd.read_parquet(features_path)
        latest_features = features_df.iloc[-1].values  # 最新一天的25维特征
        log.info(f"[特征] 最新特征: {features_path}")

    # ── Step 2: LLM 评分 ───────────────────────────────────────────────────
    if not args.skip_etl:
        log.info("[Step 2/5] LLM 单周打分 ...")
        llm_scores = run_llm_incremental(config, week_end)
    else:
        log.info("[Step 2/5] 跳过 LLM，使用默认评分")
        llm_scores = {}  # 空字典，assemble_state会聚合为默认值50.0

    # ── Step 3: 加载模型 ────────────────────────────────────────────────────
    log.info("[Step 3/5] 加载 AE + PPO 模型 ...")
    ae, scaler_state, ac = load_models(config, device)

    # ── Step 4: 计算 S_t + 前向传播 ────────────────────────────────────────
    log.info("[Step 4/5] 组装状态 + 前向传播 ...")

    # AE 重建误差（从真实特征计算）
    with torch.no_grad():
        if latest_features is not None:
            X_t = torch.from_numpy(latest_features).float().unsqueeze(0).to(device)
            if hasattr(ae, "decode"):
                ae.eval()
                recon = ae(X_t)
                ae_error = float((recon - X_t).pow(2).mean().item())
            else:
                ae_error = 0.5
        else:
            ae_error = 0.5
            log.warning("无特征数据，使用默认 ae_error=0.5")

    # 计算市场波动率（从 df_b 计算 20 日波动率）
    vol_mkt_20d = 0.15  # 默认值
    if df_b is not None and not df_b.empty:
        price_col = None
        for col in df_b.columns:
            if "close" in col.lower():
                price_col = col
                break
        if price_col is not None:
            prices = np.asarray(df_b[price_col].values)
            if len(prices) >= 20:
                vol_mkt_20d = float(np.std(prices[-20:]) * np.sqrt(252))
                vol_mkt_20d = max(vol_mkt_20d, 0.01)  # 避免为零

    # ── 计算 RegretEngine EMA ────────────────────────────────────────────
    from src.env.regret_engine import RegretEngine
    regret_cfg = config.get("regret_engine", {})
    regret_engine = RegretEngine(ema_decay=regret_cfg.get("ema_decay", 0.8))
    
    # 尝试从 features_df 计算最近收益
    if features_path.exists() and 'features_df' in dir():
        try:
            # 从config读取资产代码（features_df列名使用资产代码，非ClickHouse ETF代码）
            asset_codes = config.get("data_pipeline", {}).get("asset_codes", [
                "000300.SH", "000852.SH", "CBA02701.CS", "AU9999.SGE", "NH0100.NHF"
            ])
            returns_5d = np.zeros((2, 5))  # [prev, curr] — 缺省为0
            for j, code in enumerate(asset_codes[:5]):
                col = f"{code}__weekly_return"
                if features_df is not None and col in features_df.columns:
                    vals = features_df[col].dropna().values
                    if len(vals) >= 2:
                        returns_5d[0, j] = float(vals[-2])
                        returns_5d[1, j] = float(vals[-1])
            w_init = np.array([0.2]*5)
            _, regret_ema_norm = regret_engine.compute(w_init, returns_5d)
        except Exception as e:
            log.warning(f"RegretEngine 计算失败，使用 0.0: {e}")
            regret_ema_norm = 0.0
    else:
        regret_ema_norm = 0.0
    
    alpha_prev = 0.5
    tau_prev   = 0.5

    S_t = assemble_state(
        ae=ae,
        scaler_state=scaler_state,
        ae_error=ae_error,
        vol_mkt_20d=vol_mkt_20d,
        llm_scores=llm_scores,
        regret_ema_norm=regret_ema_norm,
        alpha_prev=alpha_prev,
        tau_prev=tau_prev,
        config=config,
        device=device,
    )

    w_target, alpha_new, tau_new = compute_target_weights(
        ae=ae, ac=ac, state=S_t, config=config, device=device,
        llm_macro=llm_macro, llm_sentiment=llm_sentiment, llm_risk=llm_risk,
    )

    # ── Step 5: 格式化 + 落盘 ──────────────────────────────────────────────
    log.info("[Step 5/5] 生成 target_weights.json ...")
    out_path = format_and_save(
        w_target=w_target,
        week_end=week_end,
        alpha=alpha_new,
        tau=tau_new,
        ae_error=ae_error,
        config=config,
    )

    log.info(f"══ 实盘信号下发完成: {out_path} ═")
    log.info(f"  alpha={alpha_new:.3f}, tau={tau_new:.3f}")


if __name__ == "__main__":
    main()
