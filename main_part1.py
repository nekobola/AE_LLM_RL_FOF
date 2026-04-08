#!/usr/bin/env python3
"""
main_part1.py — AE_LLM_RL_FOF 恐慌指数量产线演示

演示从原始数据到 Final_State 的完整流水线。

数据流：
  Track-A/B数据 → 25维特征张量X_t → AE.forward → E_t_raw
  → EMA滤波 → E_t_smoothed
  → RobustZScore → E_t_zscore
  → StateClipper → E_t_clipped
  → BurnInHandler → Final_State ∈ [-5.0, 5.0]

用法：
    python main_part1.py --date 2023-01-06
    python main_part1.py --mode full    # 完整回测演示
    python main_part1.py --mode weekly   # 单周推断演示
"""
from __future__ import annotations

import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main_part1")

# ============================================================
# 1. 加载配置
# ============================================================
def load_config() -> dict:
    """从 config.yaml 加载配置。"""
    config_path = Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found at {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


# ============================================================
# 2. 数据获取
# ============================================================
def fetch_weekly_data(config: dict, date: str, mode: str = "track_a"):
    """
    获取指定周的5资产+宏观特征数据，构造成25维张量 X_t。

    Parameters
    ----------
    config : dict
        配置字典
    date : str
        当前周五日期，格式 YYYY-MM-DD
    mode : str
        "track_a" (AkShare, 2015-) 或 "track_b" (ClickHouse, 2022-)

    Returns
    -------
    np.ndarray
        形状 (25,) 的归一化特征向量 X_t
    """
    logger.info(f"[数据获取] 模式={mode}, 日期={date}")

    if mode == "track_a":
        from src.data_pipeline.track_a import fetch_track_a
        df = fetch_track_a(start_date="2015-01-01", end_date=date)
    else:
        from src.data_pipeline.track_b import fetch_track_b
        df = fetch_track_b(start_date="2022-01-01", end_date=date)

    if df.empty:
        raise ValueError(f"无可用数据 for date={date}")

    # 计算资产特征 (5资产×4=20维)
    from src.features.asset_features import compute_asset_features
    asset_feats = compute_asset_features(df)  # shape: (T, 20)

    # 计算宏观特征 (5维)
    from src.features.macro_features import compute_macro_features
    macro_feats = compute_macro_features(df)  # shape: (T, 5)

    # 拼接为25维张量
    X = np.concatenate([asset_feats, macro_feats], axis=1)  # shape: (T, 25)

    # 取当周五的截面
    X_t = X[-1]  # shape: (25,)
    logger.info(f"[数据获取] X_t shape={X_t.shape}, 前3维={X_t[:3]}")
    return X_t


# ============================================================
# 3. 滚动归一化
# ============================================================
def normalize_X(X: np.ndarray, config: dict) -> np.ndarray:
    """
    对特征张量 X 执行滚动 Z-score 标准化（防穿越）。

    注意：实际生产中应该传入完整历史 X，然后取最后一行。
    此处演示用，仅展示接口。
    """
    from src.features.normalizer import RollingNormalizer

    normalizer = RollingNormalizer(
        window=config["features"]["normalization"]["zscore_window"],
        min_periods=config["features"]["normalization"]["min_periods"],
    )
    X_normalized = normalizer.fit_transform(X)
    return X_normalized


# ============================================================
# 4. AE 重构误差
# ============================================================
def compute_E_raw(X_t: np.ndarray, model: torch.nn.Module, device: str = "cpu") -> float:
    """计算单样本重构误差 E_t_raw = ||X_t - Decoder(Encoder(X_t))||_2^2"""
    from src.features.reconstruction_error import compute_reconstruction_error

    model.eval()
    E_raw = compute_reconstruction_error(model, X_t, device=device)
    logger.info(f"[AE推断] E_t_raw = {E_raw:.6f}")
    return E_raw


# ============================================================
# 5. 恐慌指数流水线
# ============================================================
def build_panic_pipeline(config: dict):
    """构建恐慌指数流水线（EMA + Z-score + Clip + BurnIn）"""
    from src.inference.panic_index_output import PanicIndexOutput

    pipeline = PanicIndexOutput(config)
    logger.info(f"[流水线] 构建完成: {pipeline}")
    return pipeline


# ============================================================
# 6. 单周推断演示
# ============================================================
def run_weekly_inference(date: str, model_path: str | None = None):
    """
    单周推断演示：

    1. 获取当周25维特征 X_t（来自Track-B / ClickHouse）
    2. 加载AE模型权重
    3. 计算 E_t_raw
    4. 送入恐慌指数流水线 → Final_State
    """
    config = load_config()

    # --- 数据获取（Track-B: 2022年后用ClickHouse） ---
    if date >= "2022-01-01":
        X_t = fetch_weekly_data(config, date, mode="track_b")
    else:
        X_t = fetch_weekly_data(config, date, mode="track_a")

    # --- 归一化 ---
    # 演示：手动用单位向量（实际生产用完整历史窗口）
    X_normalized = X_t  # TODO: 接入RollingNormalizer

    # --- AE推断 ---
    from src.models.regime_autoencoder import RegimeAutoEncoder

    input_dim = config["model"]["regime_autoencoder"]["input_dim"]
    latent_dim = config["model"]["regime_autoencoder"]["latent_dim"]
    ae = RegimeAutoEncoder(input_dim=input_dim, latent_dim=latent_dim)

    if model_path and Path(model_path).exists():
        ae.load_state_dict(torch.load(model_path, weights_only=True))
        logger.info(f"[AE] 权重已加载: {model_path}")
    else:
        logger.warning("[AE] 未提供模型路径，使用随机权重（演示用）")

    E_raw = compute_E_raw(X_normalized, ae)

    # --- 恐慌指数流水线 ---
    pipeline = build_panic_pipeline(config)
    final_state = pipeline.step(E_raw)

    logger.info(f"[输出] 日期={date}, Final_State={final_state:.4f}")
    logger.info(f"[Burn-in状态] 是否盲区={pipeline.is_in_burn_in}, 剩余周数={pipeline.remaining_burn_in_weeks}")

    return final_state


# ============================================================
# 7. 完整回测演示（模拟2022-01至今的历史）
# ============================================================
def run_full_backtest(start_date: str = "2022-01-07", end_date: str = "2024-12-31"):
    """
    模拟回测：每周五输出一组 Final_State。

    注意：实际回测需用 Track-A 数据（2015起）预训练AE，
    然后在 Track-B 数据（2022起）上进行测试。
    此处演示完整的 pipeline 调用逻辑。
    """
    config = load_config()
    pipeline = build_panic_pipeline(config)

    logger.info(f"[回测] 模拟日期范围: {start_date} → {end_date}")
    logger.info(f"[回测] 盲区周数: {pipeline.remaining_burn_in_weeks}")

    # 模拟每周五输出（实际生产中由 WFOScheduler 触发）
    dates = pd.bdate_range(start=start_date, end=end_date, freq="W-FRI")
    results = []

    for d in dates:
        date_str = d.strftime("%Y-%m-%d")

        # 模拟 E_raw（实际从 Track-B + AE 计算）
        E_raw = np.random.exponential(scale=1.0)
        final_state = pipeline.step(E_raw)

        in_burn = pipeline.is_in_burn_in
        results.append({
            "date": date_str,
            "E_raw": E_raw,
            "Final_State": final_state,
            "burn_in": in_burn,
        })

        if len(results) % 52 == 0:
            logger.info(f"[回测] {date_str}: Final_State={final_state:.4f} (burn_in={in_burn})")

    df = pd.DataFrame(results)

    # 统计
    burn_in_count = df["burn_in"].sum()
    active_count = len(df) - burn_in_count
    logger.info(f"[回测完成] 总周数={len(df)}, 盲区周={burn_in_count}, 有效周={active_count}")
    logger.info(f"[统计] Final_State mean={df.loc[~df['burn_in'], 'Final_State'].mean():.4f}, "
                 f"std={df.loc[~df['burn_in'], 'Final_State'].std():.4f}")

    return df


# ============================================================
# 8. 主入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AE_LLM_RL_FOF 恐慌指数量产线演示")
    parser.add_argument("--date", type=str, default="2023-01-06",
                        help="周五日期 YYYY-MM-DD（单周推断模式）")
    parser.add_argument("--mode", type=str, choices=["weekly", "full"], default="weekly",
                        help="运行模式: weekly=单周推断, full=回测演示")
    parser.add_argument("--start", type=str, default="2022-01-07",
                        help="回测起始日期（full模式）")
    parser.add_argument("--end", type=str, default="2024-12-31",
                        help="回测结束日期（full模式）")
    parser.add_argument("--model", type=str, default=None,
                        help="AE模型权重路径 (.pth)")
    args = parser.parse_args()

    if args.mode == "full":
        run_full_backtest(start_date=args.start, end_date=args.end)
    else:
        run_weekly_inference(date=args.date, model_path=args.model)
