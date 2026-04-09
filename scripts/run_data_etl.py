#!/usr/bin/env python3
"""
run_data_etl.py — 离线数据清洗与特征构建

系统的供血泵：为下游所有模型生成标准化的张量与面板数据。

调度链路：
  1. 触发 track_a (AkShare 公网) + track_b (quantchdb) 拉取量价数据
  2. 调用 asset_features 计算 5资产 × 4特征 = 20维资产特征
  3. 调用 macro_features 计算宏观特征 5维
  4. 对齐时间轴，拼接为 25维特征矩阵
  5. 调用 normalizer 执行严格 Z-score 标准化（窗口 [t-252, t-1]）
  6. 输出落盘：data/processed/features_master.parquet

用法：
  python scripts/run_data_etl.py [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD] [--force]
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import numpy as np

# ── Project Root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.data_pipeline.track_a.fetcher import fetch_track_a
from src.data_pipeline.track_b.fetcher import fetch_track_b_safe
from src.features.asset_features import compute_asset_features
from src.features.macro_features import fetch_macro_features, compute_macro_features
from src.features.normalizer import normalize_dataframe

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / "run_data_etl.log", mode="a"),
    ],
)
log = logging.getLogger("run_data_etl")


def load_config() -> dict:
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    # Expand environment variables in api_key
    import os
    llm_key = os.environ.get("OPENAI_API_KEY", "")
    if llm_key and "api_key" in raw.get("llm", {}):
        raw["llm"]["api_key"] = llm_key
    return raw


def run_etl(
    start_date: str,
    end_date: str | None = None,
    force_rebuild: bool = False,
    config: dict | None = None,
) -> pd.DataFrame:
    """
    执行完整 ETL 流程。

    Parameters
    ----------
    start_date : str
        数据起始日期，YYYY-MM-DD
    end_date : str | None
        数据终止日期，None=今天
    force_rebuild : bool
        True=强制全量重算，False=增量（跳过已有 parquet）
    config : dict | None
        配置字典

    Returns
    -------
    pd.DataFrame
        25维特征矩阵，index=date，columns=[asset_0_...asset_4_, macro_...],shape=(T,25)
    """
    if config is None:
        config = load_config()

    if end_date is None:
        end_date = date.today().isoformat()

    out_dir = PROJECT_ROOT / config.get("paths", {}).get("data_processed", "data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "features_master.parquet"

    # ── 增量检查 ────────────────────────────────────────────────────────────
    if out_path.exists() and not force_rebuild:
        existing = pd.read_parquet(out_path)
        last_date = existing.index.max()
        if pd.to_datetime(start_date) <= pd.to_datetime(last_date):
            log.info(
                f"找到已有特征文件，最后日期 {last_date.date()}，"
                f"从 {start_date} 到 {end_date} 执行增量更新"
            )
            start_date = (pd.to_datetime(last_date) + pd.DateOffset(days=1)).strftime("%Y-%m-%d")
            if pd.to_datetime(start_date) > pd.to_datetime(end_date):
                log.info("数据已是最新，无需更新")
                return existing

    # ════════════════════════════════════════════════════════════════════════
    log.info(f"══ ETL START: {start_date} → {end_date} ════════════════════")
    t0 = datetime.now()

    # ── Step 1: 拉取轨A数据 (AkShare) ─────────────────────────────────────
    log.info("[1/6] 拉取 track_a (AkShare 公网) ...")
    try:
        df_track_a = fetch_track_a(start_date=start_date, end_date=end_date)
        log.info(f"    track_a 拉取完成: {len(df_track_a)} 条, 列 {list(df_track_a.columns)}")
    except Exception as e:
        log.error(f"track_a 拉取失败: {e}")
        raise

    # ── Step 2: 拉取轨B数据 (quantchdb) ────────────────────────────────────
    log.info("[2/6] 拉取 track_b (quantchdb) ...")
    try:
        df_track_b = fetch_track_b_safe(start_date=start_date, end_date=end_date)
        log.info(f"    track_b 拉取完成: {len(df_track_b)} 条")
    except Exception as e:
        log.warning(f"track_b 拉取失败（将仅用track_a）: {e}")
        df_track_b = pd.DataFrame()

    # ── Step 3: 计算资产特征 (5资产 × 4特征 = 20维) ─────────────────────────
    log.info("[3/6] 计算资产特征 (momentum/volatility/mean_corr/weekly_return) ...")
    feat_cfg = config.get("features", {})

    # 轨B为主（ETF日频数据）做资产特征
    if not df_track_b.empty:
        price_df = df_track_b
    else:
        log.warning("track_b为空，使用track_a构建资产特征（仅有宽基数据）")
        price_df = df_track_a

    asset_feat = compute_asset_features(price_df, feat_cfg)
    log.info(f"    资产特征: {asset_feat.shape}, 列数: {len(asset_feat.columns)}")

    # ── Step 4: 计算宏观特征 (5维) ───────────────────────────────────────────
    log.info("[4/6] 计算宏观特征 (DR007/汇率/国债/利差/北向) ...")
    macro_feat = compute_macro_features(
        start_date=start_date,
        end_date=end_date,
        config=feat_cfg,
    )
    log.info(f"    宏观特征: {macro_feat.shape}, 列数: {len(macro_feat.columns)}")

    # ── Step 5: 对齐时间轴，拼接25维特征矩阵 ────────────────────────────────
    log.info("[5/6] 对齐时间轴，拼接25维特征矩阵 ...")

    # macro_feat index 是 date，asset_feat 也要对齐
    common_dates = asset_feat.index.intersection(macro_feat.index)
    if len(common_dates) == 0:
        log.error("资产特征与宏观特征无重叠日期！检查数据源是否有效。")
        raise ValueError("特征对齐失败：无重叠日期")

    asset_feat_aligned = asset_feat.loc[common_dates]
    macro_feat_aligned = macro_feat.loc[common_dates]

    # 合并
    features_master = pd.concat([asset_feat_aligned, macro_feat_aligned], axis=1)
    features_master = features_master.dropna()
    log.info(f"    拼接后: {features_master.shape}, 日期范围 {features_master.index[0].date()} → {features_master.index[-1].date()}")

    # ── Step 6: 严格 Z-score 标准化 (防穿越) ─────────────────────────────────
    log.info("[6/6] 执行严格Z-score标准化 (窗口252日，防穿越) ...")
    norm_cfg = feat_cfg.get("normalization", {})
    zscore_window = norm_cfg.get("zscore_window", 252)
    min_periods = norm_cfg.get("min_periods", 60)

    features_normalized = normalize_dataframe(
        features_master,
        window=zscore_window,
        min_periods=min_periods,
    )

    # 去掉 NaN 行（冷启动期）
    n_before = len(features_normalized)
    features_normalized = features_normalized.dropna()
    n_after = len(features_normalized)
    log.info(f"    标准化完成: {n_before}→{n_after}行 (丢弃冷启动NaN: {n_before - n_after})")

    # ── 落盘 ────────────────────────────────────────────────────────────────
    features_normalized.to_parquet(out_path, index=True)
    elapsed = (datetime.now() - t0).total_seconds()
    log.info(
        f"✅ ETL 完成！输出: {out_path}  "
        f"shape={features_normalized.shape}  耗时={elapsed:.1f}s"
    )

    return features_normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="run_data_etl: 离线数据清洗与特征构建")
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--force", action="store_true", help="强制全量重算")
    args = parser.parse_args()

    (PROJECT_ROOT / "logs").mkdir(exist_ok=True)

    try:
        df = run_etl(
            start_date=args.start_date,
            end_date=args.end_date,
            force_rebuild=args.force,
        )
        log.info(f"最终特征矩阵: {df.shape}")
    except Exception as e:
        log.error(f"ETL 失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
