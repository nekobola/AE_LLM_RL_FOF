#!/usr/bin/env python3
"""
train_ae.py — 自编码器预训练

定位：独立训练 Regime AutoEncoder，固化宏观压迫感(E_t)提取器。

调度链路：
  1. 加载 data/processed/features_master.parquet
  2. 调用 normalizer 执行时间序列安全的 Z-score 缩放（冷启动期drop）
  3. 实例化 models.regime_autoencoder
  4. 按 val_split=0.8/0.2 划分训练/验证集
  5. MSE 损失训练，Early Stopping
  6. 输出: checkpoints/ae_weights.pth, checkpoints/ae_scaler.pkl

用法：
  python scripts/train_ae.py [--epochs 50] [--batch-size 256] [--lr 1e-3]
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.models.regime_autoencoder import RegimeAutoEncoder
from src.features.normalizer import normalize_dataframe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / "train_ae.log", mode="a"),
    ],
)
log = logging.getLogger("train_ae")


def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def build_dataloaders(
    features: pd.DataFrame,
    batch_size: int,
    val_split: float,
    scaler_state: dict | None = None,
):
    """
    划分训练/验证集，返回 DataLoader。

    scaler_state: 若非None，则用该状态的mean/std进行标准化（实盘同款）
    """
    # 标准化已经在ETL中做过了，这里直接用
    # 但为了避免重复标准化，我们存一个scaler_state的副本
    X = features.values.astype(np.float32)

    # 去NaN
    valid_mask = ~np.isnan(X).any(axis=1)
    X = X[valid_mask]
    log.info(f"有效样本: {len(X)} (丢弃NaN: {valid_mask.sum() - len(X)})")

    n = len(X)
    n_train = int(n * (1 - val_split))
    indices = np.random.default_rng(42).permutation(n)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    X_train = torch.from_numpy(X[train_idx])
    X_val   = torch.from_numpy(X[val_idx])

    train_ds = TensorDataset(X_train, X_train)
    val_ds   = TensorDataset(X_val,   X_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def save_checkpoint(
    model: RegimeAutoEncoder,
    optimizer: optim.Optimizer,
    epoch: int,
    val_loss: float,
    scaler_state: dict,
    checkpoint_path: Path,
    scaler_path: Path,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_loss,
        },
        checkpoint_path,
    )
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler_state, f)
    log.info(f"Checkpoint保存: {checkpoint_path}, scaler: {scaler_path}")


def train_one_epoch(
    model: RegimeAutoEncoder,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    criterion = nn.MSELoss()

    for X_batch, X_target in loader:
        X_batch = X_batch.to(device)
        X_recon  = model(X_batch)
        loss = criterion(X_recon, X_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(X_batch)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: RegimeAutoEncoder, loader: DataLoader, device: str) -> float:
    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss()

    for X_batch, X_target in loader:
        X_batch = X_batch.to(device)
        X_recon  = model(X_batch)
        loss = criterion(X_recon, X_batch)
        total_loss += loss.item() * len(X_batch)

    return total_loss / len(loader.dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description="train_ae: 自编码器预训练")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--scaler-path", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_config()
    paths_cfg = config.get("paths", {})

    checkpoint_path = Path(args.checkpoint_path or paths_cfg.get(
        "checkpoints", "checkpoints"
    )) / "ae_weights.pth"
    scaler_path = Path(args.scaler_path or paths_cfg.get(
        "checkpoints", "checkpoints"
    )) / "ae_scaler.pkl"

    # ── Device ───────────────────────────────────────────────────────────
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    # ── 加载特征 ──────────────────────────────────────────────────────────
    data_path = PROJECT_ROOT / paths_cfg.get("data_processed", "data/processed") / "features_master.parquet"
    log.info(f"加载特征: {data_path}")
    if not data_path.exists():
        log.error(f"特征文件不存在！请先运行: python scripts/run_data_etl.py")
        sys.exit(1)

    features = pd.read_parquet(data_path)
    log.info(f"特征矩阵: {features.shape}, 日期范围 {features.index[0].date()} → {features.index[-1].date()}")

    # ── 构建 Scaler State ──────────────────────────────────────────────────
    scaler_state = {
        "mean": features.mean().values,
        "std":  features.std(ddof=1).values,
        "columns": list(features.columns),
    }

    # ── DataLoaders ───────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(
        features=features,
        batch_size=args.batch_size,
        val_split=args.val_split,
    )

    # ── 模型 ───────────────────────────────────────────────────────────────
    model_cfg = config.get("model", {}).get("regime_autoencoder", {})
    input_dim   = model_cfg.get("input_dim", 25)
    latent_dim  = model_cfg.get("latent_dim", 6)
    hidden_dim  = model_cfg.get("hidden_dim", 16)

    model = RegimeAutoEncoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    log.info(f"模型: input={input_dim}, latent={latent_dim}, hidden={hidden_dim}")
    log.info(f"参数: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")

    # ── 训练循环 ───────────────────────────────────────────────────────────
    log.info("══ AE Training Start ════════════════════════════════════")
    best_val_loss = float("inf")
    patience_counter = 0
    max_patience = 10

    for epoch in range(1, args.epochs + 1):
        t0 = datetime.now()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss   = evaluate(model, val_loader, device)
        elapsed    = (datetime.now() - t0).total_seconds()

        log.info(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
            f"time={elapsed:.1f}s  lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        scheduler.step(val_loss)

        # Early Stopping + Save Best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                scaler_state, checkpoint_path, scaler_path
            )
            log.info(f"  ⭐ 新最佳 val_loss={val_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                log.info(f"Early Stopping at epoch {epoch}")
                break

    log.info(f"══ 训练完成！最佳 val_loss={best_val_loss:.6f} ═")
    log.info(f"权重: {checkpoint_path}")
    log.info(f"Scaler: {scaler_path}")


if __name__ == "__main__":
    main()
