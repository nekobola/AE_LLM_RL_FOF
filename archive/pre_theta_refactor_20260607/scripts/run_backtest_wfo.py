#!/usr/bin/env python3
"""
run_backtest_wfo.py — Walk-Forward 滚动回测引擎
重构:
  1. 集成 WFOScheduler 实现季度重训 + 周频推断
  2. 构建 GeneralBacktest 格式的 weights_data 输出
  3. 保留双轨 + alpha融合逻辑（momentum policy 作为alpha决策）
"""
from __future__ import annotations

import argparse, json, logging, pickle, sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from src.compute.dual_track_engine import DualTrackEngine
from src.failsafe.veto_switch import VetoSwitch
from src.env.action_mapper import ActionMapper
from src.env.regret_engine import RegretEngine
from src.env.state_assembler import StateAssembler
from src.models.regime_autoencoder import RegimeAutoEncoder
from src.features.reconstruction_error import compute_reconstruction_error_batch

(PROJECT_ROOT / "logs").mkdir(exist_ok=True)
(PROJECT_ROOT / "results").mkdir(exist_ok=True)
(PROJECT_ROOT / "results" / "wfo").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / "run_backtest_wfo.log", mode="a"),
    ],
)
log = logging.getLogger("run_backtest_wfo")

# ── 5个资产 ───────────────────────────────────────────────────────────────────
ASSET_CODES = [
    "000300.SH",   # CSI 300 (股票)
    "000852.SH",   # CSI 1000 (股票)
    "CBA02701.CS", # Bond index (债券)
    "AU9999.SGE",  # Gold (黄金)
    "NH0100.NHF",  # NH index (商品/通胀)
]


def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_wfo_metrics(nav_series: pd.Series) -> dict:
    """Metrics from the WFO loop's synthetic NAV.

    NAV is computed weekly: nav[t] = nav[t-1] * (1 + w @ weekly_return).
    The `weekly_return` columns in features_master.parquet are z-scored
    features, NOT raw asset returns — so these metrics are useful only
    for relative comparison within the synthetic stream. For production
    decisions, use `metrics_real.json` (GeneralBacktest on real OHLC).
    """
    returns = nav_series.pct_change().dropna()
    total_return = (nav_series.iloc[-1] / nav_series.iloc[0]) - 1
    n_years = len(returns) / 52  # weekly data — 52 weeks per year
    annualized_return = (1 + total_return) ** (1 / n_years) - 1
    cummax = nav_series.cummax()
    drawdown = (nav_series - cummax) / cummax
    max_drawdown = float(drawdown.min())
    excess = returns.mean() * 52  # weekly mean × 52
    vol = returns.std() * np.sqrt(52)  # weekly std × sqrt(52)
    sharpe = excess / vol if vol > 0 else 0.0
    return {
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "max_drawdown": float(max_drawdown),
        "sharpe_ratio": float(sharpe),
        "volatility_annual": float(vol),
        "n_weeks": len(returns),
        "_source": "synthetic_nav_from_zscored_features",
    }


def is_quarter_end(friday: pd.Timestamp) -> bool:
    """判断是否为季度末周五"""
    # 季度末: 3/31, 6/30, 9/30, 12/31
    month = friday.month
    day = friday.day
    is_march_end = month == 3 and day >= 25
    is_june_end = month == 6 and day >= 25
    is_sep_end = month == 9 and day >= 25
    is_dec_end = month == 12 and day >= 25
    return is_march_end or is_june_end or is_sep_end or is_dec_end


def build_weights_data(records: list, codes: list) -> pd.DataFrame:
    """
    将 records 列表转换为 GeneralBacktest 格式的 weights_data DataFrame。

    Parameters
    ----------
    records : list of dict
        每条记录包含 date, w_fused (numpy array of 5 weights)
    codes : list of str
        5个资产代码

    Returns
    -------
    pd.DataFrame
        columns: date, code, weight
    """
    rows = []
    for rec in records:
        date_str = rec["date"]
        w = rec["w_fused"]
        for code, weight in zip(codes, w):
            rows.append({"date": date_str, "code": code, "weight": float(weight)})
    return pd.DataFrame(rows)


def aggregate_llm_weekly(llm_scores_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate concept-level LLM scores into one weekly macro/sentiment/risk row."""
    wide_base = {"沪深300", "中证1000"}
    df = llm_scores_df.copy()
    df["week_end"] = pd.to_datetime(df["week_end"])

    def smart_agg(week_end: pd.Timestamp, group: pd.DataFrame) -> dict:
        d1 = float(group["d1"].mean())
        d2 = float(group["d2"].mean())

        wide = group[group["concept"].isin(wide_base)]
        satellite = group[~group["concept"].isin(wide_base)]

        wide_d3_mean = float(wide["d3"].mean()) if len(wide) > 0 else 50.0
        satellite_d3_max = float(satellite["d3"].max()) if len(satellite) > 0 else 50.0
        return {
            "week_end": week_end,
            "d1": d1,
            "d2": d2,
            "d3": wide_d3_mean * 0.7 + satellite_d3_max * 0.3,
        }

    rows = [
        smart_agg(week_end, group)
        for week_end, group in df.groupby("week_end", sort=True)
    ]
    return pd.DataFrame(rows)


def run_wfo(
    start_date: str,
    end_date: str | None,
    lookback_weeks: int,
    output_dir: Path,
    config: dict,
    ae_scaler_path: Path | None = None,
    ae_weights_path: Path | None = None,
    ppo_checkpoint: Path | None = None,
    use_wfo_scheduler: bool = True,
    force_alpha: float | None = None,
) -> pd.DataFrame:
    """
    Walk-Forward 滚动回测。

    Parameters
    ----------
    use_wfo_scheduler : bool
        是否启用 WFOScheduler（冷启动 + 季度重训）
    """
    if end_date is None:
        end_date = date.today().isoformat()

    log.info(f"══ WFO 回测: start={start_date}, end={end_date}, lookback={lookback_weeks}周 ═")

    # ── 子系统初始化 ──────────────────────────────────────────────────────
    # V3 是生产开关, V3.1 是实验性 toggle. config.event_track_version 控制.
    # 默认 "v3" (生产). 设 "v3_1" 切到 V3.1 实验版.
    et_version = config.get("event_track_version", "v3")
    dual_engine = DualTrackEngine(
        config,
        use_v3=(et_version == "v3"),
        use_v3_1=(et_version == "v3_1"),
    )
    log.info(f"[EventTrack] 选用版本: {et_version} -> {type(dual_engine.event_track).__name__}")
    veto_switch = VetoSwitch(config)
    action_mapper = ActionMapper(
        alpha_min=config.get("action_mapper", {}).get("alpha_min", -0.5),
        alpha_max=config.get("action_mapper", {}).get("alpha_max", 0.1),
        tau_delta_range=config.get("action_mapper", {}).get("tau_delta_range", 0.1),
        alpha_bias=config.get("action_mapper", {}).get("alpha_bias", -0.05),
    )
    regret_engine = RegretEngine(
        ema_decay=config.get("regret_engine", {}).get("ema_decay", 0.8)
    )

    # ── WFOScheduler 初始化 ───────────────────────────────────────────────
    scheduler = None
    if use_wfo_scheduler:
        try:
            from src.schedules.wfo_scheduler import WFOScheduler
            device = "cuda" if torch.cuda.is_available() else "cpu"
            scheduler = WFOScheduler(config, device)
            log.info("[WFOScheduler] 初始化成功")
        except ImportError as e:
            log.warning(f"[WFOScheduler] 导入失败: {e}，降级为无调度模式")
        except Exception as e:
            log.warning(f"[WFOScheduler] 初始化失败: {e}，降级为无调度模式")

    # ── 冷启动 ────────────────────────────────────────────────────────────
    if scheduler is not None:
        try:
            scheduler.run_burn_in()
            log.info("[WFOScheduler] 冷启动完成")
        except Exception as e:
            log.warning(f"[WFOScheduler] 冷启动失败: {e}，跳过")
            scheduler = None
    else:
        log.info("冷启动跳过（WFOScheduler 不可用）")

    # ── 加载 AE + Scaler ─────────────────────────────────────────────────
    if ae_scaler_path and ae_scaler_path.exists():
        with open(ae_scaler_path, "rb") as f:
            scaler_state = pickle.load(f)
        log.info(f"AE Scaler: {ae_scaler_path}")
    else:
        scaler_state = None
        log.warning("AE Scaler 不存在")

    # ── 加载 features_master ────────────────────────────────────────────────
    paths_cfg = config.get("paths", {})
    features_path = PROJECT_ROOT / paths_cfg.get("data_processed", "data/processed") / "features_master.parquet"
    if features_path.exists():
        features_df = pd.read_parquet(features_path)
        weekly_return_cols = [c for c in features_df.columns if "__weekly_return" in c]
        log.info(f"features_master 已加载: {features_df.shape}")
    else:
        features_df = None
        weekly_return_cols = []
        log.warning("features_master 不存在，WFO将使用模拟数据")

    # ── 预计算每周5日窗口（百分比→小数）─────────────────────────────────
    if features_df is not None and "reconstruction_error" not in features_df.columns:
        if ae_weights_path and ae_weights_path.exists():
            try:
                model_cfg = config.get("model", {}).get("regime_autoencoder", {})
                ae_model = RegimeAutoEncoder(
                    input_dim=features_df.shape[1],
                    latent_dim=model_cfg.get("latent_dim", 6),
                    hidden_dim=model_cfg.get("hidden_dim", 16),
                ).to("cpu")
                ckpt = torch.load(ae_weights_path, map_location="cpu")
                ae_model.load_state_dict(ckpt.get("model_state", ckpt))
                X_ae = features_df.astype(float).fillna(0.0).values.astype(np.float32)
                features_df = features_df.copy()
                features_df["reconstruction_error"] = compute_reconstruction_error_batch(
                    ae_model,
                    X_ae,
                    "cpu",
                )
                log.info("AE reconstruction_error computed for gate diagnostics")
            except Exception as e:
                log.warning(f"AE reconstruction_error compute failed: {e}")
        else:
            log.warning("AE weights missing; reconstruction_error falls back to 0.0")

    if features_df is not None:
        features_df = features_df.copy()
        features_df["_week_end"] = features_df.index + pd.offsets.Week(weekday=4)
        returns_5d_by_week = {}
        for week_end, grp in features_df.groupby("_week_end"):
            if len(grp) >= 5:
                ret_5d = grp[weekly_return_cols].values[-5:].T / 100.0  # (5, 5)
            else:
                ret_5d = grp[weekly_return_cols].values.T / 100.0
            returns_5d_by_week[week_end] = ret_5d
        log.info(f"DualTrack 预计算完成: {len(returns_5d_by_week)} 周")
    else:
        returns_5d_by_week = {}

    # ── 加载 PPO Actor ──────────────────────────────────────────────────────
    if ppo_checkpoint and ppo_checkpoint.exists():
        from src.ppo.networks import ActorCritic
        ac = ActorCritic(
            state_dim=config.get("ppo", {}).get("state_dim", 10),
            action_dim=config.get("ppo", {}).get("action_dim", 2),
            hidden_dim=64,
        ).to("cpu" if not torch.cuda.is_available() else "cuda")
        ckpt = torch.load(ppo_checkpoint, map_location="cpu", weights_only=True)
        ac.load_state_dict(ckpt["ac"])
        ac.eval()
        state_assembler = StateAssembler(
            sharpe_clip_low=config.get("state_assembler", {}).get("sharpe_clip_low", -3.0),
            sharpe_clip_high=config.get("state_assembler", {}).get("sharpe_clip_high", 3.0),
        )
        log.info(f"PPO Actor 已加载: {ppo_checkpoint}")
    else:
        ac = None
        state_assembler = None
        log.warning("PPO checkpoint 不存在，使用固定 alpha=0.5")

    # ── 加载 LLM 评分历史 ─────────────────────────────────────────────────
    llm_scores_df = None
    llm_db_path = PROJECT_ROOT / "data" / "llm_cache" / "llm_scores.db"
    if llm_db_path.exists():
        import sqlite3
        try:
            conn = sqlite3.connect(llm_db_path)
            llm_scores_df = pd.read_sql(
                "SELECT week_end, concept, d1, d2, d3 FROM llm_scores WHERE error IS NULL",
                conn
            )
            conn.close()
            if llm_scores_df is not None and not llm_scores_df.empty:
                llm_scores_df = aggregate_llm_weekly(llm_scores_df)
                llm_scores_df = llm_scores_df.set_index("week_end")
                llm_scores_df.index = pd.to_datetime(llm_scores_df.index)
                llm_scores_df = llm_scores_df.sort_index()
                log.info(f"LLM 评分已加载: {len(llm_scores_df)} 条，覆盖 {llm_scores_df.index.min().date()} ~ {llm_scores_df.index.max().date()}")
            else:
                llm_scores_df = None
        except Exception as e:
            log.warning(f"LLM 评分加载失败: {e}")
            llm_scores_df = None
    else:
        log.info("LLM 评分数据库不存在，使用中性值")

    # ── WFO 滚动窗口 ─────────────────────────────────────────────────────
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    fridays = pd.bdate_range(start=start, end=end, freq="W-FRI")
    log.info(f"总周数: {len(fridays)}, 首周五: {fridays[0].date()}, 末周五: {fridays[-1].date()}")

    # ── Fit state_assembler on burn-in data (first 104 weeks) ──────────
    if state_assembler is not None and features_df is not None and len(fridays) >= 2:
        # Use first 2 years of data (104 weeks) for fitting normalizers
        sa_burnin_start = fridays[0] - pd.DateOffset(weeks=104)
        sa_burnin_data = features_df[features_df.index >= sa_burnin_start]
        if len(sa_burnin_data) > 0:
            ae_errors_for_sa = np.array(
                sa_burnin_data.get(
                    "reconstruction_error", pd.Series(0.0, index=sa_burnin_data.index)
                ).fillna(0.0).values
            )
            vol_for_sa = np.array(
                sa_burnin_data.get(
                    "000300.SH__volatility_20d", pd.Series(0.15, index=sa_burnin_data.index)
                ).fillna(0.15).values
            )
            tau_for_sa = np.full(len(sa_burnin_data), 0.5)
            state_assembler.fit_normalizers(ae_errors_for_sa, vol_for_sa, tau_for_sa)
            log.info(f"StateAssembler normalizers fitted on {len(sa_burnin_data)} burn-in samples")
        else:
            log.warning("Burn-in data empty, state_assembler not fitted")
    elif state_assembler is not None:
        log.warning("StateAssembler provided but insufficient data for fitting")

    log.info("══ WFO 主循环开始 ════════════════════════════════════════════")

    # ── 净值序列记录 ─────────────────────────────────────────────────────
    records = []
    gate_records = []
    nav = 1.0
    hwm = 1.0
    alpha_prev = config.get("env", {}).get("initial_alpha", 0.5)
    tau_prev = config.get("env", {}).get("initial_tau", 0.5)
    w_fused_prev = np.array([0.2] * 5)
    last_retrain_quarter = None  # 避免同一季度重复重训

    for i, friday in enumerate(fridays):
        week_label = friday.strftime("%Y-%m-%d")
        week_ts = pd.Timestamp(week_label)

        # ── (A) 季度边界检测 & 重训触发 ─────────────────────────────────
        if scheduler is not None:
            quarter = (friday.year, (friday.month - 1) // 3 + 1)
            if is_quarter_end(friday) and quarter != last_retrain_quarter:
                log.info(f"  ╔══ 季度末重训触发: {week_label} (Q{quarter[1]} {quarter[0]})")
                try:
                    retrain_path = scheduler.trigger_quarterly_retrain(week_label)
                    log.info(f"  ╚══ 季度重训完成: {retrain_path}")
                except Exception as e:
                    log.warning(f"  ╚══ 季度重训失败: {e}")
                last_retrain_quarter = quarter

            # ── (B) 周频推断 (E_t) ────────────────────────────────────────
            try:
                E_t_raw = scheduler.trigger_weekly_inference(week_label)
                log.debug(f"  Weekly inference E_t_raw={E_t_raw:.4f}")
            except Exception as e:
                log.debug(f"  Weekly inference failed: {e}")
                E_t_raw = None

        # ── (C) 双轨极限权重 ─────────────────────────────────────────────
        # 加载本周 LLM 信号（用于 EventTrack 进攻决策）
        llm_macro = llm_sentiment = llm_risk = 50.0
        if llm_scores_df is not None and not llm_scores_df.empty:
            try:
                available_llm = llm_scores_df.index[llm_scores_df.index <= week_ts]
                if len(available_llm) > 0:
                    last_date = available_llm[-1]
                    row_llm = llm_scores_df.loc[last_date]
                    if hasattr(row_llm, 'd1'):
                        llm_macro = float(row_llm.d1) if not pd.isna(row_llm.d1) else 50.0
                        llm_sentiment = float(row_llm.d2) if not pd.isna(row_llm.d2) else 50.0
                        llm_risk = float(row_llm.d3) if not pd.isna(row_llm.d3) else 50.0
            except Exception:
                pass

        try:
            if week_ts in returns_5d_by_week:
                returns_5d = returns_5d_by_week[week_ts]
            elif features_df is not None:
                available = sorted([k for k in returns_5d_by_week.keys() if k <= week_ts])
                returns_5d = (
                    returns_5d_by_week[available[-1]]
                    if available
                    else np.random.randn(5, 5) * 0.01
                )
            else:
                returns_5d = np.random.randn(5, 5) * 0.01
            w_normal, w_event = dual_engine.compute(
                returns_5d,
                llm_macro=llm_macro,
                llm_sentiment=llm_sentiment,
                llm_risk=llm_risk,
                ae_error=ae_error,
                tau=tau_prev,
            )
        except Exception as e:
            log.warning(f"双轨引擎失败: {e}，使用等权")
            w_normal = np.array([0.2] * 5)
            w_event = np.array([0.2] * 5)

        # ── (D) PPO Actor-Based Alpha Decision (with momentum fallback) ────
        # Extract market signals for actor input and fallback
        if features_df is not None:
            try:
                if week_ts in features_df.index:
                    row = features_df.loc[week_ts]
                else:
                    available = [d for d in features_df.index if d <= week_ts]
                    row = features_df.loc[max(available)]
                mom_20d = float(row.get("000300.SH__momentum_20d", 0.0))
                vol_mkt_20d = float(row.get("000300.SH__volatility_20d", 0.15))
            except Exception:
                mom_20d, vol_mkt_20d = 0.0, 0.15
        else:
            mom_20d, vol_mkt_20d = 0.0, 0.15

        ae_error = 0.0

        # Use PPO actor if available
        alpha_t = None
        if ac is not None and state_assembler is not None:
            try:
                # Extract AE error from features_master
                if features_df is not None:
                    try:
                        if week_ts in features_df.index:
                            ae_row = features_df.loc[week_ts]
                        else:
                            available = [d for d in features_df.index if d <= week_ts]
                            ae_row = features_df.loc[max(available)]
                        ae_error = float(ae_row.get("reconstruction_error", 0.0))
                    except Exception:
                        ae_error = 0.0
                else:
                    ae_error = 0.0

                # LLM scores from SQLite
                if llm_scores_df is not None and not llm_scores_df.empty:
                    try:
                        available_dates = llm_scores_df.index[llm_scores_df.index <= week_ts]
                        if len(available_dates) > 0:
                            last_date = available_dates[-1]
                            row_llm = llm_scores_df.loc[last_date]
                            if hasattr(row_llm, 'd1'):
                                llm_macro = float(row_llm.d1) if not pd.isna(row_llm.d1) else 50.0
                                llm_sentiment = float(row_llm.d2) if not pd.isna(row_llm.d2) else 50.0
                                llm_risk = float(row_llm.d3) if not pd.isna(row_llm.d3) else 50.0
                            else:
                                llm_macro = llm_sentiment = llm_risk = 50.0
                        else:
                            llm_macro = llm_sentiment = llm_risk = 50.0
                    except Exception:
                        llm_macro = llm_sentiment = llm_risk = 50.0
                else:
                    llm_macro = llm_sentiment = llm_risk = 50.0

                port_sharpe = 0.0  # not computed in backtest
                port_mdd = 0.0  # not tracked per-week
                regret_norm = 0.0  # will be computed after weight fusion

                S_t = state_assembler.assemble(
                    ae_error=ae_error,
                    vol_mkt_20d=vol_mkt_20d,
                    llm_macro=llm_macro,
                    llm_sentiment=llm_sentiment,
                    llm_risk=llm_risk,
                    port_sharpe_20d=port_sharpe,
                    port_mdd_current=port_mdd,
                    regret_ema_norm=regret_norm,
                    tau_prev=tau_prev,
                    alpha_prev=alpha_prev,
                )

                state_tensor = torch.from_numpy(S_t).float().unsqueeze(0)
                with torch.no_grad():
                    action_mean, _ = ac.actor(state_tensor)
                    a1, a2 = action_mean[0].numpy()

                    delta_alpha, delta_tau = action_mapper.map(a1, a2)
                alpha_candidate = alpha_prev + delta_alpha
                alpha_t = float(np.clip(alpha_candidate, 0.0, 1.0))

                if np.isnan(alpha_t):
                    raise ValueError("Actor output degenerate")

                log.debug(f"  Actor: mom={mom_20d:.2f}%, action=({a1:.3f},{a2:.3f}), alpha={alpha_t:.3f}")

            except Exception as e:
                log.debug(f"  Actor failed: {e}, using momentum fallback")

        # Fallback: Direct Momentum-Based Alpha Policy
        if alpha_t is None:
            ALPHA_HIGH = 0.6
            ALPHA_LOW = 0.0
            MOM_REF = 5.0
            mom_norm = float(np.clip(mom_20d / MOM_REF, -1.0, 1.0))
            alpha_t = float(np.clip(0.3 + mom_norm * 0.3, ALPHA_LOW, ALPHA_HIGH))
            # High volatility regime: reduce alpha further
            if features_df is not None:
                try:
                    if week_ts in features_df.index:
                        row = features_df.loc[week_ts]
                    else:
                        available = [d for d in features_df.index if d <= week_ts]
                        row = features_df.loc[max(available)]
                    vol_zscore = float(row.get("000300.SH__volatility_20d", 0.0))
                except Exception:
                    vol_zscore = 0.0
                if vol_zscore > 1.0:
                    alpha_t = float(np.clip(alpha_t - 0.15, ALPHA_LOW, ALPHA_HIGH))
            log.debug(f"  Momentum fallback: mom={mom_20d:.2f}%, alpha={alpha_t:.3f}")

        # Force alpha override (for diagnostic testing)
        if force_alpha is not None:
            alpha_t = float(np.clip(force_alpha, 0.0, 1.0))
            log.debug(f"  Force alpha: {alpha_t:.3f}")

        # Apply PPO's delta_tau (Stage 2 fix: prior code used `tau_t = tau_prev` which wasted the a2 action)
        env_cfg = config.get("env", {})
        tau_t = action_mapper.clip_tau(
            tau_prev + delta_tau,
            env_cfg.get("tau_min", 5.0),
            env_cfg.get("tau_max", 50.0),
        )

        # ── (E) 融合权重 ─────────────────────────────────────────────────
        # w_final = alpha_t * w_event + (1 - alpha_t) * w_normal
        w_fused = alpha_t * w_event + (1 - alpha_t) * w_normal
        w_fused = np.clip(w_fused, 0, 1)
        w_fused = w_fused / (w_fused.sum() + 1e-9)

        # ── (F) Regret 计算（百分比→小数）───────────────────────────────
        if features_df is not None:
            if week_ts in features_df.index:
                week_returns = (
                    features_df.loc[week_ts, weekly_return_cols].values.astype(float) / 100.0
                )
            else:
                available = [d for d in features_df.index if d <= week_ts]
                if available:
                    week_returns = (
                        features_df.loc[max(available), weekly_return_cols].values.astype(float) / 100.0
                    )
                else:
                    week_returns = np.random.randn(5) * 0.01
        else:
            week_returns = np.random.randn(5) * 0.01
        week_returns = np.nan_to_num(week_returns, nan=0.0)

        try:
            _, regret_norm = regret_engine.compute(w_fused_prev, week_returns)
        except Exception:
            regret_norm = 0.0

        # ── (G) Veto Switch ──────────────────────────────────────────────
        if regret_norm > 0.8:
            log.warning(f"  regret偏高: {week_label} regret={regret_norm:.3f}")

        # ── (H) 本周组合收益 ─────────────────────────────────────────────
        r_normal = float(np.dot(w_normal, week_returns))
        r_event = float(np.dot(w_event, week_returns))
        edge_event_minus_normal = r_event - r_normal
        regime_label = "bull_normal" if ae_error < tau_t else "event_stress"
        week_return = float(np.dot(w_fused, week_returns))
        nav = nav * (1 + week_return)
        hwm = max(hwm, nav)

        records.append({
            "date": week_label,
            "NAV": nav,
            "alpha": alpha_t,
            "tau": tau_t,
            "regret_ema_norm": regret_norm,
            "week_return": week_return,
            "max_drawdown": (hwm - nav) / hwm,
            "w_fused": w_fused.copy(),  # 保存用于 weights_data 输出
        })

        gate_records.append({
            "date": week_label,
            "alpha": alpha_t,
            "tau": tau_t,
            "ae_error": ae_error,
            "regime_label": regime_label,
            "llm_macro": llm_macro,
            "llm_sentiment": llm_sentiment,
            "llm_risk": llm_risk,
            "r_normal": r_normal,
            "r_event": r_event,
            "edge_event_minus_normal": edge_event_minus_normal,
            "week_return_fused": week_return,
            "regret_ema_norm": regret_norm,
            "w_normal_broad": float(w_normal[0]),
            "w_normal_satellite": float(w_normal[1]),
            "w_normal_fi": float(w_normal[2]),
            "w_normal_safe": float(w_normal[3]),
            "w_normal_cash": float(w_normal[4]),
            "w_event_broad": float(w_event[0]),
            "w_event_satellite": float(w_event[1]),
            "w_event_fi": float(w_event[2]),
            "w_event_safe": float(w_event[3]),
            "w_event_cash": float(w_event[4]),
            "w_fused_broad": float(w_fused[0]),
            "w_fused_satellite": float(w_fused[1]),
            "w_fused_fi": float(w_fused[2]),
            "w_fused_safe": float(w_fused[3]),
            "w_fused_cash": float(w_fused[4]),
        })

        w_fused_prev = w_fused.copy()
        alpha_prev, tau_prev = alpha_t, tau_t

        if (i + 1) % 52 == 0:
            log.info(
                f"  Week {i+1:4d}/{len(fridays)}: NAV={nav:.4f}  "
                f"alpha={alpha_t:.3f}  regret={regret_norm:.3f}"
            )

    # ── 构建净值 DataFrame ─────────────────────────────────────────────────
    df_nav = pd.DataFrame(records).set_index("date")
    df_nav.index = pd.to_datetime(df_nav.index)

    # ── 构建 weights_data (GeneralBacktest 格式) ───────────────────────────
    weights_records = [
        {k: v for k, v in rec.items() if k != "w_fused"}
        for rec in records
    ]
    for rec, orig in zip(weights_records, records):
        rec["w_fused"] = orig["w_fused"]
    df_weights_gb = build_weights_data(weights_records, ASSET_CODES)

    # ── 落盘 ──────────────────────────────────────────────────────────────
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = output_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    nav_path = out_dir / "nav_series.csv"
    weights_path = out_dir / "weights_trajectory.csv"
    weights_gb_path = out_dir / "weights_data_generalbt.csv"  # GeneralBacktest格式
    metrics_path = out_dir / "metrics.json"
    gate_diag_path = out_dir / "gate_diagnostics.csv"

    df_nav.to_csv(nav_path)
    df_gate = pd.DataFrame(gate_records)
    df_gate.to_csv(gate_diag_path, index=False)
    log.info(f"净值序列: {nav_path}")

    # 原始格式
    df_weights_legacy = df_nav.reset_index(drop=True)
    df_weights_legacy.to_csv(weights_path, index=False)
    log.info(f"权重轨迹(legacy): {weights_path}")

    # GeneralBacktest 格式
    df_weights_gb.to_csv(weights_gb_path, index=False)
    log.info(f"权重轨迹(GeneralBacktest): {weights_gb_path}")

    metrics = compute_wfo_metrics(df_nav["NAV"])
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"指标: {metrics}")

    tearsheet = out_dir / "tearsheet.txt"
    with open(tearsheet, "w") as f:
        f.write(f"═══════════════════════════════════════════\n")
        f.write(f"  AE-LLM-RL-FOF Walk-Forward 回测撕页\n")
        f.write(f"═══════════════════════════════════════════\n")
        f.write(f"回测区间: {fridays[0].date()} → {fridays[-1].date()}\n")
        f.write(f"WFO窗口: {lookback_weeks}周\n")
        f.write(f"总周数: {len(fridays)}\n")
        f.write(f"WFOScheduler: {'启用' if scheduler else '禁用'}\n")
        f.write(f"\n── 业绩指标 ──────────────────────────────\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write(f"\n── 策略参数 ──────────────────────────────\n")
        f.write(f"  alpha (融合比): mean={df_nav['alpha'].mean():.3f}, std={df_nav['alpha'].std():.3f}\n")
        f.write(f"  tau (阈值):     mean={df_nav['tau'].mean():.3f}, std={df_nav['tau'].std():.3f}\n")
        f.write(f"  Regret_ema:    mean={df_nav['regret_ema_norm'].mean():.3f}\n")
    log.info(f"撕页: {tearsheet}")

    log.info(f"══ WFO 回测完成！输出目录: {out_dir} ═")

    # ── 集成 GeneralBacktest 绘图 ──────────────────────────────────────────
    log.info("══ 集成 GeneralBacktest 绘图 ═")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from GeneralBacktest import GeneralBacktest
        from src.data_pipeline.track_b.fetcher import fetch_track_b

        # 1. 从 ClickHouse 获取日频 OHLC
        price_raw = fetch_track_b(
            start_date=start_date,
            end_date=end_date,
            columns=["open", "close", "adj_factor"],
        )  # 返回 DataFrame, index=date, columns=[code__open, code__close, code__adj_factor]

        # 2. 转换为 GeneralBacktest 格式 (melt)
        records_for_gbt = []
        for date_str, row in price_raw.iterrows():
            for code in ASSET_CODES:
                open_col = f"{code}__open"
                close_col = f"{code}__close"
                adj_col = f"{code}__adj_factor"
                open_val = row.get(open_col, None)
                close_val = row.get(close_col, None)
                adj_val = row.get(adj_col, None)

                # fallback: open -> close, adj_factor -> 1.0
                if pd.isna(close_val) and pd.isna(open_val):
                    continue
                final_close = float(close_val) if not pd.isna(close_val) else float(open_val)
                final_open = float(open_val) if not pd.isna(open_val) else final_close
                final_adj = float(adj_val) if adj_val is not None and not pd.isna(adj_val) else 1.0

                records_for_gbt.append({
                    "date": date_str.strftime("%Y-%m-%d"),
                    "code": code,
                    "open": final_open,
                    "close": final_close,
                    "adj_factor": final_adj,
                })
        price_data = pd.DataFrame(records_for_gbt)

        # 3. 构建 weights_data (从 records 中提取 w_fused)
        weights_data = build_weights_data(records, ASSET_CODES)

        # 4. 运行 GeneralBacktest
        bt = GeneralBacktest(
            start_date=start_date,
            end_date=end_date,
        )
        bt.run_backtest(
            weights_data=weights_data,
            price_data=price_data,
            buy_price="open",
            sell_price="close",
            adj_factor_col="adj_factor",
            close_price_col="close",
            rebalance_threshold=0.005,
            transaction_cost=[0.0003, 0.0003],
            slippage=0.0001,
            initial_capital=1.0,
        )

        # 5. 打印指标 & 绘图
        bt.print_metrics()

        # 5a. 保存 GeneralBacktest 真实业绩到 metrics_real.json
        # GeneralBacktest.metrics keys 是中文,转换为英文便于 verify 脚本读取
        gbt_metrics = bt.metrics or {}
        key_map = {
            "累计收益率": "total_return",
            "年化收益率": "annualized_return",
            "年化波动率": "volatility_annual",
            "最大回撤": "max_drawdown",
            "夏普比率": "sharpe_ratio",
            "卡玛比率": "calmar_ratio",
            "索提诺比率": "sortino_ratio",
            "胜率": "win_rate",
            "交易次数": "n_trades",
            "平均换手率": "avg_turnover",
            "累计换手率": "cumulative_turnover",
            "VaR (95%)": "var_95",
            "CVaR (95%)": "cvar_95",
        }
        metrics_real = {}
        for cn_key, en_key in key_map.items():
            if cn_key in gbt_metrics:
                v = gbt_metrics[cn_key]
                try:
                    metrics_real[en_key] = float(v)
                except (TypeError, ValueError):
                    metrics_real[en_key] = str(v)
        metrics_real["n_weeks"] = len(df_nav)
        metrics_real["_source"] = "generalbacktest_real_ohlc"
        metrics_real_path = out_dir / "metrics_real.json"
        with open(metrics_real_path, "w", encoding="utf-8") as f:
            json.dump(metrics_real, f, indent=2, ensure_ascii=False)
        log.info(f"真实业绩指标(GeneralBacktest): {metrics_real_path}")

        fig_dir = out_dir / "figures"
        fig_dir.mkdir(exist_ok=True)

        # plot_dashboard 会显示所有图，保存一份
        try:
            bt.plot_dashboard(save_path=str(fig_dir / "dashboard.png"))
        except Exception:
            pass  # dashboard 绘图失败不阻塞

        # 分别保存各子图
        plot_methods = [
            ("nav_curve", "nav_curve.png"),
            ("excess_returns", "excess_returns.png"),
            ("monthly_returns_heatmap", "monthly_returns.png"),
            ("drawdown", "drawdown.png"),
        ]
        for method_name, filename in plot_methods:
            try:
                method = getattr(bt, f"plot_{method_name}", None)
                if method is not None:
                    method(save_path=str(fig_dir / filename))
            except Exception:
                pass

        log.info(f"GeneralBacktest 绘图完成，图片保存在 {fig_dir}")

    except Exception as e:
        log.warning(f"GeneralBacktest 绘图失败: {e}，跳过绘图")

    return df_nav


def main() -> None:
    parser = argparse.ArgumentParser(description="run_backtest_wfo: WFO滚动回测")
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--lookback-weeks", type=int, default=104)
    parser.add_argument("--output-dir", default="results/wfo")
    parser.add_argument(
        "--no-scheduler",
        action="store_true",
        help="禁用 WFOScheduler（降级为纯手动NAV模式）",
    )
    parser.add_argument(
        "--force-alpha",
        type=float,
        default=None,
        help="强制指定alpha值（0=纯NormalTrack, 1=纯EventTrack），覆盖Actor决策",
    )
    args = parser.parse_args()

    config = load_config()
    paths_cfg = config.get("paths", {})

    output_dir = PROJECT_ROOT / args.output_dir

    ae_scaler = PROJECT_ROOT / paths_cfg.get("checkpoints", "checkpoints") / "ae_scaler.pkl"
    ae_weights = PROJECT_ROOT / paths_cfg.get("checkpoints", "checkpoints") / "ae_weights.pth"
    ppo_ckpt = PROJECT_ROOT / paths_cfg.get("checkpoints", "checkpoints") / "actor_critic.pth"

    try:
        df = run_wfo(
            start_date=args.start_date,
            end_date=args.end_date,
            lookback_weeks=args.lookback_weeks,
            output_dir=output_dir,
            config=config,
            ae_scaler_path=ae_scaler if ae_scaler.exists() else None,
            ae_weights_path=ae_weights if ae_weights.exists() else None,
            ppo_checkpoint=ppo_ckpt if ppo_ckpt.exists() else None,
            use_wfo_scheduler=not args.no_scheduler,
            force_alpha=args.force_alpha,
        )
        log.info("WFO 回测完成")
    except Exception as e:
        log.error(f"WFO 回测失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
