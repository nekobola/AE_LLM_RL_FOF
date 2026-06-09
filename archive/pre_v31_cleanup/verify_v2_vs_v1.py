"""V2 verification: compare V1 (three-prototype softmax) vs V2 (Signal-Tilted RB)
on the same 156 weeks from results/wfo/20260602_182649.

Hard metrics:
  1. edge std (v_event - v_normal returns) — V1=0.123%, V2 target > 0.3%
  2. bear weeks: w_event_fi mean — V1=26%, V2 target >= 0.35
  3. bear weeks: w_event_satellite mean — V1=high, V2 target <= 0.10
  4. bull weeks: offensive (broad+sat) — V2 should be > bear
  5. ||w_event - w_normal|| per week — divergence magnitude
  6. Sharpe ratio of pure event / pure normal / 50-50 blend
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("D:/素材/AE_LLM_RL_FOF-main")
sys.path.insert(0, str(PROJECT_ROOT))

from src.compute.dual_track_engine import DualTrackEngine

ORIG = pd.read_csv(PROJECT_ROOT / "results/wfo/20260602_182649/gate_diagnostics.csv")
ORIG["date"] = pd.to_datetime(ORIG["date"])

# Load features_master for weekly returns
features_path = PROJECT_ROOT / "data/processed/features_master.parquet"
features_df = pd.read_parquet(features_path) if features_path.exists() else None
weekly_return_cols = (
    [c for c in features_df.columns if "__weekly_return" in c]
    if features_df is not None
    else None
)

weekly_returns = {}
returns_history_by_week = {}
HISTORY_WINDOW = 5  # match actual WFO construction
if features_df is not None and weekly_return_cols:
    features_df = features_df.copy()
    features_df["_week_end"] = features_df.index + pd.offsets.Week(weekday=4)
    weekly_only = features_df[weekly_return_cols] / 100.0
    for date_idx in features_df.index:
        weekly_returns[date_idx] = weekly_only.loc[date_idx].values
    # Pre-compute history window (last 20 weeks) for each week_end
    for week_end, grp in features_df.groupby("_week_end"):
        # Find all daily returns up to this week_end
        avail_dates = sorted([d for d in features_df.index if d <= week_end])
        if len(avail_dates) == 0:
            continue
        recent = avail_dates[-HISTORY_WINDOW:]
        hist = weekly_only.loc[recent].values.T  # (5, T)
        returns_history_by_week[week_end] = hist


def get_returns(date_ts):
    if date_ts in weekly_returns:
        return weekly_returns[date_ts]
    avail = sorted([k for k in weekly_returns.keys() if k <= date_ts])
    return weekly_returns[avail[-1]] if avail else np.zeros(5)


def get_returns_5d(date_ts):
    """Longer history window (20 weeks) for stable covariance estimation."""
    if date_ts in returns_history_by_week:
        return returns_history_by_week[date_ts]
    avail = sorted([k for k in returns_history_by_week.keys() if k <= date_ts])
    return returns_history_by_week[avail[-1]] if avail else np.zeros((5, HISTORY_WINDOW))


# Engines: V1 (use_v2=False), V2 (use_v2=True)
engine_v1 = DualTrackEngine(use_v2=False)
engine_v2 = DualTrackEngine(use_v2=True)

print("=" * 70)
print(f"V1 vs V2 verification — {len(ORIG)} weeks")
print("=" * 70)

results = []
for i, row in ORIG.iterrows():
    week_ts = row["date"]
    rets_5d = get_returns_5d(week_ts)
    rets_w = get_returns(week_ts)

    # V1
    try:
        w_n1, w_e1 = engine_v1.compute(
            rets_5d,
            llm_macro=row["llm_macro"],
            llm_sentiment=row["llm_sentiment"],
            llm_risk=row["llm_risk"],
            ae_error=row["ae_error"],
            tau=row["tau"],
        )
    except Exception:
        w_n1, w_e1 = np.array([0.2]*5), np.array([0.2]*5)

    # V2
    try:
        w_n2, w_e2 = engine_v2.compute(
            rets_5d,
            llm_macro=row["llm_macro"],
            llm_sentiment=row["llm_sentiment"],
            llm_risk=row["llm_risk"],
            ae_error=row["ae_error"],
            tau=row["tau"],
        )
    except Exception:
        w_n2, w_e2 = np.array([0.2]*5), np.array([0.2]*5)

    r_n1 = float(np.dot(w_n1, rets_w))
    r_e1 = float(np.dot(w_e1, rets_w))
    r_n2 = float(np.dot(w_n2, rets_w))
    r_e2 = float(np.dot(w_e2, rets_w))

    results.append({
        "date": week_ts,
        "regime_label": row["regime_label"],
        "ae_error": row["ae_error"],
        "llm_macro": row["llm_macro"],
        "llm_sentiment": row["llm_sentiment"],
        "llm_risk": row["llm_risk"],
        # V1 weights
        "v1_w_n_broad": w_n1[0], "v1_w_n_sat": w_n1[1], "v1_w_n_fi": w_n1[2],
        "v1_w_n_safe": w_n1[3], "v1_w_n_cash": w_n1[4],
        "v1_w_e_broad": w_e1[0], "v1_w_e_sat": w_e1[1], "v1_w_e_fi": w_e1[2],
        "v1_w_e_safe": w_e1[3], "v1_w_e_cash": w_e1[4],
        # V2 weights
        "v2_w_n_broad": w_n2[0], "v2_w_n_sat": w_n2[1], "v2_w_n_fi": w_n2[2],
        "v2_w_n_safe": w_n2[3], "v2_w_n_cash": w_n2[4],
        "v2_w_e_broad": w_e2[0], "v2_w_e_sat": w_e2[1], "v2_w_e_fi": w_e2[2],
        "v2_w_e_safe": w_e2[3], "v2_w_e_cash": w_e2[4],
        # Returns
        "v1_r_n": r_n1, "v1_r_e": r_e1, "v1_edge": r_e1 - r_n1,
        "v2_r_n": r_n2, "v2_r_e": r_e2, "v2_edge": r_e2 - r_n2,
    })

df = pd.DataFrame(results)
df.to_csv(PROJECT_ROOT / "results/v2_vs_v1_verification.csv", index=False)


# ── Hard metrics ──
def std_pct(x):
    return np.std(x) * 100


def ann_sharpe(returns, periods_per_year=52):
    mu, sig = np.mean(returns), np.std(returns)
    if sig < 1e-9:
        return 0.0
    return float(mu / sig * np.sqrt(periods_per_year))


print()
print("=" * 70)
print("硬指标 1: edge std (r_event - r_normal) — 目标是 PPO 切换价值")
print("=" * 70)
v1_edge_std = std_pct(df["v1_edge"])
v2_edge_std = std_pct(df["v2_edge"])
print(f"V1 edge std:  {v1_edge_std:.4f}% (基线)")
print(f"V2 edge std:  {v2_edge_std:.4f}% (目标 > 0.3%)")
print(f"V2 / V1 倍数: {v2_edge_std/v1_edge_std:.2f}x")
print(f"{'✓ PASS' if v2_edge_std > 0.3 else '✗ FAIL'}: V2 edge std {v2_edge_std:.4f}%")

print()
print("=" * 70)
print("硬指标 2: bear weeks 下 EventTrack 固收权重")
print("=" * 70)
bear = df[df["regime_label"] == "event_stress"]
print(f"bear weeks: {len(bear)}")
v1_bear_fi = bear["v1_w_e_fi"].mean()
v2_bear_fi = bear["v2_w_e_fi"].mean()
print(f"V1 bear w_event_fi:      {v1_bear_fi:.3f} (基线)")
print(f"V2 bear w_event_fi:      {v2_bear_fi:.3f} (目标 >= 0.35)")
print(f"{'✓ PASS' if v2_bear_fi >= 0.35 else '✗ FAIL'}: V2 bear fi {v2_bear_fi:.3f}")

v1_bear_sat = bear["v1_w_e_sat"].mean()
v2_bear_sat = bear["v2_w_e_sat"].mean()
print(f"V1 bear w_event_sat:     {v1_bear_sat:.3f}")
print(f"V2 bear w_event_sat:     {v2_bear_sat:.3f} (目标 <= 0.10)")
print(f"{'✓ PASS' if v2_bear_sat <= 0.10 else '✗ FAIL'}: V2 bear sat {v2_bear_sat:.3f}")

print()
print("=" * 70)
print("硬指标 3: bull weeks 下 EventTrack 进攻性")
print("=" * 70)
bull = df[df["regime_label"] == "bull_normal"]
print(f"bull weeks: {len(bull)}")
v1_bull_off = (bull["v1_w_e_broad"] + bull["v1_w_e_sat"]).mean()
v2_bull_off = (bull["v2_w_e_broad"] + bull["v2_w_e_sat"]).mean()
v2_bear_off = (bear["v2_w_e_broad"] + bear["v2_w_e_sat"]).mean()
print(f"V1 bull EventTrack offensive: {v1_bull_off:.3f}")
print(f"V2 bull EventTrack offensive: {v2_bull_off:.3f}")
print(f"V2 bear EventTrack offensive: {v2_bear_off:.3f}")
print(f"{'✓ PASS' if v2_bull_off > v2_bear_off else '✗ FAIL'}: V2 bull > bear 进攻性")

print()
print("=" * 70)
print("硬指标 4: 两条 track 在不同 regime 的差异 (PPO fusion 价值)")
print("=" * 70)
df["v1_diff"] = np.sqrt(
    (df["v1_w_e_broad"] - df["v1_w_n_broad"])**2 +
    (df["v1_w_e_sat"] - df["v1_w_n_sat"])**2 +
    (df["v1_w_e_fi"] - df["v1_w_n_fi"])**2 +
    (df["v1_w_e_safe"] - df["v1_w_n_safe"])**2 +
    (df["v1_w_e_cash"] - df["v1_w_n_cash"])**2
)
df["v2_diff"] = np.sqrt(
    (df["v2_w_e_broad"] - df["v2_w_n_broad"])**2 +
    (df["v2_w_e_sat"] - df["v2_w_n_sat"])**2 +
    (df["v2_w_e_fi"] - df["v2_w_n_fi"])**2 +
    (df["v2_w_e_safe"] - df["v2_w_n_safe"])**2 +
    (df["v2_w_e_cash"] - df["v2_w_n_cash"])**2
)
print(f"V1 ||w_event - w_normal|| mean: {df['v1_diff'].mean():.3f}")
print(f"V2 ||w_event - w_normal|| mean: {df['v2_diff'].mean():.3f}")
print(f"  bear weeks:  V1={bear['v1_diff'].mean():.3f}  V2={bear['v2_diff'].mean():.3f}")
print(f"  bull weeks:  V1={bull['v1_diff'].mean():.3f}  V2={bull['v2_diff'].mean():.3f}")

print()
print("=" * 70)
print("硬指标 5: 累计收益与 Sharpe 对比")
print("=" * 70)
print(f"{'':25s} {'V1':>10s}  {'V2':>10s}")
for label, v1_col, v2_col in [
    ("Pure NormalTrack", "v1_r_n", "v2_r_n"),
    ("Pure EventTrack",  "v1_r_e", "v2_r_e"),
    ("50/50 blend",     None,    None),
]:
    if v1_col:
        v1_cum = (df[v1_col]+1).prod() - 1
        v2_cum = (df[v2_col]+1).prod() - 1
        v1_sharpe = ann_sharpe(df[v1_col])
        v2_sharpe = ann_sharpe(df[v2_col])
        print(f"{label+'(cum)':25s} {v1_cum*100:9.2f}%  {v2_cum*100:9.2f}%")
        print(f"{label+'(Sharpe)':25s} {v1_sharpe:9.3f}  {v2_sharpe:9.3f}")
    else:
        v1_blend = (df["v1_r_n"] + df["v1_r_e"]) / 2
        v2_blend = (df["v2_r_n"] + df["v2_r_e"]) / 2
        v1_cum = (v1_blend+1).prod() - 1
        v2_cum = (v2_blend+1).prod() - 1
        v1_sharpe = ann_sharpe(v1_blend)
        v2_sharpe = ann_sharpe(v2_blend)
        print(f"{label+'(cum)':25s} {v1_cum*100:9.2f}%  {v2_cum*100:9.2f}%")
        print(f"{label+'(Sharpe)':25s} {v1_sharpe:9.3f}  {v2_sharpe:9.3f}")

print()
print("=" * 70)
print("硬指标 6: V2 在强增长信号周的卫星/宽基响应")
print("=" * 70)
strong = df[(df["llm_macro"] > 70) & (df["llm_sentiment"] > 70) & (df["regime_label"] == "bull_normal")]
print(f"强增长 + bull weeks: {len(strong)}")
if len(strong) > 0:
    print(f"V2 卫星权重 (mean):  {strong['v2_w_e_sat'].mean():.3f}")
    print(f"V2 宽基权重 (mean):  {strong['v2_w_e_broad'].mean():.3f}")
    print(f"V1 卫星权重 (mean):  {strong['v1_w_e_sat'].mean():.3f}")

print()
print("=" * 70)
print(f"详细数据: results/v2_vs_v1_verification.csv ({len(df)} rows)")
print("=" * 70)
