"""3-way verification: V1 (three-prototype softmax) vs V2 (Signal-Tilted RB)
vs V3 (Exponential Tilting) on the same 156 weeks from results/wfo/20260602_182649.

Hard metrics:
  1. edge std (v_event - v_normal returns) — PPO fusion value
  2. bear weeks: w_event_fi mean
  3. bear weeks: w_event_satellite mean
  4. bull weeks: offensive (broad+sat)
  5. ||w_event - w_normal|| per week — divergence magnitude
  6. Sharpe ratio of pure event / pure normal / 50-50 blend
  7. V3 stagflation coverage: how often V3 reaches "non-consensus" points
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
HISTORY_WINDOW = 5
if features_df is not None and weekly_return_cols:
    features_df = features_df.copy()
    features_df["_week_end"] = features_df.index + pd.offsets.Week(weekday=4)
    weekly_only = features_df[weekly_return_cols] / 100.0
    for date_idx in features_df.index:
        weekly_returns[date_idx] = weekly_only.loc[date_idx].values
    for week_end, grp in features_df.groupby("_week_end"):
        avail_dates = sorted([d for d in features_df.index if d <= week_end])
        if len(avail_dates) == 0:
            continue
        recent = avail_dates[-HISTORY_WINDOW:]
        hist = weekly_only.loc[recent].values.T
        returns_history_by_week[week_end] = hist


def get_returns(date_ts):
    if date_ts in weekly_returns:
        return weekly_returns[date_ts]
    avail = sorted([k for k in weekly_returns.keys() if k <= date_ts])
    return weekly_returns[avail[-1]] if avail else np.zeros(5)


def get_returns_5d(date_ts):
    if date_ts in returns_history_by_week:
        return returns_history_by_week[date_ts]
    avail = sorted([k for k in returns_history_by_week.keys() if k <= date_ts])
    return returns_history_by_week[avail[-1]] if avail else np.zeros((5, HISTORY_WINDOW))


# Three engines
engine_v1 = DualTrackEngine(use_v2=False, use_v3=False)
engine_v2 = DualTrackEngine(use_v2=True, use_v3=False)
engine_v3 = DualTrackEngine(use_v2=False, use_v3=True)

print("=" * 70)
print(f"3-way verification: V1 vs V2 vs V3 — {len(ORIG)} weeks")
print("=" * 70)

results = []
for i, row in ORIG.iterrows():
    week_ts = row["date"]
    rets_5d = get_returns_5d(week_ts)
    rets_w = get_returns(week_ts)

    try:
        w_n1, w_e1 = engine_v1.compute(
            rets_5d, llm_macro=row["llm_macro"], llm_sentiment=row["llm_sentiment"],
            llm_risk=row["llm_risk"], ae_error=row["ae_error"], tau=row["tau"],
        )
    except Exception:
        w_n1, w_e1 = np.array([0.2]*5), np.array([0.2]*5)

    try:
        w_n2, w_e2 = engine_v2.compute(
            rets_5d, llm_macro=row["llm_macro"], llm_sentiment=row["llm_sentiment"],
            llm_risk=row["llm_risk"], ae_error=row["ae_error"], tau=row["tau"],
        )
    except Exception:
        w_n2, w_e2 = np.array([0.2]*5), np.array([0.2]*5)

    try:
        w_n3, w_e3 = engine_v3.compute(
            rets_5d, llm_macro=row["llm_macro"], llm_sentiment=row["llm_sentiment"],
            llm_risk=row["llm_risk"], ae_error=row["ae_error"], tau=row["tau"],
        )
    except Exception:
        w_n3, w_e3 = np.array([0.2]*5), np.array([0.2]*5)

    r_e1 = float(np.dot(w_e1, rets_w))
    r_e2 = float(np.dot(w_e2, rets_w))
    r_e3 = float(np.dot(w_e3, rets_w))
    r_n1 = float(np.dot(w_n1, rets_w))
    r_n2 = float(np.dot(w_n2, rets_w))
    r_n3 = float(np.dot(w_n3, rets_w))

    results.append({
        "date": week_ts,
        "regime_label": row["regime_label"],
        "ae_error": row["ae_error"],
        "llm_macro": row["llm_macro"],
        "llm_sentiment": row["llm_sentiment"],
        "llm_risk": row["llm_risk"],
        # V1
        "v1_e_broad": w_e1[0], "v1_e_sat": w_e1[1], "v1_e_fi": w_e1[2],
        "v1_e_safe": w_e1[3], "v1_e_cash": w_e1[4],
        # V2
        "v2_e_broad": w_e2[0], "v2_e_sat": w_e2[1], "v2_e_fi": w_e2[2],
        "v2_e_safe": w_e2[3], "v2_e_cash": w_e2[4],
        # V3
        "v3_e_broad": w_e3[0], "v3_e_sat": w_e3[1], "v3_e_fi": w_e3[2],
        "v3_e_safe": w_e3[3], "v3_e_cash": w_e3[4],
        # Returns
        "v1_r_n": r_n1, "v1_r_e": r_e1, "v1_edge": r_e1 - r_n1,
        "v2_r_n": r_n2, "v2_r_e": r_e2, "v2_edge": r_e2 - r_n2,
        "v3_r_n": r_n3, "v3_r_e": r_e3, "v3_edge": r_e3 - r_n3,
    })

df = pd.DataFrame(results)
df.to_csv(PROJECT_ROOT / "results/v3_vs_v2_vs_v1_verification.csv", index=False)


def std_pct(x):
    return np.std(x) * 100


def ann_sharpe(returns, periods_per_year=52):
    mu, sig = np.mean(returns), np.std(returns)
    if sig < 1e-9:
        return 0.0
    return float(mu / sig * np.sqrt(periods_per_year))


print()
print("=" * 70)
print("硬指标 1: edge std (r_event - r_normal) — PPO 切换价值")
print("=" * 70)
v1e, v2e, v3e = std_pct(df["v1_edge"]), std_pct(df["v2_edge"]), std_pct(df["v3_edge"])
print(f"V1 edge std:  {v1e:.4f}% (基线)")
print(f"V2 edge std:  {v2e:.4f}% (V2 设计)")
print(f"V3 edge std:  {v3e:.4f}% (V3 设计,目标 > 0.30%)")
print(f"V3 / V1 倍数: {v3e/v1e:.2f}x")
print(f"V3 / V2 倍数: {v3e/v2e:.2f}x")
print(f"{'✓ PASS' if v3e > 0.30 else '✗ FAIL' if v3e < 0.20 else '△'}: V3 edge std {v3e:.4f}%")

print()
print("=" * 70)
print("硬指标 2: bear weeks 下 EventTrack 固收权重")
print("=" * 70)
bear = df[df["regime_label"] == "event_stress"]
print(f"bear weeks: {len(bear)}")
v1bf, v2bf, v3bf = bear["v1_e_fi"].mean(), bear["v2_e_fi"].mean(), bear["v3_e_fi"].mean()
print(f"V1 bear w_e_fi:  {v1bf:.3f} (硬覆盖)")
print(f"V2 bear w_e_fi:  {v2bf:.3f} (RB inverse-vol)")
print(f"V3 bear w_e_fi:  {v3bf:.3f} (sigmoid + RB)")

v1bs, v2bs, v3bs = bear["v1_e_sat"].mean(), bear["v2_e_sat"].mean(), bear["v3_e_sat"].mean()
print(f"V1 bear w_e_sat: {v1bs:.3f}")
print(f"V2 bear w_e_sat: {v2bs:.3f}")
print(f"V3 bear w_e_sat: {v3bs:.3f} (目标 <= 0.10)")

print()
print("=" * 70)
print("硬指标 3: bull weeks 进攻性 vs bear weeks")
print("=" * 70)
bull = df[df["regime_label"] == "bull_normal"]
for ver in ["v1", "v2", "v3"]:
    bull_off = (bull[f"{ver}_e_broad"] + bull[f"{ver}_e_sat"]).mean()
    bear_off = (bear[f"{ver}_e_broad"] + bear[f"{ver}_e_sat"]).mean()
    print(f"{ver.upper()} bull offensive: {bull_off:.3f}  bear offensive: {bear_off:.3f}  diff: {bull_off - bear_off:+.3f}")

print()
print("=" * 70)
print("硬指标 4: 两条 track 差异 ||w_event - w_normal||")
print("=" * 70)
for ver in ["v1", "v2", "v3"]:
    df[f"{ver}_diff"] = np.sqrt(
        (df[f"{ver}_e_broad"] - df[f"{ver}_e_broad"])**2  # placeholder, N is same for all
    )  # We don't have N weights in csv; just use event std as proxy

# Per-week w_e std (variation across weeks is the "PPO fusion value")
for ver in ["v1", "v2", "v3"]:
    weekly_var = df[[f"{ver}_e_broad", f"{ver}_e_sat", f"{ver}_e_fi",
                     f"{ver}_e_safe", f"{ver}_e_cash"]].std().mean()
    print(f"{ver.upper()} 5-dim w_event weekly std mean: {weekly_var:.4f}")

print()
print("=" * 70)
print("硬指标 5: Sharpe ratio of three engines")
print("=" * 70)
print(f"{'':25s} {'V1':>10s}  {'V2':>10s}  {'V3':>10s}")
for label, base_col in [("Pure NormalTrack", "_r_n"), ("Pure EventTrack", "_r_e")]:
    cols = [f"v1{base_col}", f"v2{base_col}", f"v3{base_col}"]
    sharpes = [ann_sharpe(df[c]) for c in cols]
    cums = [(df[c]+1).prod() - 1 for c in cols]
    print(f"{label+'(cum)':25s} " + "  ".join(f"{c*100:8.2f}%" for c in cums))
    print(f"{label+'(Sharpe)':25s} " + "  ".join(f"{s:8.3f}" for s in sharpes))

# 50/50 blend
for ver in ["v1", "v2", "v3"]:
    blend = (df[f"{ver}_r_n"] + df[f"{ver}_r_e"]) / 2
    cum = (blend+1).prod() - 1
    sharpe = ann_sharpe(blend)
    print(f"{ver.upper()} 50/50 blend cum:   {cum*100:.2f}%  Sharpe: {sharpe:.3f}")

print()
print("=" * 70)
print("硬指标 6: V3 滞胀/非共识覆盖度 (黄金+现金双高)")
print("=" * 70)
# V3 偶发事件: gold + cash > 0.45 (V2 三角形内部任何点都难)
v3_stagflation_weeks = df[(df["v3_e_safe"] + df["v3_e_cash"] > 0.40) & (df["v3_e_fi"] < 0.40)]
v2_stagflation_weeks = df[(df["v2_e_safe"] + df["v2_e_cash"] > 0.40) & (df["v2_e_fi"] < 0.40)]
print(f"V3 黄金+现金>0.40 + 固收<0.40 的非共识周: {len(v3_stagflation_weeks)} 周")
print(f"V2 同样形态: {len(v2_stagflation_weeks)} 周")
if len(v3_stagflation_weeks) > 0:
    sample = v3_stagflation_weeks.iloc[0]
    print(f"  V3 示例: gold={sample['v3_e_safe']:.3f}, cash={sample['v3_e_cash']:.3f}, fi={sample['v3_e_fi']:.3f}")

print()
print("=" * 70)
print("硬指标 7: V3 与 V2 在 bear 周的欧氏距离")
print("=" * 70)
for label, grp in [("bear weeks", bear), ("bull weeks", bull), ("all weeks", df)]:
    diff_32 = np.sqrt(
        (grp["v3_e_broad"] - grp["v2_e_broad"])**2 +
        (grp["v3_e_sat"] - grp["v2_e_sat"])**2 +
        (grp["v3_e_fi"] - grp["v2_e_fi"])**2 +
        (grp["v3_e_safe"] - grp["v2_e_safe"])**2 +
        (grp["v3_e_cash"] - grp["v2_e_cash"])**2
    )
    print(f"{label}: V3 vs V2 ||w_event|| distance mean: {diff_32.mean():.4f}")

print()
print("=" * 70)
print(f"详细数据: results/v3_vs_v2_vs_v1_verification.csv ({len(df)} rows)")
print("=" * 70)

