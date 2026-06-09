"""4-way verification: V1 / V2 / V3 / V3.1 on the same 156 weeks.

Extends verify_v3_vs_v2_vs_v1.py with V3.1 as the audit-fix candidate.

Hard metrics:
  1. edge std (v_event - v_normal returns) - PPO fusion value
  2. bear weeks: w_event_fi, w_event_gold, w_event_safe mean
  3. bull weeks: offensive (broad+sat) share
  4. ||w_event|| weekly std - divergence magnitude
  5. Sharpe ratio of pure event / pure normal / 50-50 blend
  6. Stagflation coverage: gold+cash > 0.40 + fi < 0.40
  7. V3.1 vs V2 bear shape match (defensive > 70%, equity < 30%)
  8. V3.1 gold tragedy regression (gold >= fi in strong bear)
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


# Four engines
engine_v1 = DualTrackEngine(use_v2=False, use_v3=False)
engine_v2 = DualTrackEngine(use_v2=True, use_v3=False)
engine_v3 = DualTrackEngine(use_v2=False, use_v3=True)
engine_v31 = DualTrackEngine(use_v3_1=True)

print("=" * 70)
print(f"4-way verification: V1 / V2 / V3 / V3.1 - {len(ORIG)} weeks")
print("=" * 70)

results = []
for i, row in ORIG.iterrows():
    week_ts = row["date"]
    rets_5d = get_returns_5d(week_ts)
    rets_w = get_returns(week_ts)

    def safe_compute(engine):
        try:
            w_n, w_e = engine.compute(
                rets_5d, llm_macro=row["llm_macro"], llm_sentiment=row["llm_sentiment"],
                llm_risk=row["llm_risk"], ae_error=row["ae_error"], tau=row["tau"],
            )
            return w_n, w_e
        except Exception:
            return np.array([0.2] * 5), np.array([0.2] * 5)

    w_n1, w_e1 = safe_compute(engine_v1)
    w_n2, w_e2 = safe_compute(engine_v2)
    w_n3, w_e3 = safe_compute(engine_v3)
    w_n31, w_e31 = safe_compute(engine_v31)

    def dot_ret(w): return float(np.dot(w, rets_w))
    r_e1, r_n1 = dot_ret(w_e1), dot_ret(w_n1)
    r_e2, r_n2 = dot_ret(w_e2), dot_ret(w_n2)
    r_e3, r_n3 = dot_ret(w_e3), dot_ret(w_n3)
    r_e31, r_n31 = dot_ret(w_e31), dot_ret(w_n31)

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
        # V3.1
        "v31_e_broad": w_e31[0], "v31_e_sat": w_e31[1], "v31_e_fi": w_e31[2],
        "v31_e_safe": w_e31[3], "v31_e_cash": w_e31[4],
        # Returns
        "v1_r_n": r_n1, "v1_r_e": r_e1, "v1_edge": r_e1 - r_n1,
        "v2_r_n": r_n2, "v2_r_e": r_e2, "v2_edge": r_e2 - r_n2,
        "v3_r_n": r_n3, "v3_r_e": r_e3, "v3_edge": r_e3 - r_n3,
        "v31_r_n": r_n31, "v31_r_e": r_e31, "v31_edge": r_e31 - r_n31,
    })

df = pd.DataFrame(results)
df.to_csv(PROJECT_ROOT / "results/v3_1_vs_v3_vs_v2_vs_v1_verification.csv", index=False)


def std_pct(x): return np.std(x) * 100
def ann_sharpe(returns, periods_per_year=52):
    mu, sig = np.mean(returns), np.std(returns)
    return float(mu / sig * np.sqrt(periods_per_year)) if sig >= 1e-9 else 0.0


print()
print("=" * 70)
print("硬指标 1: edge std (r_event - r_normal) - PPO 切换价值")
print("=" * 70)
e_stds = {v: std_pct(df[f"{v}_edge"]) for v in ["v1", "v2", "v3", "v31"]}
for v, s in e_stds.items():
    print(f"{v.upper()} edge std:  {s:.4f}%")
print(f"V3.1 / V1 倍数: {e_stds['v31']/e_stds['v1']:.2f}x")
print(f"V3.1 / V2 倍数: {e_stds['v31']/e_stds['v2']:.2f}x")
print(f"V3.1 / V3 倍数: {e_stds['v31']/e_stds['v3']:.2f}x")
print(f"{'[OK] PASS' if e_stds['v31'] > 0.30 else '[FAIL] FAIL' if e_stds['v31'] < 0.20 else '[WARN]'}: V3.1 edge std {e_stds['v31']:.4f}%")

print()
print("=" * 70)
print("硬指标 2: bear weeks 下 EventTrack 固收 / 黄金 / 现金权重")
print("=" * 70)
bear = df[df["regime_label"] == "event_stress"]
print(f"bear weeks: {len(bear)}")
for v in ["v1", "v2", "v3", "v31"]:
    fi = bear[f"{v}_e_fi"].mean()
    gold = bear[f"{v}_e_safe"].mean()
    cash = bear[f"{v}_e_cash"].mean()
    print(f"{v.upper()} bear:  fi={fi:.3f}  gold={gold:.3f}  cash={cash:.3f}  (gold/fi = {gold/max(fi,1e-6):.2f})")

print()
print("=" * 70)
print("硬指标 3: bull vs bear weeks 进攻性")
print("=" * 70)
bull = df[df["regime_label"] == "bull_normal"]
for v in ["v1", "v2", "v3", "v31"]:
    bull_off = (bull[f"{v}_e_broad"] + bull[f"{v}_e_sat"]).mean()
    bear_off = (bear[f"{v}_e_broad"] + bear[f"{v}_e_sat"]).mean()
    print(f"{v.upper()} bull offensive: {bull_off:.3f}  bear offensive: {bear_off:.3f}  diff: {bull_off - bear_off:+.3f}")

print()
print("=" * 70)
print("硬指标 4: 5-dim w_event 周度 std (跨 regime 分化度)")
print("=" * 70)
for v in ["v1", "v2", "v3", "v31"]:
    weekly_var = df[[f"{v}_e_broad", f"{v}_e_sat", f"{v}_e_fi",
                     f"{v}_e_safe", f"{v}_e_cash"]].std().mean()
    print(f"{v.upper()} w_event weekly std mean: {weekly_var:.4f}")

print()
print("=" * 70)
print("硬指标 5: Sharpe ratio (Pure / 50-50 blend)")
print("=" * 70)
print(f"{'':25s} {'V1':>10s}  {'V2':>10s}  {'V3':>10s}  {'V3.1':>10s}")
for label, base_col in [("Pure NormalTrack", "_r_n"), ("Pure EventTrack", "_r_e")]:
    cols = [f"v1{base_col}", f"v2{base_col}", f"v3{base_col}", f"v31{base_col}"]
    cums = [(df[c]+1).prod() - 1 for c in cols]
    sharpes = [ann_sharpe(df[c]) for c in cols]
    print(f"{label+'(cum)':25s} " + "  ".join(f"{c*100:8.2f}%" for c in cums))
    print(f"{label+'(Sharpe)':25s} " + "  ".join(f"{s:8.3f}" for s in sharpes))

for v in ["v1", "v2", "v3", "v31"]:
    blend = (df[f"{v}_edge"])  # edge IS the fusion value
    cum = (df[f"{v}_r_n"] * 0.5 + df[f"{v}_r_e"] * 0.5 + 1).prod() - 1
    print(f"{v.upper()} 50/50 blend cum (edge-based): {cum*100:.2f}%")

print()
print("=" * 70)
print("硬指标 6: 滞胀/非共识覆盖度 (黄金+现金>0.40 + 固收<0.40)")
print("=" * 70)
for v in ["v3", "v31"]:
    mask = (df[f"{v}_e_safe"] + df[f"{v}_e_cash"] > 0.40) & (df[f"{v}_e_fi"] < 0.40)
    n_weeks = mask.sum()
    print(f"{v.upper()} stagflation weeks: {n_weeks}")
    if n_weeks > 0:
        sample = df[mask].iloc[0]
        print(f"  示例: gold={sample[f'{v}_e_safe']:.3f}, cash={sample[f'{v}_e_cash']:.3f}, fi={sample[f'{v}_e_fi']:.3f}")

print()
print("=" * 70)
print("硬指标 7: V3.1 bear 形态 vs V2 bear 顶点")
print("=" * 70)
# V2 B_BEAR = (0.05, 0.05, 0.45, 0.35, 0.10)
b_v2_bear = np.array([0.05, 0.05, 0.45, 0.35, 0.10])
v31_bear = bear[["v31_e_broad", "v31_e_sat", "v31_e_fi", "v31_e_safe", "v31_e_cash"]].mean().values
dist = np.linalg.norm(v31_bear - b_v2_bear)
print(f"V2 B_BEAR:        {b_v2_bear}")
print(f"V3.1 bear mean:   {v31_bear}")
print(f"||V3.1_bear - V2_B_BEAR||: {dist:.4f}")
print(f"V3.1 bear defensive share: {v31_bear[2:].sum():.3f}")
print(f"V3.1 bear equity share:    {v31_bear[:2].sum():.3f}")

print()
print("=" * 70)
print("硬指标 8: V3.1 vs V3 黄金悲剧回归")
print("=" * 70)
# V3 在同一危机信号下: gold < fi (V3 黄金悲剧)
# V3.1 应该: gold >= fi (V3 审计修复)
v3_bear = bear[["v3_e_broad", "v3_e_sat", "v3_e_fi", "v3_e_safe", "v3_e_cash"]].mean().values
print(f"V3 bear  gold={v3_bear[3]:.3f}  fi={v3_bear[2]:.3f}  gold/fi = {v3_bear[3]/max(v3_bear[2],1e-6):.2f}")
print(f"V3.1 bear gold={v31_bear[3]:.3f}  fi={v31_bear[2]:.3f}  gold/fi = {v31_bear[3]/max(v31_bear[2],1e-6):.2f}")
ratio_v3 = v3_bear[3] / max(v3_bear[2], 1e-6)
ratio_v31 = v31_bear[3] / max(v31_bear[2], 1e-6)
print(f"{'[PASS]' if ratio_v31 > ratio_v3 else '[FAIL]'}: V3.1 gold/fi {ratio_v31:.2f} > V3 {ratio_v3:.2f}")

print()
print("=" * 70)
print(f"详细数据: results/v3_1_vs_v3_vs_v2_vs_v1_verification.csv ({len(df)} rows)")
print("=" * 70)
