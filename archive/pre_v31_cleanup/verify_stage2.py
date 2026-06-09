"""Stage 2 硬指标验证:
- 重新跑 NormalTrack/EventTrack,使用相同 LLM 信号 + AE_error + returns
- 对比新的 (w_normal, w_event) vs 原 WFO 记录
- 验证 edge std, bear w_event_fi, etc.
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

# 从原 WFO 加载 weekly returns
# 没有现成的 returns_5d,只有 r_normal/r_event/r_fused 周收益
# 验证方式:用原 WFO 的 AE_error + LLM 信号,跑新的 dual_track,与原 WFO 对比
# 关键验证指标: edge std, bear 段 w_event_fi, bull 段 w_event_satellite 等

ASSET_CODES = ["000300.SH", "000852.SH", "CBA02701.CS", "AU9999.SGE", "NH0100.NHF"]

# 由于我们没有保存 returns_5d,使用 5 个资产的 proxy:用 LLM signals + AE_error + 模拟 returns
# 更稳妥的验证:直接调用 dual_track 用随机/固定 returns,看 regime 切换的"形态"是否正确

print("=" * 70)
print("Stage 2 硬指标验证")
print("=" * 70)

# ── 验证 1: 收集原 WFO 中所有 (ae_error, tau=20, llm_macro/sent/risk) 元组 ──
samples = []
for _, row in ORIG.iterrows():
    samples.append(dict(
        ae_error=row["ae_error"],
        tau=20.0,
        llm_macro=row["llm_macro"],
        llm_sentiment=row["llm_sentiment"],
        llm_risk=row["llm_risk"],
    ))

# 加载 features_master 获取 weekly returns
features_path = PROJECT_ROOT / "data/processed/features_master.parquet"
if not features_path.exists():
    print(f"⚠ features_master 不存在,使用模拟 returns 验证")
    features_df = None
    weekly_return_cols = None
else:
    features_df = pd.read_parquet(features_path)
    weekly_return_cols = [c for c in features_df.columns if "__weekly_return" in c]
    print(f"✓ features_master 加载: {features_df.shape}")

# 构造 returns_5d 字典(按周)
returns_5d_by_week = {}
if features_df is not None and weekly_return_cols:
    for week_end, grp in features_df.groupby(features_df.index + pd.offsets.Week(weekday=4)):
        if len(grp) >= 5:
            ret_5d = grp[weekly_return_cols].values[-5:].T / 100.0
        else:
            ret_5d = grp[weekly_return_cols].values.T / 100.0
        returns_5d_by_week[week_end] = ret_5d
    print(f"✓ returns_5d 预计算: {len(returns_5d_by_week)} 周")

# ── 验证 2: 用新 dual_track 跑一遍,记录 w_event_fi/satellite 等 ──
engine = DualTrackEngine()
results = []
for i, row in ORIG.iterrows():
    week_ts = row["date"]
    # 取这周的 returns
    if week_ts in returns_5d_by_week:
        rets = returns_5d_by_week[week_ts]
    else:
        # 用上一个可用的
        avail = sorted([k for k in returns_5d_by_week.keys() if k <= week_ts])
        rets = returns_5d_by_week[avail[-1]] if avail else np.random.randn(5, 5) * 0.01

    try:
        w_n, w_e = engine.compute(
            rets,
            llm_macro=row["llm_macro"],
            llm_sentiment=row["llm_sentiment"],
            llm_risk=row["llm_risk"],
            ae_error=row["ae_error"],
            tau=20.0,
        )
    except Exception as ex:
        print(f"  Week {i}: dual_track 失败: {ex}")
        w_n, w_e = np.array([0.2]*5), np.array([0.2]*5)

    results.append({
        "date": week_ts,
        "alpha": row["alpha"],
        "ae_error": row["ae_error"],
        "regime_label": row["regime_label"],
        "w_normal_broad": w_n[0],
        "w_normal_satellite": w_n[1],
        "w_normal_fi": w_n[2],
        "w_normal_safe": w_n[3],
        "w_normal_cash": w_n[4],
        "w_event_broad": w_e[0],
        "w_event_satellite": w_e[1],
        "w_event_fi": w_e[2],
        "w_event_safe": w_e[3],
        "w_event_cash": w_e[4],
    })

df = pd.DataFrame(results)

# ── 指标计算 ──
print()
print("=" * 70)
print("硬指标 1: edge 信号 (r_event - r_normal) std")
print("=" * 70)

# 用新权重 × 真实 weekly returns 计算 r_event, r_normal, edge
# 但更准确的 edge 应使用原 WFO 的"下一周 returns"乘以"本周权重" — 这需要知道当周 weights
# 由于 run_wfo 里 w_normal_t/w_event_t 在 t 周用,week_returns 也在 t 周取,实际 edge 应该用
# 同周的 weights × returns

# 构造 weekly_returns_lookup
weekly_returns = {}
if features_df is not None and weekly_return_cols:
    for date_idx in features_df.index:
        weekly_returns[date_idx] = features_df.loc[date_idx, weekly_return_cols].values / 100.0

# 用 closest available
def get_returns(date_ts):
    if date_ts in weekly_returns:
        return weekly_returns[date_ts]
    avail = sorted([k for k in weekly_returns.keys() if k <= date_ts])
    return weekly_returns[avail[-1]] if avail else np.zeros(5)

r_normal_list = []
r_event_list = []
edge_list = []
for _, r in df.iterrows():
    rets = get_returns(r["date"])
    w_n = np.array([r["w_normal_broad"], r["w_normal_satellite"], r["w_normal_fi"],
                    r["w_normal_safe"], r["w_normal_cash"]])
    w_e = np.array([r["w_event_broad"], r["w_event_satellite"], r["w_event_fi"],
                    r["w_event_safe"], r["w_event_cash"]])
    r_n = float(np.dot(w_n, rets))
    r_e = float(np.dot(w_e, rets))
    r_normal_list.append(r_n)
    r_event_list.append(r_e)
    edge_list.append(r_e - r_n)

df["r_normal"] = r_normal_list
df["r_event"] = r_event_list
df["edge"] = edge_list

print(f"edge std:  {np.std(edge_list)*100:.4f}% (目标 > 0.3%)")
print(f"edge mean: {np.mean(edge_list)*100:.4f}%")
print(f"edge min:  {np.min(edge_list)*100:.4f}%, max: {np.max(edge_list)*100:.4f}%")
edge_std_pct = np.std(edge_list) * 100
df["r_normal"] = r_normal_list
df["r_event"] = r_event_list
df["edge"] = edge_list
df["diff_norm"] = np.sqrt(
    (df["w_event_broad"] - df["w_normal_broad"])**2 +
    (df["w_event_satellite"] - df["w_normal_satellite"])**2 +
    (df["w_event_fi"] - df["w_normal_fi"])**2 +
    (df["w_event_safe"] - df["w_normal_safe"])**2 +
    (df["w_event_cash"] - df["w_normal_cash"])**2
)
if edge_std_pct > 0.3:
    print(f"✓ PASS: edge std {edge_std_pct:.4f}% > 0.3%")
else:
    print(f"✗ FAIL: edge std {edge_std_pct:.4f}% <= 0.3%")

print()
print("=" * 70)
print("硬指标 2: bear-regime 下 EventTrack 固收权重")
print("=" * 70)
bear = df[df["regime_label"] == "event_stress"]
bull = df[df["regime_label"] == "bull_normal"]
print(f"bear weeks: {len(bear)}, bull weeks: {len(bull)}")
print(f"bear w_event_fi mean:         {bear['w_event_fi'].mean():.3f} (目标 ≥ 0.35)")
print(f"bear w_event_satellite mean:  {bear['w_event_satellite'].mean():.3f} (目标 ≤ 0.10)")
print(f"bear defensive (fi+safe+cash): {(bear['w_event_fi']+bear['w_event_safe']+bear['w_event_cash']).mean():.3f}")
print(f"bull w_event_fi mean:         {bull['w_event_fi'].mean():.3f}")
print(f"bull w_event_satellite mean:  {bull['w_event_satellite'].mean():.3f}")
print(f"bull offensive (broad+sat):   {(bull['w_event_broad']+bull['w_event_satellite']).mean():.3f}")
if bear["w_event_fi"].mean() >= 0.35:
    print(f"✓ PASS: bear w_event_fi {bear['w_event_fi'].mean():.3f} >= 0.35")
else:
    print(f"✗ FAIL: bear w_event_fi {bear['w_event_fi'].mean():.3f} < 0.35")

print()
print("=" * 70)
print("硬指标 3: bear vs bull 进攻性对比")
print("=" * 70)
bull_offensive = (bull["w_event_broad"] + bull["w_event_satellite"]).mean()
bear_offensive = (bear["w_event_broad"] + bear["w_event_satellite"]).mean()
print(f"bull EventTrack offensive mean:  {bull_offensive:.3f}")
print(f"bear EventTrack offensive mean:  {bear_offensive:.3f}")
if bull_offensive > bear_offensive:
    print(f"✓ PASS: bull offensive > bear offensive (PPO 切换有梯度)")
else:
    print(f"✗ FAIL")

print()
print("=" * 70)
print("硬指标 4: 两条 track 在不同 regime 下的差异 (PPO fusion 价值)")
print("=" * 70)
df["diff_norm"] = np.sqrt(
    (df["w_event_broad"] - df["w_normal_broad"])**2 +
    (df["w_event_satellite"] - df["w_normal_satellite"])**2 +
    (df["w_event_fi"] - df["w_normal_fi"])**2 +
    (df["w_event_safe"] - df["w_normal_safe"])**2 +
    (df["w_event_cash"] - df["w_normal_cash"])**2
)
print(f"Overall ||w_event - w_normal|| mean: {df['diff_norm'].mean():.3f}")
print(f"  bear weeks:  {bear['diff_norm'].mean():.3f}")
print(f"  bull weeks:  {bull['diff_norm'].mean():.3f}")
print(f"  bull+strong growth signal weeks (llm_macro+senti > 130):")
strong_growth = df[(df["llm_macro"] > 65) & (df["llm_sentiment"] > 65) & (df["regime_label"] == "bull_normal")]
print(f"    count={len(strong_growth)}, ||diff|| mean: {strong_growth['diff_norm'].mean():.3f}")

print()
print("=" * 70)
print("硬指标 5: 累计收益对比")
print("=" * 70)
print(f"Pure NormalTrack (新 dual_track): {(df['r_normal']+1).prod()-1:.4f}")
print(f"Pure EventTrack  (新 dual_track): {(df['r_event']+1).prod()-1:.4f}")
print(f"50/50 blend (反事实):              {((df['r_normal']+df['r_event'])/2+1).prod()-1:.4f}")
print(f"Regime-conditional switch (反事实):  {(((df['regime_label']=='event_stress')*df['r_event'] + (df['regime_label']=='bull_normal')*df['r_normal'])+1).prod()-1:.4f}")

# Save results
df.to_csv(PROJECT_ROOT / "results/stage2_verification.csv", index=False)
print(f"\n详细结果已保存: results/stage2_verification.csv")
