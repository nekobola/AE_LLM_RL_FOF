"""Stage 5 report: V3 vs V3.1 5-year (2021-2026, 278 weeks) WFO comparison.

Uses metrics_real.json (GeneralBacktest on real OHLC prices) instead of the
old metrics.json (synthetic NAV from z-scored features, with broken 252
annualization). Old thresholds (Sharpe >= 0.85) were calibrated against the
broken synthetic numbers; real-OHLC Sharpes are ~0.46 (V3) and ~1.10 (V3.1).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("D:/素材/AE_LLM_RL_FOF-main")

V3_DIR = PROJECT_ROOT / "results/wfo/stage5_v3/20260607_163404"
V31_DIR = PROJECT_ROOT / "results/wfo/stage5_v3_1/20260607_163543"


def load_run(run_dir: Path) -> dict:
    real_metrics_path = run_dir / "metrics_real.json"
    if not real_metrics_path.exists():
        raise FileNotFoundError(
            f"{real_metrics_path} missing — run scripts/backfill_metrics_real.py first"
        )
    metrics = json.load(open(real_metrics_path, encoding="utf-8"))
    gate = pd.read_csv(run_dir / "gate_diagnostics.csv")
    return {"metrics": metrics, "gate": gate}


v3 = load_run(V3_DIR)
v31 = load_run(V31_DIR)
m3 = v3["metrics"]
m31 = v31["metrics"]


print("=" * 78)
print("STAGE 5: V3 (PRODUCTION) vs V3.1 (EXPERIMENTAL) - REAL OHLC BACKTEST")
print("=" * 78)
print()
print(f"  {'Metric':30s}  {'V3 (prod)':>12s}  {'V3.1 (exp)':>12s}  {'Delta':>10s}")
print("  " + "-" * 70)

rows = [
    ("Weeks", "n_weeks", "{:.0f}"),
    ("Total return", "total_return", "{:+.2%}"),
    ("Annualized return", "annualized_return", "{:+.2%}"),
    ("Sharpe ratio", "sharpe_ratio", "{:.3f}"),
    ("Sortino ratio", "sortino_ratio", "{:.3f}"),
    ("Calmar ratio", "calmar_ratio", "{:.3f}"),
    ("Max drawdown", "max_drawdown", "{:.2%}"),
    ("Volatility (annual)", "volatility_annual", "{:.2%}"),
    ("Win rate", "win_rate", "{:.2%}"),
    ("Avg turnover", "avg_turnover", "{:.2%}"),
]

for label, key, fmt in rows:
    v3v = m3.get(key, 0.0)
    v31v = m31.get(key, 0.0)
    v3s = fmt.format(v3v)
    v31s = fmt.format(v31v)
    try:
        delta = fmt.format(v31v - v3v)
    except (ValueError, TypeError):
        delta = "-"
    print(f"  {label:30s}  {v3s:>12s}  {v31s:>12s}  {delta:>10s}")


print()
print("=" * 78)
print("ALPHA / TAU DYNAMICS (post Stage 1 ML fix)")
print("=" * 78)

for label, run in [("V3", v3), ("V3.1", v31)]:
    g = run["gate"]
    a = g["alpha"].values
    t = g["tau"].values
    print(f"\n  {label}:")
    print(f"    alpha: mean={a.mean():.3f}, std={a.std():.3f}, "
          f"min={a.min():.3f}, max={a.max():.3f}")
    print(f"    tau:   mean={t.mean():.2f}, std={t.std():.2f}, "
          f"min={t.min():.2f}, max={t.max():.2f}")


print()
print("=" * 78)
print("EDGE & FUSION VALUE (PPO meta-controller value)")
print("=" * 78)

for label, run in [("V3", v3), ("V3.1", v31)]:
    g = run["gate"]
    edge = g["edge_event_minus_normal"].values
    edge_std = float(np.std(edge) * 100)
    edge_mean = float(np.mean(edge) * 100)
    print(f"\n  {label}:")
    print(f"    edge std (PPO fusion value):  {edge_std:.4f}%  "
          f"{'[OK]' if edge_std >= 0.30 else '[WEAK]'}")
    print(f"    edge mean:                    {edge_mean:+.4f}%")


print()
print("=" * 78)
print("BEAR REGIME (event_stress) WEIGHT PATTERNS")
print("=" * 78)

for label, run in [("V3", v3), ("V3.1", v31)]:
    g = run["gate"]
    bear = g[g["regime_label"] == "event_stress"]
    if len(bear) == 0:
        print(f"\n  {label}: no bear weeks in sample")
        continue
    print(f"\n  {label}  (n_bear={len(bear)}):")
    print(f"    w_event_broad  = {bear['w_event_broad'].mean():.3f}")
    print(f"    w_event_sat    = {bear['w_event_satellite'].mean():.3f}")
    print(f"    w_event_fi     = {bear['w_event_fi'].mean():.3f}")
    print(f"    w_event_gold   = {bear['w_event_safe'].mean():.3f}")
    print(f"    w_event_cash   = {bear['w_event_cash'].mean():.3f}")
    gf = bear['w_event_safe'].mean() / max(bear['w_event_fi'].mean(), 1e-6)
    print(f"    gold/fi        = {gf:.2f}  "
          f"{'[GOLD DOMINATES]' if gf > 1.0 else '[GOLD TRAGEDY]'}")


print()
print("=" * 78)
print("STAGE 5 GO/NO-GO DECISION (real-OHLC thresholds)")
print("=" * 78)

checks = {
    "V3.1 Sharpe >= 1.0 (real-OHLC target)": m31["sharpe_ratio"] >= 1.0,
    "V3.1 Sharpe > V3 Sharpe (V3.1 wins)":
        m31["sharpe_ratio"] > m3["sharpe_ratio"],
    "V3.1 total return > V3 total return":
        m31["total_return"] > m3["total_return"],
    "V3.1 max drawdown < V3 max drawdown (better risk)":
        m31["max_drawdown"] < m3["max_drawdown"],
    "V3.1 Calmar ratio > V3 Calmar ratio":
        m31["calmar_ratio"] > m3["calmar_ratio"],
    "V3.1 bear gold/fi > 1.0 (gold tragedy fixed)":
        (v31["gate"][v31["gate"]["regime_label"] == "event_stress"]["w_event_safe"].mean() /
         max(v31["gate"][v31["gate"]["regime_label"] == "event_stress"]["w_event_fi"].mean(), 1e-6)) > 1.0,
}

all_pass = True
for name, ok in checks.items():
    mark = "[PASS]" if ok else "[FAIL]"
    print(f"  {mark} {name}")
    if not ok:
        all_pass = False

print()
if all_pass:
    print("STAGE 5 RESULT: GO. Switch main line to V3.1.")
else:
    print("STAGE 5 RESULT: NO-GO. Keep V3 as production; V3.1 stays experimental.")

print()
print("Detailed data:")
print(f"  V3  : {V3_DIR}/metrics_real.json")
print(f"  V3.1: {V31_DIR}/metrics_real.json")
