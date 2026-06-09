"""Stage 6 report: V3 vs V3.1 5-year WFO with tuned config.

Uses metrics_real.json (GeneralBacktest on real OHLC) instead of metrics.json
(synthetic NAV from z-scored features). Cross-stage comparison is now valid.

Stage 6 changes (vs Stage 5):
  - lambda_alpha_direct: 0.05 -> 0.15 (强化 regime-conditional alpha signal)
  - V3.1 THETA: 1.0 -> 0.7 (减 sharp, weekly std 回升)
  - PPO retrain: 5000 iter -> 2000 iter (降低计算成本)
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("D:/素材/AE_LLM_RL_FOF-main")

STAGE5_V3 = PROJECT_ROOT / "results/wfo/stage5_v3/20260607_163404"
STAGE5_V31 = PROJECT_ROOT / "results/wfo/stage5_v3_1/20260607_163543"
STAGE6_V3 = PROJECT_ROOT / "results/wfo/stage6_v3/20260607_165431"
STAGE6_V31 = PROJECT_ROOT / "results/wfo/stage6_v3_1/20260607_165549"


def load_run(run_dir: Path) -> dict:
    real_metrics_path = run_dir / "metrics_real.json"
    if not real_metrics_path.exists():
        raise FileNotFoundError(
            f"{real_metrics_path} missing — run scripts/backfill_metrics_real.py first"
        )
    metrics = json.load(open(real_metrics_path, encoding="utf-8"))
    gate = pd.read_csv(run_dir / "gate_diagnostics.csv")
    return {"metrics": metrics, "gate": gate}


runs = {
    "S5 V3 (prod)":   load_run(STAGE5_V3),
    "S5 V3.1 (exp)":  load_run(STAGE5_V31),
    "S6 V3 (prod)":   load_run(STAGE6_V3),
    "S6 V3.1 (exp)":  load_run(STAGE6_V31),
}


def summarize(label, run):
    m = run["metrics"]
    g = run["gate"]
    edge_std = float(np.std(g["edge_event_minus_normal"]) * 100)
    bear = g[g["regime_label"] == "event_stress"]
    if len(bear) > 0:
        gold_fi = (bear["w_event_safe"].mean() / max(bear["w_event_fi"].mean(), 1e-6))
        bear_gold = bear["w_event_safe"].mean()
        bear_fi = bear["w_event_fi"].mean()
    else:
        gold_fi = bear_gold = bear_fi = 0.0
    a = g["alpha"].values
    t = g["tau"].values
    return {
        "label": label,
        "sharpe": m["sharpe_ratio"],
        "total": m["total_return"],
        "annual": m.get("annualized_return", 0.0),
        "maxdd": m["max_drawdown"],
        "calmar": m.get("calmar_ratio", 0.0),
        "vol": m["volatility_annual"],
        "win_rate": m.get("win_rate", 0.0),
        "turnover": m.get("avg_turnover", 0.0),
        "edge_std": edge_std,
        "alpha_mean": a.mean(),
        "alpha_std": a.std(),
        "tau_mean": t.mean(),
        "tau_std": t.std(),
        "bear_gold": bear_gold,
        "bear_fi": bear_fi,
        "gold_fi": gold_fi,
    }


summaries = {k: summarize(k, v) for k, v in runs.items()}


print("=" * 96)
print("STAGE 6: V3 vs V3.1 5-YEAR WFO (REAL OHLC) — tuned config: λ_alpha=0.15, V3.1 θ=0.7")
print("=" * 96)
print()
hdr = f"  {'Metric':28s}  {'S5 V3':>10s}  {'S5 V3.1':>10s}  {'S6 V3':>10s}  {'S6 V3.1':>10s}"
print(hdr)
print("  " + "-" * 92)

s5v3, s5v31, s6v3, s6v31 = (
    summaries["S5 V3 (prod)"],
    summaries["S5 V3.1 (exp)"],
    summaries["S6 V3 (prod)"],
    summaries["S6 V3.1 (exp)"],
)


def fmt_row(label, formatter, *vals):
    s = "  " + f"{label:28s}"
    for v in vals:
        s += f"  {formatter.format(v):>10s}"
    print(s)


fmt_row("Sharpe ratio", "{:.3f}", s5v3["sharpe"], s5v31["sharpe"], s6v3["sharpe"], s6v31["sharpe"])
fmt_row("Annual return", "{:+.2%}", s5v3["annual"], s5v31["annual"], s6v3["annual"], s6v31["annual"])
fmt_row("Total return", "{:+.2%}", s5v3["total"], s5v31["total"], s6v3["total"], s6v31["total"])
fmt_row("Max drawdown", "{:.2%}", s5v3["maxdd"], s5v31["maxdd"], s6v3["maxdd"], s6v31["maxdd"])
fmt_row("Calmar ratio", "{:.3f}", s5v3["calmar"], s5v31["calmar"], s6v3["calmar"], s6v31["calmar"])
fmt_row("Volatility", "{:.2%}", s5v3["vol"], s5v31["vol"], s6v3["vol"], s6v31["vol"])
fmt_row("Win rate", "{:.2%}", s5v3["win_rate"], s5v31["win_rate"], s6v3["win_rate"], s6v31["win_rate"])
fmt_row("Avg turnover", "{:.2%}", s5v3["turnover"], s5v31["turnover"], s6v3["turnover"], s6v31["turnover"])
fmt_row("edge std (synth)", "{:.4f}%", s5v3["edge_std"], s5v31["edge_std"], s6v3["edge_std"], s6v31["edge_std"])
fmt_row("alpha mean", "{:.3f}", s5v3["alpha_mean"], s5v31["alpha_mean"], s6v3["alpha_mean"], s6v31["alpha_mean"])
fmt_row("alpha std", "{:.4f}", s5v3["alpha_std"], s5v31["alpha_std"], s6v3["alpha_std"], s6v31["alpha_std"])
fmt_row("tau mean", "{:.2f}", s5v3["tau_mean"], s5v31["tau_mean"], s6v3["tau_mean"], s6v31["tau_mean"])
fmt_row("tau std", "{:.2f}", s5v3["tau_std"], s5v31["tau_std"], s6v3["tau_std"], s6v31["tau_std"])
fmt_row("bear w_gold", "{:.3f}", s5v3["bear_gold"], s5v31["bear_gold"], s6v3["bear_gold"], s6v31["bear_gold"])
fmt_row("bear w_fi", "{:.3f}", s5v3["bear_fi"], s5v31["bear_fi"], s6v3["bear_fi"], s6v31["bear_fi"])
fmt_row("bear gold/fi", "{:.2f}", s5v3["gold_fi"], s5v31["gold_fi"], s6v3["gold_fi"], s6v31["gold_fi"])


print()
print("=" * 96)
print("CROSS-STAGE DELTAS")
print("=" * 96)

print()
print("  Within V3: Stage 6 vs Stage 5")
v3_delta = s6v3["sharpe"] - s5v3["sharpe"]
print(f"    Sharpe delta:    {v3_delta:+.3f}  (S5: {s5v3['sharpe']:.3f} -> S6: {s6v3['sharpe']:.3f})")
print(f"    Annual delta:    {(s6v3['annual']-s5v3['annual'])*100:+.2f}%")
print(f"    MaxDD delta:     {(s6v3['maxdd']-s5v3['maxdd'])*100:+.2f}%")

print()
print("  Within V3.1: Stage 6 vs Stage 5")
v31_delta = s6v31["sharpe"] - s5v31["sharpe"]
print(f"    Sharpe delta:    {v31_delta:+.3f}  (S5: {s5v31['sharpe']:.3f} -> S6: {s6v31['sharpe']:.3f})")
print(f"    Annual delta:    {(s6v31['annual']-s5v31['annual'])*100:+.2f}%")
print(f"    MaxDD delta:     {(s6v31['maxdd']-s5v31['maxdd'])*100:+.2f}%")


print()
print("=" * 96)
print("STAGE 6 GO/NO-GO (real-OHLC)")
print("=" * 96)

checks = {
    "V3.1 Sharpe >= 1.0 (real-OHLC target)": s6v31["sharpe"] >= 1.0,
    "V3.1 Sharpe > V3 Sharpe (V3.1 wins within Stage 6)": s6v31["sharpe"] > s6v3["sharpe"],
    "V3.1 max DD < V3 max DD (better risk in Stage 6)": s6v31["maxdd"] < s6v3["maxdd"],
    "V3.1 Calmar > V3 Calmar (better risk-adj return)": s6v31["calmar"] > s6v3["calmar"],
    "V3.1 total return > V3 total return": s6v31["total"] > s6v3["total"],
    "V3.1 bear gold/fi > 1.0 (gold tragedy fixed)": s6v31["gold_fi"] > 1.0,
    "V3.1 Stage 6 NOT regressed vs Stage 5 (Sharpe diff > -0.1)":
        (s6v31["sharpe"] - s5v31["sharpe"]) > -0.1,
}

for name, ok in checks.items():
    mark = "[PASS]" if ok else "[FAIL]"
    print(f"  {mark} {name}")

print()
print("STAGE 6 KEY INSIGHT:")
print(f"  Within Stage 6 (real OHLC):")
print(f"    V3:   Sharpe={s6v3['sharpe']:.3f}, AnnRet={s6v3['annual']*100:+.2f}%, MaxDD={s6v3['maxdd']*100:.2f}%")
print(f"    V3.1: Sharpe={s6v31['sharpe']:.3f}, AnnRet={s6v31['annual']*100:+.2f}%, MaxDD={s6v31['maxdd']*100:.2f}%")
print()
print(f"  V3.1 vs V3 in Stage 6 by real Sharpe: {s6v31['sharpe'] - s6v3['sharpe']:+.3f}")
print(f"  Stage 6 vs Stage 5 V3.1 change:        {s6v31['sharpe'] - s5v31['sharpe']:+.3f}  "
      f"({'IMPROVED' if s6v31['sharpe'] > s5v31['sharpe'] else 'REGRESSED'})")
print()
print("DECISION:")
if s6v31["sharpe"] > s6v3["sharpe"] and s6v31["sharpe"] >= 1.0:
    print("  STAGE 6 RESULT: GO. V3.1 is the production main line.")
else:
    print("  STAGE 6 RESULT: NO-GO. Keep V3 as production.")
print()
print(f"Detailed data:")
print(f"  Stage 6 V3:   {STAGE6_V3}/metrics_real.json")
print(f"  Stage 6 V3.1: {STAGE6_V31}/metrics_real.json")
