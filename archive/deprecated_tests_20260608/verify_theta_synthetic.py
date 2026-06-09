"""Offline (no ClickHouse) theta-ppo verification.

Runs WFO in synthetic-NAV-only mode (no GeneralBacktest at the end).
For relative reward-tuning comparison: same data, same checkpoint format,
same Sharpe formula. Sharpe absolute value is NOT comparable to dashboard
because synthetic NAV uses z-scored features (not real returns).

Use only for A/B testing reward variants before committing to 200k retrain.

Usage:
    python scripts/verify_theta_synthetic.py --stage 7b
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path("D:/素材/AE_LLM_RL_FOF-main")
sys.path.insert(0, str(PROJECT_ROOT))


def main(stage_label: str):
    """Print synthetic Sharpe + theta distribution for the most recent WFO in --output-dir."""
    output_dir = PROJECT_ROOT / f"results/wfo/{stage_label}_theta"
    run_dirs = sorted(output_dir.iterdir()) if output_dir.exists() else []
    if not run_dirs:
        print(f"  No WFO output found in {output_dir}")
        return
    run = run_dirs[-1]

    # Use the same 52-week annualization (correct for weekly data)
    df = pd.read_csv(run / "gate_diagnostics.csv")
    nav_path = run / "nav_series.csv"
    if not nav_path.exists():
        print(f"  No nav_series.csv in {run}")
        return
    nav = pd.read_csv(nav_path)["NAV"]
    rets = nav.pct_change().dropna()
    total = (nav.iloc[-1] / nav.iloc[0]) - 1
    n_yrs = len(rets) / 52
    ann_ret = (1 + total) ** (1 / n_yrs) - 1
    ann_vol = rets.std() * np.sqrt(52)
    sharpe = rets.mean() * 52 / ann_vol if ann_vol > 0 else 0
    mdd = ((nav / nav.cummax()) - 1).min()

    theta = df["theta"].values
    print(f"Stage {stage_label} (synthetic NAV, weekly 52x):")
    print(f"  Sharpe={sharpe:+.3f}  total={total*100:+.2f}%  ann_ret={ann_ret*100:+.2f}%  ann_vol={ann_vol*100:.2f}%  mdd={mdd*100:.2f}%")
    print(f"  theta: mean={theta.mean():.3f}, std={theta.std():.3f}, "
          f"[0.3,1.0]={((theta>=0.3)&(theta<=1.0)).mean()*100:.1f}%, "
          f">1.0={(theta>1.0).mean()*100:.1f}%, "
          f"<0.3={(theta<0.3).mean()*100:.1f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, help="Stage label (e.g. 7b, 7a3_200k)")
    args = parser.parse_args()
    main(args.stage)
