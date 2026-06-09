"""Backfill metrics_real.json for existing WFO runs using GeneralBacktest.

The old WFO runs only wrote metrics.json (computed from synthetic z-scored NAV)
which gave Sharpe ≈ 0.65 — but dashboard.png shows real Sharpe ≈ 1.09 from
GeneralBacktest on actual OHLC prices. This script re-runs GeneralBacktest on
existing weights_data_generalbt.csv files and writes metrics_real.json with
real-OHLC-based metrics for every Stage 4/5/6 run.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from GeneralBacktest import GeneralBacktest
from src.data_pipeline.track_b.fetcher import fetch_track_b

sys.stdout.reconfigure(encoding="utf-8")

ASSET_CODES = [
    "000300.SH",
    "000852.SH",
    "CBA02701.CS",
    "AU9999.SGE",
    "NH0100.NHF",
]

KEY_MAP = {
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

RUNS = [
    "results/wfo/stage5_v3/20260607_163404",
    "results/wfo/stage5_v3_1/20260607_163543",
    "results/wfo/stage6_v3/20260607_165431",
    "results/wfo/stage6_v3_1/20260607_165549",
]


def backfill_one(run_dir: Path) -> dict | None:
    w_path = run_dir / "weights_data_generalbt.csv"
    if not w_path.exists():
        print(f"  skip {run_dir.name}: no weights_data_generalbt.csv")
        return None

    w_df = pd.read_csv(w_path)
    start_date = w_df["date"].min()
    end_date = w_df["date"].max()
    print(f"  {run_dir.parent.name}/{run_dir.name}: {start_date} -> {end_date}")

    price_raw = fetch_track_b(
        start_date=start_date,
        end_date=end_date,
        columns=["open", "close", "adj_factor"],
    )
    records = []
    for date_str, row in price_raw.iterrows():
        for code in ASSET_CODES:
            o = row.get(f"{code}__open")
            c = row.get(f"{code}__close")
            adj = row.get(f"{code}__adj_factor", 1.0)
            if pd.isna(c) and pd.isna(o):
                continue
            close_val = float(c) if not pd.isna(c) else float(o)
            open_val = float(o) if not pd.isna(o) else close_val
            records.append({
                "date": date_str.strftime("%Y-%m-%d"),
                "code": code,
                "open": open_val,
                "close": close_val,
                "adj_factor": float(adj) if not pd.isna(adj) else 1.0,
            })
    price_data = pd.DataFrame(records)

    bt = GeneralBacktest(start_date=start_date, end_date=end_date)
    bt.run_backtest(
        weights_data=w_df,
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

    metrics_real = {}
    for cn, en in KEY_MAP.items():
        if cn in bt.metrics:
            v = bt.metrics[cn]
            try:
                metrics_real[en] = float(v)
            except (TypeError, ValueError):
                metrics_real[en] = str(v)
    metrics_real["n_weeks"] = w_df["date"].nunique()
    metrics_real["_source"] = "generalbacktest_real_ohlc"

    out_path = run_dir / "metrics_real.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics_real, f, indent=2, ensure_ascii=False)
    print(f"     -> {out_path.name}: Sharpe={metrics_real['sharpe_ratio']:.3f}, "
          f"Total={metrics_real['total_return']*100:.2f}%, "
          f"MDD={metrics_real['max_drawdown']*100:.2f}%")
    return metrics_real


def main():
    print("Backfilling metrics_real.json for existing WFO runs...")
    for run in RUNS:
        run_dir = PROJECT_ROOT / run
        try:
            backfill_one(run_dir)
        except Exception as e:
            print(f"  FAILED {run}: {e}")
    print("Done.")


if __name__ == "__main__":
    main()
