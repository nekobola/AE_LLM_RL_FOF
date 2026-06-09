"""ClickHouse-based quantitative hard filters: liquidity veto and momentum tiebreaker."""

from typing import Optional

import pandas as pd

from quantchdb import ClickHouseDatabase


class ETFSelector:
    """Local ClickHouse quantitative hard defense line.

    Applies:
    1. Liquidity veto: 5-day average amount must exceed threshold
    2. Momentum tiebreaker: 20-day return ranking when scores tie
    """

    def __init__(self, config: dict):
        self.config = config
        db_cfg = config["data_pipeline"]["track_b"]["db_config"]
        self.db = ClickHouseDatabase(config=db_cfg)

    def liquidity_veto(
        self,
        codes: list[str],
        t: str,
        min_amt: int = 30_000_000,
    ) -> list[str]:
        """Return codes passing 5-day avg amount >= min_amt.

        Strictly historical window: [t-5, t] (t is latest date in window).
        """
        if not codes:
            return []
        codes_str = ",".join(f"'{c}'" for c in codes)
        sql = f"""
        SELECT
            code,
            SUM(amount) / 5 AS avg_amt
        FROM etf.etf_daily
        WHERE date BETWEEN dateSub(5, DAY, '{t}') AND '{t}'
          AND code IN ({codes_str})
        GROUP BY code
        HAVING avg_amt >= {min_amt}
        """
        df: pd.DataFrame = self.db.fetch(sql)
        return df["code"].tolist() if not df.empty else []

    def tiebreaker_momentum(
        self,
        codes: list[str],
        t: str,
        top_n: int = 1,
        window: int = 20,
    ) -> list[str]:
        """Return top-N codes by 20-day momentum (strictly historical).

        Window: [t-25, t] to compute 20-day return, QUALIFY picks latest date.
        """
        if not codes:
            return []
        codes_str = ",".join(f"'{c}'" for c in codes)
        sql = f"""
        SELECT
            code,
            (close - lagInFrame(close, {window}) OVER (
                PARTITION BY code ORDER BY date
            )) / lagInFrame(close, {window}) OVER (
                PARTITION BY code ORDER BY date
            ) AS momentum_{window}d
        FROM etf.etf_daily
        WHERE code IN ({codes_str})
          AND date BETWEEN dateSub({window + 5}, DAY, '{t}') AND '{t}'
        QUALIFY ROW_NUMBER() OVER (PARTITION BY code ORDER BY date DESC) = 1
        ORDER BY momentum_{window}d DESC
        LIMIT {top_n}
        """
        df: pd.DataFrame = self.db.fetch(sql)
        return df["code"].tolist() if not df.empty else []
