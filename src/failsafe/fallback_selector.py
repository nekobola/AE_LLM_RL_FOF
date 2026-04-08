"""
LLM宕机时的备选选基逻辑（基于动量）。

触发条件：API超时/抛错/JSON缺失字段，重试max_retries次均失败。

选基规则（纯本地SQL，无需LLM）：
- 宽基：取20日动量Top1
- 卫星：取20日动量Top3（不同行业）
- 固收：按20日波动率升序取Top2（利率债+信用债）
- 避险：固定518880.SH（黄金ETF）
- 现金：固定511850.SH（货币ETF）
"""

from quantchdb import ClickHouseDatabase


class FallbackSelector:
    """
    LLM宕机时的备选选基逻辑（基于动量）。
    """

    def __init__(self, config: dict):
        self.config = config
        self.db = ClickHouseDatabase(config=config["data_pipeline"]["track_b"]["db_config"])
        self._etf_pools = config["data_pipeline"]["etf_pools"]
        self._momentum_window = config["selection"]["momentum_window"]

    def select_8(self, current_date: str) -> dict[str, str]:
        """
        返回 {slot_name: etf_code}，共8只ETF。
        全部基于本地quantchdb SQL，不依赖LLM。
        """
        result = {}

        # ---- 宽基：20日动量Top1 ----
        result["wide_base"] = self._select_wide_base(current_date)

        # ---- 卫星：20日动量Top3（不同行业）----
        result["satellite"] = self._select_satellite(current_date)

        # ---- 固收：波动率升序Top2 ----
        result["fixed_income"] = self._select_fixed_income(current_date)

        # ---- 避险：固定黄金ETF ----
        result["safe_haven"] = "518880.SH"

        # ---- 现金：固定货币ETF ----
        result["cash"] = "511850.SH"

        return result

    def _select_wide_base(self, current_date: str) -> str:
        """取20日动量Top1宽基ETF"""
        sql = f"""
        WITH price_series AS (
            SELECT
                code,
                close,
                row_number() OVER (PARTITION BY code ORDER BY trade_date DESC) AS rn
            FROM oafmd
            WHERE trade_date <= '{current_date}'
              AND code IN ({','.join(repr(c) for c in self._etf_pools["wide_base"])})
        ),
        recent_20 AS (
            SELECT code, close
            FROM price_series
            WHERE rn <= {self._momentum_window}
        ),
        oldest AS (
            SELECT code, close AS price_old
            FROM price_series
            WHERE rn = {self._momentum_window}
        ),
        newest AS (
            SELECT code, close AS price_new
            FROM price_series
            WHERE rn = 1
        ),
        momentum AS (
            SELECT
                n.code,
                (n.price_new - o.price_old) / o.price_old AS mom
            FROM newest n
            JOIN oldest o ON n.code = o.code
        )
        SELECT code
        FROM momentum
        ORDER BY mom DESC
        LIMIT 1
        """
        rows = self.db.query(sql)
        return rows[0][0] if rows else self._etf_pools["wide_base"][0]

    def _select_satellite(self, current_date: str) -> list[str]:
        """取20日动量Top3卫星ETF（不同行业，自动去重）"""
        satellite_codes = list(self._etf_pools["satellite"])
        codes_str = ",".join(repr(c) for c in satellite_codes)

        sql = f"""
        WITH price_series AS (
            SELECT
                code,
                trade_date,
                close,
                row_number() OVER (PARTITION BY code ORDER BY trade_date DESC) AS rn
            FROM oafmd
            WHERE trade_date <= '{current_date}'
              AND code IN ({codes_str})
        ),
        recent_20 AS (
            SELECT code, close
            FROM price_series
            WHERE rn <= {self._momentum_window}
        ),
        oldest AS (
            SELECT code, close AS price_old
            FROM price_series
            WHERE rn = {self._momentum_window}
        ),
        newest AS (
            SELECT code, close AS price_new
            FROM price_series
            WHERE rn = 1
        ),
        momentum AS (
            SELECT
                n.code,
                (n.price_new - o.price_old) / o.price_old AS mom
            FROM newest n
            JOIN oldest o ON n.code = o.code
        )
        SELECT code
        FROM momentum
        ORDER BY mom DESC
        LIMIT 3
        """
        rows = self.db.query(sql)
        return [r[0] for r in rows] if rows else satellite_codes[:3]

    def _select_fixed_income(self, current_date: str) -> list[str]:
        """按20日波动率升序取Top2（利率债+信用债）"""
        fi_codes = self._etf_pools["fixed_income"]
        codes_str = ",".join(repr(c) for c in fi_codes)

        sql = f"""
        WITH price_series AS (
            SELECT
                code,
                trade_date,
                close,
                row_number() OVER (PARTITION BY code ORDER BY trade_date DESC) AS rn
            FROM oafmd
            WHERE trade_date <= '{current_date}'
              AND code IN ({codes_str})
        ),
        recent_20 AS (
            SELECT code, close
            FROM price_series
            WHERE rn <= {self._momentum_window}
        ),
        daily_returns AS (
            SELECT
                code,
                (close - lag(close) OVER (PARTITION BY code ORDER BY trade_date)) / lag(close) OVER (PARTITION BY code ORDER BY trade_date) AS ret
            FROM recent_20
            QUALIFY row_number() OVER (PARTITION BY code ORDER BY trade_date DESC) > 1
        ),
        volatility AS (
            SELECT
                code,
                stddevPop(ret) AS vol_20d
            FROM daily_returns
            GROUP BY code
        )
        SELECT code
        FROM volatility
        ORDER BY vol_20d ASC
        LIMIT 2
        """
        rows = self.db.query(sql)
        return [r[0] for r in rows] if rows else fi_codes[:2]
