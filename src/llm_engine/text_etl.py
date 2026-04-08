"""Text ETL — extract text data from quantchdb within strict time windows.

Time windows enforced:
  - [t-30, t]  for concept-matched govcn (broad search)
  - [t-7,  t]  for CSRC, govcn global, and news titles
"""

from __future__ import annotations

from typing import Optional

from quantchdb import ClickHouseDatabase


class TextETL:
    """Extract macro text data from quantchdb, strictly within [t-30,t] / [t-7,t] windows."""

    def __init__(self, config: dict) -> None:
        self.config = config
        db_cfg = config["data_pipeline"]["track_b"]["db_config"]
        self.db = ClickHouseDatabase(config=db_cfg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_all(self, t: str) -> dict:
        """Extract all text data for date t.

        Returns
        -------
        dict
            Keys: zgrmyh, csrc_titles, govcn_global
        """
        return {
            "zgrmyh": self.fetch_zgrmyh(t),
            "csrc_titles": self.fetch_csrc_titles(t),
            "govcn_global": self.fetch_govcn_global(t),
        }

    # ------------------------------------------------------------------
    # Individual fetchers
    # ------------------------------------------------------------------

    def fetch_zgrmyh(self, t: str) -> list[dict]:
        """Return the most recent PBoC MPC meeting record with date <= t."""
        sql = (
            f"SELECT uuid, title, date, date_time, url, content "
            f"FROM text_db.zgrmyh "
            f"WHERE date <= '{t}' "
            f"ORDER BY date DESC LIMIT 1"
        )
        return self.db.fetch(sql).to_dicts()

    def fetch_csrc_titles(self, t: str) -> list[str]:
        """CSRC titles published within [t-7, t]."""
        sql = (
            f"SELECT title "
            f"FROM text_db.csrc "
            f"WHERE date BETWEEN dateSub(7, DAY, '{t}') AND '{t}' "
            f"ORDER BY date DESC"
        )
        result = self.db.fetch(sql)
        return result["title"].tolist() if not result.empty else []

    def fetch_govcn_global(self, t: str) -> list[dict]:
        """Global (industry_name = '') govcn policies within [t-7, t]."""
        sql = (
            f"SELECT title, content, date, passage_type "
            f"FROM text_db.govcn "
            f"WHERE date BETWEEN dateSub(7, DAY, '{t}') AND '{t}' "
            f"AND industry_name = '' "
            f"ORDER BY date DESC"
        )
        return self.db.fetch(sql).to_dicts()

    def fetch_govcn_by_concept(self, concept: str, t: str, lookback: int = 30) -> list[dict]:
        """Fuzzy-match govcn by concept within [t-lookback, t]."""
        sql = (
            f"SELECT title, content, date, passage_type "
            f"FROM text_db.govcn "
            f"WHERE date BETWEEN dateSub({lookback}, DAY, '{t}') AND '{t}' "
            f"AND (title LIKE '%{concept}%' OR content LIKE '%{concept}%') "
            f"ORDER BY date DESC"
        )
        return self.db.fetch(sql).to_dicts()

    def fetch_news_titles(
        self,
        source: str,
        t: str,
        concept: Optional[str] = None,
        limit: int = 20,
    ) -> list[str]:
        """Fetch up to `limit` news titles from eastmoney/sina within [t-7, t].

        Parameters
        ----------
        source : str
            "eastmoney" or "sina"
        t : str
            Reference date in YYYY-MM-DD.
        concept : str, optional
            If given, only titles matching the concept are returned.
        limit : int, default 20
            Maximum number of titles to return (Top-20 truncation per spec).

        Returns
        -------
        list[str]
        """
        table = f"text_db.{source}"
        if concept:
            sql = (
                f"SELECT title, date "
                f"FROM {table} "
                f"WHERE date BETWEEN dateSub(7, DAY, '{t}') AND '{t}' "
                f"AND (title LIKE '%{concept}%' OR content LIKE '%{concept}%') "
                f"ORDER BY date DESC LIMIT {limit}"
            )
        else:
            sql = (
                f"SELECT title, date "
                f"FROM {table} "
                f"WHERE date BETWEEN dateSub(7, DAY, '{t}') AND '{t}' "
                f"ORDER BY date DESC LIMIT {limit}"
            )
        result = self.db.fetch(sql)
        return result["title"].tolist()[:limit] if not result.empty else []
