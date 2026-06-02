"""Text ETL — extract macro text data from ClickHouse text_db within strict time windows.

Graceful degradation: if tables do not exist, return empty lists.

Time windows enforced:
  - [t-30, t]  for concept-matched govcn (broad search)
  - [t-7,  t]  for CSRC, govcn global, and news titles
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Any, Optional

from clickhouse_driver import Client

logger = logging.getLogger(__name__)


def _date_range(t: str, lookback: int) -> tuple[str, str]:
    """Return (t_minus_lookback, t) as YYYY-MM-DD strings."""
    t_dt = datetime.strptime(t, "%Y-%m-%d")
    t_minus = (t_dt - timedelta(days=lookback)).strftime("%Y-%m-%d")
    return t_minus, t


class TextETL:
    """Extract macro text data from ClickHouse text_db, strictly within [t-30,t] / [t-7,t] windows."""

    def __init__(self, config: dict) -> None:
        self.config = config
        ch_cfg = config.get("text_db", {})
        self.db_config = {
            "host": ch_cfg.get("host", os.environ.get("CHDB_HOST", "10.13.66.5")),
            "port": int(ch_cfg.get("port", os.environ.get("CHDB_PORT", 20108))),
            "user": ch_cfg.get("user", os.environ.get("CHDB_USER", "hqy_404")),
            "password": ch_cfg.get("password", os.environ.get("CHDB_PASSWORD", "hqy_404")),
            "database": ch_cfg.get("database", os.environ.get("CHDB_DATABASE", "text_db")),
        }
        self._client: Any = None

    def _get_client(self) -> Optional[Client]:
        try:
            if self._client is None:
                self._client = Client(**self.db_config)
            return self._client
        except Exception as e:
            logger.warning(f"[TextETL] ClickHouse connect failed: {e}")
            return None

    def _fetch(self, sql: str) -> list[dict]:
        """Execute SQL and return list of dict rows. Returns [] on error."""
        try:
            client = self._get_client()
            if client is None:
                return []
            result = client.execute(sql, with_column_types=True)
            rows, columns = result[0], [c[0] for c in result[1]]
            return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.warning(f"[TextETL] SQL error: {e} | SQL: {sql[:80]}")
            return []

    def _fetch_col(self, sql: str, col: str = "title") -> list:
        """Execute SQL and return a single column as list. Returns [] on error."""
        rows = self._fetch(sql)
        return [row.get(col) for row in rows if col in row] if rows else []

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
        sql = f"""
            SELECT uuid, title, date, date_time, url, content
            FROM zgrmyh
            WHERE date <= toDate('{t}')
            ORDER BY date DESC
            LIMIT 1
        """
        return self._fetch(sql)

    def fetch_csrc_titles(self, t: str) -> list[str]:
        """CSRC titles published within [t-7, t]."""
        t_start, t_end = _date_range(t, 7)
        sql = f"""
            SELECT title
            FROM csrc
            WHERE date BETWEEN toDate('{t_start}') AND toDate('{t_end}')
            ORDER BY date DESC
        """
        return self._fetch_col(sql, col="title")

    def fetch_govcn_global(self, t: str) -> list[dict]:
        """Global (industry_name = '') govcn policies within [t-7, t]."""
        t_start, t_end = _date_range(t, 7)
        sql = f"""
            SELECT title, content, date, passage_type
            FROM govcn
            WHERE date BETWEEN toDate('{t_start}') AND toDate('{t_end}')
              AND industry_name = ''
            ORDER BY date DESC
        """
        return self._fetch(sql)

    def fetch_govcn_by_concept(self, concept: str, t: str, lookback: int = 30) -> list[dict]:
        """Fuzzy-match govcn by concept within [t-lookback, t]."""
        t_start, t_end = _date_range(t, lookback)
        like_pat = f"%{concept}%"
        sql = f"""
            SELECT title, content, date, passage_type
            FROM govcn
            WHERE date BETWEEN toDate('{t_start}') AND toDate('{t_end}')
              AND (title LIKE '{like_pat}' OR content LIKE '{like_pat}')
            ORDER BY date DESC
        """
        return self._fetch(sql)

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
        t_start, t_end = _date_range(t, 7)
        if concept:
            like_pat = f"%{concept}%"
            sql = f"""
                SELECT title
                FROM {source}
                WHERE date BETWEEN toDate('{t_start}') AND toDate('{t_end}')
                  AND (title LIKE '{like_pat}' OR content LIKE '{like_pat}')
                ORDER BY date DESC
                LIMIT {limit}
            """
        else:
            sql = f"""
                SELECT title
                FROM {source}
                WHERE date BETWEEN toDate('{t_start}') AND toDate('{t_end}')
                ORDER BY date DESC
                LIMIT {limit}
            """
        return self._fetch_col(sql, col="title")

    def extract_per_concept(
        self,
        t: str,
        concepts: list[str],
        lookback: int = 30,
    ) -> dict:
        """Extract per-concept text data for themed prompt generation.

        Returns
        -------
        dict
            {
                "shared": {"mpc": {...}, "csrc": [...]},
                "concepts": {
                    concept_name: {
                        "govcn": [...],   # fetch_govcn_by_concept results
                        "news": [...],     # combined eastmoney + sina titles
                    }
                }
            }
        """
        # Shared data: MPC + CSRC
        mpc = self.fetch_zgrmyh(t)
        csrc = self.fetch_csrc_titles(t)

        # Per-concept data
        concept_data: dict[str, dict] = {}
        for concept in concepts:
            govcn = self.fetch_govcn_by_concept(concept, t, lookback=lookback)
            # Combine eastmoney + sina news for this concept
            em_news = self.fetch_news_titles("eastmoney", t, concept=concept, limit=10)
            sina_news = self.fetch_news_titles("sina", t, concept=concept, limit=10)
            concept_data[concept] = {
                "govcn": govcn,
                "news": em_news + sina_news,
            }

        return {
            "shared": {"mpc": mpc, "csrc": csrc},
            "concepts": concept_data,
        }
