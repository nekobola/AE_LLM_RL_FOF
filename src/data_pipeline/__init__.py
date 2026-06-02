"""Data Pipeline Module

Dual-Track Data Architecture:
- Track A: AkShare public API (index data, 2015-present)
- Track B: quantchdb / ClickHouse (ETF data, 2022-present)
"""
from src.data_pipeline.track_a.fetcher import fetch_track_a
from src.data_pipeline.track_b.fetcher import fetch_track_b, fetch_track_b_safe

__all__ = ["fetch_track_a", "fetch_track_b", "fetch_track_b_safe"]
