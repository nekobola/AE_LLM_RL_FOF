"""Concept -> ETF pool mapping with configurable override support."""

from typing import Optional

# 概念名称 -> 对应ETF代码池（与ClickHouse etf_day表中的code字段对应，无后缀）
DEFAULT_ETF_POOLS = {
    "人工智能": ["159819", "515070"],
    "创新药": ["512010", "159992"],
    "半导体": ["512480", "159813"],
    "煤炭": ["515220"],
    "新能源": ["515030", "159928"],
    "红利低波": ["515050"],
    "沪深300": ["510300"],
    "中证1000": ["512100"],
    "长期利率债": ["511010"],
    "信用债": ["511020"],
    "黄金": ["518880"],
    "货币": ["511850"],
}

# 概念 -> 插槽类别映射
CONCEPT_CATEGORY_MAP = {
    # 宽基
    "沪深300": "wide_base",
    "中证1000": "wide_base",
    # 卫星
    "人工智能": "satellite",
    "创新药": "satellite",
    "半导体": "satellite",
    "煤炭": "satellite",
    "新能源": "satellite",
    "红利低波": "satellite",
    # 固收
    "长期利率债": "fixed_income",
    "信用债": "fixed_income",
    # 避险
    "黄金": "hedging",
    # 现金
    "货币": "cash",
}

# 固定插槽ETF（不参与评分排序）
FIXED_SLOT_ETFS = {
    "黄金": ["518880"],
    "货币": ["511850"],
}


def get_etf_pool_by_concept(concept: str, etf_pools: Optional[dict] = None) -> list[str]:
    """Get ETF pool for a given concept, with optional runtime override."""
    pools = etf_pools if etf_pools is not None else DEFAULT_ETF_POOLS
    return pools.get(concept, [])


def get_concepts_by_category(category: str) -> list[str]:
    """Return all concepts belonging to a given slot category."""
    return [c for c, cat in CONCEPT_CATEGORY_MAP.items() if cat == category]
