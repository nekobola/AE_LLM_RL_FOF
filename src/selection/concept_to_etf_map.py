"""Concept -> ETF pool mapping with configurable override support."""

from typing import Optional

# 概念名称 -> 对应ETF代码池（示例，需要可配置）
DEFAULT_ETF_POOLS = {
    "人工智能": ["159819.SZ", "515070.SH"],
    "创新药": ["512010.SH", "159992.SZ"],
    "半导体": ["512480.SH", "159813.SZ"],
    "煤炭": ["515220.SH"],
    "新能源": ["515030.SH", "159928.SZ"],
    "红利低波": ["515050.SH"],
    "沪深300": ["510300.SH"],
    "中证1000": ["512100.SH"],
    "长期利率债": ["511010.SH"],
    "信用债": ["511020.SH"],
    "黄金": ["518880.SH"],
    "货币": ["511850.SH"],
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
    "黄金": ["518880.SH"],
    "货币": ["511850.SH"],
}


def get_etf_pool_by_concept(concept: str, etf_pools: Optional[dict] = None) -> list[str]:
    """Get ETF pool for a given concept, with optional runtime override."""
    pools = etf_pools if etf_pools is not None else DEFAULT_ETF_POOLS
    return pools.get(concept, [])


def get_concepts_by_category(category: str) -> list[str]:
    """Return all concepts belonging to a given slot category."""
    return [c for c, cat in CONCEPT_CATEGORY_MAP.items() if cat == category]
