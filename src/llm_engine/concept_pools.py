"""Concept pool definitions — configurable concept universe for macro scoring."""

# Three concept pools: wide-base, satellite, fixed-income
# Keys match config.yaml (concept_pools.wide_base / satellite / fixed_income)
CONCEPT_POOLS = {
    "wide_base": ["沪深300", "中证1000"],
    "satellite": ["人工智能", "创新药", "半导体", "煤炭", "新能源", "红利低波"],
    "fixed_income": ["长期利率债", "信用债"],
}

# Dimension semantics for LLM prompt construction
DIMENSIONS = {
    "d1": "流动性顺风（流动性与政策支撑度）",
    "d2": "资金情绪（新闻看多一致性）",
    "d3": "尾部风险（政策打压或地缘风险）",
}
