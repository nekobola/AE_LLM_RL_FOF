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
    "d1": "流动性顺风（流动性充裕程度）",
    "d2": "资金情绪（市场看多一致性）",
    "d3": "风险压力指数（市场隐含风险、地缘压力）",
}
