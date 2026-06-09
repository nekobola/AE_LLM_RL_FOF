"""Stage 7c 8-ETF selector: simple LLM-score-driven ETF ranking.

简化版 AsymmetricSelector, 专为 8 资产 V3.1 优化:
- 8 个 ETF 池 (5y 回测验证后的子集)
- LLM d1 (宏观) > 50 时倾向 satellite + 中证 1000, < 50 时倾向 bond + gold
- LLM d3 (风险) > 70 时强制降低 equity 仓位
- VetoSwitch: d3 > 85 直接 100% 防御 (国债+黄金+信用债)

如果 LLM 评分缺失 (LLM 宕机), fallback 到 FallbackSelector (动量).
"""
from __future__ import annotations
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# 8 ETF 池 (按 V31EngineN.ASSET_CODES 顺序)
ETF_POOL_8 = {
    "511010": {"category": "fi",      "concept": "国债",    "name": "国债 ETF"},
    "518880": {"category": "hedging", "concept": "黄金",    "name": "黄金 ETF"},
    "511020": {"category": "fi",      "concept": "信用债",  "name": "信用债 ETF"},
    "159985": {"category": "commodity","concept": "商品",    "name": "商品 ETF"},
    "512100": {"category": "satellite","concept": "中证1000","name": "中证1000 ETF"},
    "515050": {"category": "satellite","concept": "红利低波","name": "红利低波 ETF"},
    "159915": {"category": "satellite","concept": "创业板","name": "创业板 ETF"},
    "510300": {"category": "satellite","concept": "沪深300", "name": "沪深300 ETF"},
}


class Select8N:
    """8-ETF selector for Stage 7c.

    核心逻辑 (简化版):
      score_etf(etf) = base_score + 0.5 * (d1 - 50) * weight + 0.3 * (d2 - 50) * weight - 0.4 * (d3 - 50) * weight
      base_score:
        - bond/gold: +30 (always attractive)
        - satellite: +0
      Veto d3 > 85: drop all equity ETFs
    """

    def __init__(self, config: dict):
        self.config = config
        self.liquid_min = config.get("selection", {}).get("liquidity_min_amt", 30_000_000)
        self.d3_veto = config.get("failsafe", {}).get("veto_d3_threshold", 85.0)

    def select_8(
        self,
        llm_macro: float,
        llm_sentiment: float,
        llm_risk: float,
    ) -> Dict[str, float]:
        """Score 8 ETFs based on LLM signals. Returns {etf_code: composite_score}.

        不做 ClickHouse 流动性过滤 (5y 样本里这 8 个 ETF 都满足流动性).
        不做动量 tiebreak (简化).
        """
        # d1 ∈ [0, 100], 50 = 中性
        d1_norm = (llm_macro - 50.0) / 50.0  # [-1, 1]
        d2_norm = (llm_sentiment - 50.0) / 50.0
        d3_norm = (llm_risk - 50.0) / 50.0  # [0, 1] tail risk

        scores = {}
        for etf_code, meta in ETF_POOL_8.items():
            cat = meta["category"]
            # base score: bond/gold 天然 30 分, satellite 0 分
            if cat in ("fi", "hedging", "commodity"):
                base = 30.0
            else:
                base = 0.0

            # LLM 信号加权
            if cat == "fi":
                # 债: 喜欢高 d1 (宽松政策) + 高 d2 (景气) - 高 d3 (避险)
                score = base + 0.3 * d1_norm * 30 + 0.2 * d2_norm * 20 - 0.4 * d3_norm * 20
            elif cat == "hedging":
                # 黄金: 喜欢低 d1 (避险) + 高 d3 (避险)
                score = base - 0.3 * d1_norm * 30 - 0.2 * d2_norm * 20 + 0.5 * d3_norm * 30
            elif cat == "commodity":
                # 商品: 喜欢高 d1 (通胀) + 高 d2 (景气)
                score = base + 0.4 * d1_norm * 30 + 0.3 * d2_norm * 20 - 0.2 * d3_norm * 20
            else:  # satellite
                # 卫星: 喜欢高 d1 + 高 d2 - 高 d3
                score = base + 0.5 * d1_norm * 30 + 0.4 * d2_norm * 30 - 0.5 * d3_norm * 30

            scores[etf_code] = score

        # Veto: d3 > 85 直接压低所有 equity (satellite) 评分
        if llm_risk > self.d3_veto:
            for etf_code, meta in ETF_POOL_8.items():
                if meta["category"] == "satellite":
                    scores[etf_code] = -100.0

        return scores


def fallback_8_etf_scores() -> Dict[str, float]:
    """LLM 宕机时, 用等权重 fallback (所有 ETF 30 分)."""
    return {etf_code: 30.0 for etf_code in ETF_POOL_8.keys()}
