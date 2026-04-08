"""Asymmetric truncation selection: maps LLM scores to 8-slot FOF portfolio.

Slot rules:
  0_宽基   -> Score rank 1  (wide_base concept with highest score)
  1_卫星_1 -> Score rank 1 among remaining satellite concepts
  2_卫星_2 -> Score rank 2 among remaining satellite concepts
  3_卫星_3 -> Score rank 3 among remaining satellite concepts
  4_固收_利率债 -> fixed_income concept "长期利率债" (dumbbell leg 1)
  5_固收_信用债 -> fixed_income concept "信用债" (dumbbell leg 2)
  6_避险   -> Fixed gold ETF: 518880.SH
  7_现金   -> Fixed money market ETF: 511850.SH
"""

from typing import Dict, Optional

from .clickhouse_hard_clip import ETFSelector
from .concept_to_etf_map import (
    CONCEPT_CATEGORY_MAP,
    DEFAULT_ETF_POOLS,
    FIXED_SLOT_ETFS,
    get_etf_pool_by_concept,
)
from .slot_weighting import P_VECTORS, compute_slot_score


class AsymmetricSelector:
    """Asymmetric truncation selector for 8-slot FOF portfolio construction."""

    # 8 fixed slot names
    SLOT_NAMES = [
        "0_宽基",
        "1_卫星_1",
        "2_卫星_2",
        "3_卫星_3",
        "4_固收_利率债",
        "5_固收_信用债",
        "6_避险",
        "7_现金",
    ]

    def __init__(
        self,
        config: dict,
        etf_pools: Optional[dict] = None,
    ):
        """Initialize with runtime config and optional ETF pool override."""
        self.config = config
        self.etf_pools = etf_pools if etf_pools is not None else DEFAULT_ETF_POOLS
        self.sel = ETFSelector(config)
        self._liquid_min = config["selection"]["liquidity_min_amt"]
        self._momentum_window = config["selection"]["momentum_window"]

    def _score_concepts_for_category(
        self,
        llm_scores: Dict[str, Dict[str, float]],
        category: str,
    ) -> list[tuple[str, float, str]]:
        """Score all concepts in a category, return sorted (concept, score, pool_type)."""
        scored = []
        for concept, dims in llm_scores.items():
            if CONCEPT_CATEGORY_MAP.get(concept) != category:
                continue
            d1 = dims.get("d1", 50.0)
            d2 = dims.get("d2", 50.0)
            d3 = dims.get("d3", 50.0)
            score = compute_slot_score(d1, d2, d3, category)
            scored.append((concept, score, category))
        # Sort descending by score
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _select_one_by_concept(
        self,
        concept: str,
        t: str,
        pool_type: str,
    ) -> Optional[str]:
        """Select best ETF for a given concept, applying liquidity veto + momentum tiebreak."""
        codes = get_etf_pool_by_concept(concept, self.etf_pools)
        if not codes:
            return None
        # Hard filter: liquidity
        liquid_codes = self.sel.liquidity_veto(
            codes, t, min_amt=self._liquid_min
        )
        if not liquid_codes:
            return None
        # Tiebreak: momentum
        winners = self.sel.tiebreaker_momentum(
            liquid_codes, t, top_n=1, window=self._momentum_window
        )
        return winners[0] if winners else None

    def select_8(
        self,
        llm_scores: Dict[str, Dict[str, float]],
        current_date: str,
    ) -> Dict[str, str]:
        """Select 8 ETFs based on LLM semantic scores and hard quant filters.

        Args:
            llm_scores: {concept_name: {"d1": float, "d2": float, "d3": float}}
            current_date: Selection date string (YYYY-MM-DD)

        Returns:
            {slot_name: etf_code} dict with exactly 8 entries
        """
        result: Dict[str, str] = {}

        # ── Slot 0: Wide Base (top 1) ───────────────────────────────────────
        wide_scored = self._score_concepts_for_category(llm_scores, "wide_base")
        if wide_scored:
            concept, _, _ = wide_scored[0]
            etf = self._select_one_by_concept(concept, current_date, "wide_base")
            if etf:
                result["0_宽基"] = etf

        # ── Slots 1-3: Satellite (top 3 concepts, no industry overlap) ───────
        satellite_scored = self._score_concepts_for_category(llm_scores, "satellite")
        selected_satellite_concepts: list[str] = []
        for concept, score, pool_type in satellite_scored:
            if len(selected_satellite_concepts) >= 3:
                break
            etf = self._select_one_by_concept(concept, current_date, "satellite")
            if etf is None:
                continue
            selected_satellite_concepts.append(concept)
            slot_idx = len(selected_satellite_concepts)
            result[f"{slot_idx}_卫星_{slot_idx}"] = etf

        # Fill remaining satellite slots if not enough qualified
        for i in range(len(selected_satellite_concepts) + 1, 4):
            result[f"{i}_卫星_{i}"] = ""

        # ── Slots 4-5: Fixed Income dumbbell ─────────────────────────────────
        fi_concepts = ["长期利率债", "信用债"]
        for idx, concept in enumerate(fi_concepts, start=4):
            etf = self._select_one_by_concept(
                concept, current_date, "fixed_income"
            )
            result[f"{idx}_固收_{concept}"] = etf if etf else ""

        # ── Slot 6: Hedging (fixed gold ETF) ────────────────────────────────
        gold_codes = FIXED_SLOT_ETFS.get("黄金", ["518880.SH"])
        liquid_gold = self.sel.liquidity_veto(
            gold_codes, current_date, min_amt=self._liquid_min
        )
        result["6_避险"] = liquid_gold[0] if liquid_gold else "518880.SH"

        # ── Slot 7: Cash (fixed money market ETF) ───────────────────────────
        cash_codes = FIXED_SLOT_ETFS.get("货币", ["511850.SH"])
        liquid_cash = self.sel.liquidity_veto(
            cash_codes, current_date, min_amt=self._liquid_min
        )
        result["7_现金"] = liquid_cash[0] if liquid_cash else "511850.SH"

        return result
