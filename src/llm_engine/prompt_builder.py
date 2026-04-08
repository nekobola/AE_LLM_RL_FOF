"""Build LLM prompts from ETL text data and concept pool definitions."""

from __future__ import annotations

from src.llm_engine.concept_pools import CONCEPT_POOLS, DIMENSIONS


class PromptBuilder:
    """Aggregate ETL text into a structured prompt for macro concept scoring."""

    SYSTEM_PROMPT = """你是一个专业的宏观策略分析师。请根据以下文本信息，对各个概念进行评分。
输出必须为严格的JSON格式，包含所有概念的d1/d2/d3评分（浮点数，值域1.0~100.0）。
维度说明：
- d1: 流动性顺风（流动性与政策支撑度，越高越好）
- d2: 资金情绪（新闻看多一致性，越高越看多）
- d3: 尾部风险（政策打压或地缘风险，越高风险越大）
"""

    def build(self, etl_data: dict, concept_pools: dict = None) -> str:
        """Build the user prompt from ETL data and concept pools.

        Parameters
        ----------
        etl_data : dict
            Output from TextETL.extract_all().
        concept_pools : dict, optional
            Override CONCEPT_POOLS. Defaults to the module-level constant.

        Returns
        -------
        str
            Complete user prompt string.
        """
        if concept_pools is None:
            concept_pools = CONCEPT_POOLS

        sections: list[str] = []

        # --- 1. PBoC MPC ---
        zgrmyh = etl_data.get("zgrmyh") or []
        if zgrmyh:
            rec = zgrmyh[0]
            content = rec.get("content") or rec.get("title") or "（内容暂无）"
            sections.append(
                f"【货币政策例会】\n日期：{rec.get('date', 'N/A')}\n标题：{rec.get('title', 'N/A')}\n内容：{content}"
            )

        # --- 2. CSRC titles ---
        csrc = etl_data.get("csrc_titles") or []
        if csrc:
            titles = "\n".join(f"- {t}" for t in csrc[:20])
            sections.append(f"【证监会动态】（近7天，共{len(csrc)}条）\n{titles}")

        # --- 3. Global govcn ---
        govcn = etl_data.get("govcn_global") or []
        if govcn:
            items = "\n".join(f"- {g.get('title', '')}" for g in govcn[:20])
            sections.append(f"【全局政策动态】（近7天，共{len(govcn)}条）\n{items}")

        # --- 4. Concept pool summary (English key → Chinese display name) ---
        DISPLAY_NAMES = {
            "wide_base": "宽基池",
            "satellite": "卫星池",
            "fixed_income": "固收池",
        }
        pool_lines: list[str] = []
        for pool_key, concepts in concept_pools.items():
            display = DISPLAY_NAMES.get(pool_key, pool_key)
            pool_lines.append(f"  {display}：{', '.join(concepts)}")
        pool_summary = "\n".join(pool_lines)

        # --- 5. Dimension reference ---
        dim_lines = "\n".join(f"  {k}: {v}" for k, v in DIMENSIONS.items())

        user_prompt = (
            f"【待评分概念池】\n{pool_summary}\n\n"
            f"【维度定义】\n{dim_lines}\n\n"
            f"【近期宏观文本】\n" + "\n\n".join(sections) + "\n\n"
            "请对每个概念输出JSON，格式：\n"
            "{\n  \"概念名\": {\"d1\": float, \"d2\": float, \"d3\": float},\n  ...\n}\n"
            "所有评分必须在1.0~100.0范围内。"
        )
        return user_prompt
