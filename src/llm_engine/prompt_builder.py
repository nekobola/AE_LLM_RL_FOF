"""Build LLM prompts from ETL text data and concept pool definitions."""

from __future__ import annotations

from typing import Optional

from src.llm_engine.concept_pools import CONCEPT_POOLS, DIMENSIONS


# Equity / satellite pool: 8 concepts sharing macro sentiment
EQUITY_CONCEPTS = CONCEPT_POOLS["wide_base"] + CONCEPT_POOLS["satellite"]

# Fixed income pool: 2 concepts with distinct yield-curve drivers
FIXED_INCOME_CONCEPTS = CONCEPT_POOLS["fixed_income"]


class PromptBuilder:
    """Aggregate ETL text into structured prompts for macro concept scoring."""

    SYSTEM_PROMPT_EQUITY = """你是一个专业的宏观量化策略师。你需要根据文本信息，判断不同股票概念板块的以下三个维度的相对强弱（满分100）：

- d1 流动性顺风：货币宽松、流动性充裕、融资环境好 → 高分
- d2 资金情绪：机构看多、主题炒作热情、净流入 → 高分
- d3 风险压力指数：市场隐含风险、地缘压力、流动性紧张程度 → 高分

【d3评分指南 - 关键！】
d3不是简单的"坏消息计数"，而是市场隐含的风险压力程度。评分逻辑：

【高风险区 (d3=70~100)】
- 中美关系紧张、美国制裁、出口管制升级
- 国内金融危机信号：银行流动性紧张、债市违约潮、房地产崩盘
- 政策突然转向收紧：加息、监管骤变
- 地缘冲突升级：台海、南海紧张

【中等风险区 (d3=50~70)】
- d1/d2环比下降（流动性/情绪边际恶化）
- 市场波动率明显上升
- 外部环境不确定性增加（美联储紧缩、全球经济放缓）

【低风险区 (d3=30~50)】
- 政策宽松基调延续、流动性平稳
- 无重大风险事件
- 市场情绪稳定

【特别注意】
1. 你在对多个相关概念做横向比较评分，而非绝对评分
2. 优先参考近1周【变化】而非绝对水平。政策/新闻数量环比增加=利好，减少=利空
3. d1/d2/d3 之间可以有明显差异，不需要趋同
4. 当某概念本周没有任何相关政策/新闻时：
   - 如果上周d1/d2/d3处于高位，d3建议维持60-70（风险有惯性）
   - 如果上周d1/d2/d3处于低位，d3建议40-55（边际变化判断）
5. 分数必须是1.0~100.0的浮点数"""

    SYSTEM_PROMPT_FIXED_INCOME = """你是一个专业的固定收益宏观策略师。你需要根据文本信息，判断债券板块的以下三个维度（满分100）：

- d1 流动性顺风：货币宽松、降准降息、流动性充裕 → 高分
- d2 资金情绪：机构看多债市、配置需求强、净买入 → 高分
- d3 风险压力指数：通胀预期升温、流动性收紧、供给冲击、信用利差扩大 → 高分

【d3评分指南 - 关键！】
d3不是简单的"坏消息计数"，而是债市隐含的风险压力程度。

【高风险区 (d3=70~100)】
- 通胀预期显著升温：CPI/PPI大幅上行
- 流动性突然收紧：银行间利率飙升、资金面紧张
- 信用风险爆发：大面积违约、信用利差急剧扩大
- 供给冲击：利率债供给大量发行导致承压

【中等风险区 (d3=50~70)】
- d1/d2环比下降
- 通胀预期边际升温
- 资金面边际收紧

【低风险区 (d3=30~50)】
- 央行维持宽松基调
- 通胀平稳
- 资金面平稳

【特别注意】
1. 长期利率债（10Y国债）主要受宏观利率驱动，信用债同时受信用利差驱动
2. 对两个板块做横向比较
3. 当本周无相关政策/新闻时：
   - 如果上周处于低风险状态，d3建议40-50
   - 如果上周处于高风险状态，d3建议60-70（风险有惯性）
4. 重点关注近7天内的变化"""

    def build(
        self,
        etl_data: dict,
        concept_list: list[str],
        prior_scores: Optional[dict[str, dict[str, float]]] = None,
    ) -> str:
        """Build a themed prompt for a given list of concepts.

        Parameters
        ----------
        etl_data : dict
            Output from TextETL.extract_per_concept().
        concept_list : list[str]
            Concepts to score in this prompt (e.g. EQUITY_CONCEPTS).
        prior_scores : dict, optional
            {concept_name: {"d1": float, "d2": float, "d3": float}} from prior week.

        Returns
        -------
        str
            Complete user prompt string.
        """
        shared = etl_data.get("shared", {})
        concepts_data = etl_data.get("concepts", {})

        sections: list[str] = []

        # --- Prior week reference ---
        if prior_scores:
            prior_lines = []
            for concept in concept_list:
                scores = prior_scores.get(concept)
                if scores:
                    prior_lines.append(
                        f"  {concept}: d1={scores['d1']:.1f}, d2={scores['d2']:.1f}, d3={scores['d3']:.1f}"
                    )
            if prior_lines:
                sections.append("【参考：上周评分】\n" + "\n".join(prior_lines))

        # --- MPC ---
        mpc = shared.get("mpc") or []
        if mpc:
            rec = mpc[0]
            content = rec.get("content") or rec.get("title") or "（内容暂无）"
            mpc_date = rec.get("date", "N/A")
            sections.append(
                f"【央行货币政策例会】（最近一次，日期：{mpc_date}）\n{content[:500]}"
            )

        # --- CSRC ---
        csrc = shared.get("csrc") or []
        if csrc:
            titles = "\n".join(f"- {t}" for t in csrc[:15])
            sections.append(f"【证监会动态】（近7天，共{len(csrc)}条）\n{titles}")

        # --- Per-concept sections ---
        for concept in concept_list:
            data = concepts_data.get(concept, {})
            govcn = data.get("govcn") or []
            news = data.get("news") or []

            concept_lines = [f"【{concept}】"]

            if not govcn and not news:
                prior_hint = ""
                if prior_scores and concept in prior_scores:
                    p = prior_scores[concept]
                    # 无新闻时：d1/d2向50回归（0.95衰减），d3维持或略升（风险有惯性）
                    d1_hint = p["d1"] * 0.95 + 50 * 0.05  # 向50回归5%
                    d2_hint = p["d2"] * 0.95 + 50 * 0.05
                    d3_hint = max(40.0, p["d3"] * 1.02)    # 风险有惯性，最少40
                    prior_hint = (
                        f"\n参考上周该指标：d1={p['d1']:.1f}, d2={p['d2']:.1f}, d3={p['d3']:.1f}。"
                        f"\n本周无新信息，建议：d1≈{d1_hint:.1f}, d2≈{d2_hint:.1f}, d3≈{d3_hint:.1f}。"
                    )
                else:
                    # 无历史数据时的基准：d1/d2中性50，d3中等偏高40（而非极低25）
                    prior_hint = "\n本周无相关政策/新闻，请给出中性基准评分：d1=50, d2=50, d3=40。"
                concept_lines.append(
                    "⚠️ 本周无相关政策文件或市场新闻。"
                    "请根据共享宏观文本（货币政策例会等）做合理推断。"
                    + prior_hint
                )
            else:
                if govcn:
                    concept_lines.append(f"相关政策文件（共{len(govcn)}条，近30天）：")
                    for item in govcn[:8]:
                        title = item.get("title", "")[:40]
                        content = (item.get("content") or "")[:150]
                        date = item.get("date", "")
                        concept_lines.append(f"  [{date}] {title}")
                        if content:
                            concept_lines.append(f"    {content}")
                else:
                    concept_lines.append("相关政策文件：0条")

                if news:
                    concept_lines.append(f"\n市场新闻标题（共{len(news)}条，近7天）：")
                    for title in news[:10]:
                        concept_lines.append(f"  - {title}")
                else:
                    concept_lines.append("\n市场新闻标题：0条")

            sections.append("\n".join(concept_lines))

        # --- Output format ---
        json_example = ",\n  ".join(
            f'"{c}": {{"d1": float, "d2": float, "d3": float}}' for c in concept_list
        )

        user_prompt = (
            "【待评分概念】\n" + ", ".join(concept_list) + "\n\n"
            + "\n\n".join(sections)
            + f"\n\n请输出JSON格式评分：\n{{\n  {json_example}\n}}"
        )

        return user_prompt

    def build_equity(
        self,
        etl_data: dict,
        prior_scores: Optional[dict[str, dict[str, float]]] = None,
    ) -> str:
        """Build the equity/satellite pool prompt."""
        return self.build(etl_data, EQUITY_CONCEPTS, prior_scores)

    def build_fixed_income(
        self,
        etl_data: dict,
        prior_scores: Optional[dict[str, dict[str, float]]] = None,
    ) -> str:
        """Build the fixed income pool prompt."""
        return self.build(etl_data, FIXED_INCOME_CONCEPTS, prior_scores)

    def system_prompt(self, pool: str = "equity") -> str:
        """Return the system prompt for a given pool."""
        if pool == "fixed_income":
            return self.SYSTEM_PROMPT_FIXED_INCOME
        return self.SYSTEM_PROMPT_EQUITY
