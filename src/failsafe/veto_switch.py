"""
绝对风险一票否决逻辑。

触发条件：LLM评分结果中，任意概念的 d3（尾部风险）> veto_d3_threshold。

执行动作：
1. 从备选池中抹除该概念
2. 由Score排行榜下一顺位替补
"""


class VetoSwitch:
    """
    绝对风险一票否决逻辑。
    """

    VETO_THRESHOLD: float = 85.0  # 可被config覆盖

    def __init__(self, config: dict | None = None):
        if config is not None:
            self.VETO_THRESHOLD = config["failsafe"]["veto_d3_threshold"]

    def apply_veto(
        self,
        llm_scores: dict,  # {概念名: {"d1": float, "d2": float, "d3": float}}
        ranked_concepts: list,  # 已按Score排序的概念列表
    ) -> list:
        """
        返回经否决过滤后的概念列表。

        如果某概念的d3 > veto_threshold，则：
        - 从ranked_concepts中移除
        - 用下一顺位替补
        """
        vetoed = [
            concept for concept, scores in llm_scores.items()
            if scores.get("d3", 0) > self.VETO_THRESHOLD
        ]
        if vetoed:
            print(f"[VetoSwitch] 否决概念: {vetoed}")

        # 从ranked_concepts中移除否决概念，顺序补入
        result = [c for c in ranked_concepts if c not in vetoed]
        return result
