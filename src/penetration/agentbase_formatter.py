class AgentBaseFormatter:
    """
    将穿透后的ETF权重组装为AgentBase要求的嵌套字典格式。

    输出格式:
    {
        "YYYY-MM-DD": {
            "510300.SH": 0.25,
            "159819.SZ": 0.0833,
            ...
        }
    }
    """

    def format(
        self,
        etf_weights: dict[str, float],
        etf_codes: dict[str, str],  # {slot_name: etf_code}
        current_date: str,
    ) -> dict:
        """
        Parameters
        ----------
        etf_weights : dict
            weight_unwrapper.unwrap() 的输出
        etf_codes : dict
            {slot_name: etf_code} 映射
        current_date : str
            YYYY-MM-DD 格式

        Returns
        -------
        dict
            AgentBase格式的嵌套字典
        """
        holdings = {}
        for slot, etf_code in etf_codes.items():
            holdings[etf_code] = etf_weights.get(slot, 0.0)

        return {current_date: holdings}
