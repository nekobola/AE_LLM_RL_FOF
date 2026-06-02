"""Async LLM macro semantic engine — weekly Friday evaluation pipeline."""

from __future__ import annotations

import asyncio
from typing import Optional

import httpx
from openai import AsyncOpenAI

from src.llm_engine.concept_pools import CONCEPT_POOLS
from src.llm_engine.prompt_builder import PromptBuilder
from src.llm_engine.response_parser import ParseError, ResponseParser
from src.llm_engine.text_etl import TextETL


class LLMCallError(RuntimeError):
    """Raised when LLM invocation fails after all retries."""


class AsyncSemanticEngine:
    """Async LLM engine for weekly macro concept scoring.

    Pipeline: ETL (per-concept) → 2 themed prompts (equity + fixed income)
            → 2 AsyncOpenAI calls → merge scores
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.llm_config = config["llm"]
        http_client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=30.0))
        self.client = AsyncOpenAI(
            api_key=self.llm_config["api_key"],
            base_url=self.llm_config.get("base_url"),
            http_client=http_client,
        )
        self.etl = TextETL(config)
        self.prompt_builder = PromptBuilder()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        current_date: str,
        prior_scores: Optional[dict[str, dict[str, float]]] = None,
    ) -> dict[str, dict[str, float]]:
        """Run full evaluation pipeline for the given Friday date.

        Makes 2 LLM calls: equity pool + fixed income pool, then merges.

        Parameters
        ----------
        current_date : str
            Friday date in YYYY-MM-DD format.
        prior_scores : dict, optional
            {concept_name: {"d1": float, "d2": float, "d3": float}} from prior week.

        Returns
        -------
        dict[str, dict[str, float]]
            {concept_name: {"d1": float, "d2": float, "d3": float}}

        Raises
        ------
        LLMCallError
            If LLM call fails after max retries.
        ParseError
            If LLM output fails validation.
        """
        # 1. ETL: per-concept extraction
        all_concepts = CONCEPT_POOLS["wide_base"] + CONCEPT_POOLS["satellite"] + CONCEPT_POOLS["fixed_income"]
        etl_data = self.etl.extract_per_concept(current_date, all_concepts, lookback=30)

        # 2. Build 2 themed prompts
        equity_prompt = self.prompt_builder.build_equity(etl_data, prior_scores)
        fi_prompt = self.prompt_builder.build_fixed_income(etl_data, prior_scores)
        equity_sys = self.prompt_builder.system_prompt("equity")
        fi_sys = self.prompt_builder.system_prompt("fixed_income")

        # 3. Two concurrent LLM calls
        equity_response, fi_response = await asyncio.gather(
            self._call_llm_with_retry(equity_sys, equity_prompt),
            self._call_llm_with_retry(fi_sys, fi_prompt),
        )

        # 4. Parse both responses
        equity_scores = ResponseParser().parse(equity_response)
        fi_scores = ResponseParser().parse(fi_response)

        # 5. Merge
        all_scores = {**equity_scores, **fi_scores}
        return all_scores

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    async def _call_llm_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Call AsyncOpenAI with exponential-backoff retry."""
        max_retries = self.llm_config.get("max_retries", 3)
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.llm_config.get("model", "gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                if attempt == max_retries - 1:
                    raise LLMCallError(f"LLM调用失败 {max_retries}次: {e}") from e
                await asyncio.sleep(2**attempt)
        # Defensive: should not reach here
        raise LLMCallError("重试耗尽")
