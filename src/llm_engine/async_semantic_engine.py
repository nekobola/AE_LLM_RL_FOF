"""Async LLM macro semantic engine — weekly Friday evaluation pipeline."""

from __future__ import annotations

import asyncio

from openai import AsyncOpenAI

from src.llm_engine.concept_pools import DIMENSIONS, CONCEPT_POOLS
from src.llm_engine.prompt_builder import PromptBuilder
from src.llm_engine.response_parser import ParseError, ResponseParser
from src.llm_engine.text_etl import TextETL


class LLMCallError(RuntimeError):
    """Raised when LLM invocation fails after all retries."""


class AsyncSemanticEngine:
    """Async LLM engine for weekly macro concept scoring.

    Pipeline: ETL → PromptBuilder → AsyncOpenAI → ResponseParser → scores
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.llm_config = config["llm"]
        self.client = AsyncOpenAI(api_key=self.llm_config["api_key"])
        self.etl = TextETL(config)
        self.prompt_builder = PromptBuilder()
        self.parser = ResponseParser()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def evaluate(self, current_date: str) -> dict[str, dict[str, float]]:
        """Run full evaluation pipeline for the given Friday date.

        Parameters
        ----------
        current_date : str
            Friday date in YYYY-MM-DD format.

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
        # 1. ETL
        etl_data = self.etl.extract_all(current_date)

        # 2. Build prompt (prefer config concept_pools, fallback to module constant)
        concept_pools = self.config.get("concept_pools", CONCEPT_POOLS)
        prompt = self.prompt_builder.build(etl_data, concept_pools)

        # 3. LLM call with retry
        response = await self._call_llm_with_retry(prompt)

        # 4. Parse
        scores = self.parser.parse(response)
        return scores

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    async def _call_llm_with_retry(self, prompt: str) -> str:
        """Call AsyncOpenAI with exponential-backoff retry."""
        max_retries = self.llm_config.get("max_retries", 3)
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.llm_config.get("model", "gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": PromptBuilder.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt == max_retries - 1:
                    raise LLMCallError(f"LLM调用失败 {max_retries}次: {e}") from e
                await asyncio.sleep(2**attempt)
        # Defensive: should not reach here
        raise LLMCallError("重试耗尽")
