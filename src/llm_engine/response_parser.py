"""Parse and validate LLM JSON responses for macro concept scores."""

from __future__ import annotations

import json
from typing import Any


class ParseError(ValueError):
    """Raised when LLM output cannot be parsed or fails validation."""


class ResponseParser:
    """Parse LLM JSON output and validate d1/d2/d3 score ranges."""

    MIN_SCORE = 1.0
    MAX_SCORE = 100.0
    REQUIRED_DIMENSIONS = {"d1", "d2", "d3"}

    def parse(self, raw: str) -> dict[str, dict[str, float]]:
        """Parse and validate LLM JSON response.

        Parameters
        ----------
        raw : str
            Raw JSON string from LLM response.

        Returns
        -------
        dict[str, dict[str, float]]
            Mapping of concept name -> {d1, d2, d3} scores.

        Raises
        ------
        ParseError
            If JSON is malformed or any score is out of [1.0, 100.0] range.
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ParseError(f"JSON decode error: {e}") from e

        if not isinstance(data, dict):
            raise ParseError(f"Expected JSON object, got {type(data).__name__}")

        self._validate_scores(data)
        return data

    def _validate_scores(self, data: dict) -> None:
        """Validate that every concept entry has valid d1/d2/d3 scores."""
        for concept, scores in data.items():
            if not isinstance(scores, dict):
                raise ParseError(
                    f"Concept '{concept}': expected dict of scores, got {type(scores).__name__}"
                )
            missing = self.REQUIRED_DIMENSIONS - set(scores.keys())
            if missing:
                raise ParseError(
                    f"Concept '{concept}' missing dimensions: {missing}"
                )
            for dim in self.REQUIRED_DIMENSIONS:
                val = scores.get(dim)
                if not isinstance(val, (int, float)):
                    raise ParseError(
                        f"Concept '{concept}', dimension '{dim}': "
                        f"expected float, got {type(val).__name__}"
                    )
                if not (self.MIN_SCORE <= float(val) <= self.MAX_SCORE):
                    raise ParseError(
                        f"Concept '{concept}', dimension '{dim}': "
                        f"value {val} outside [{self.MIN_SCORE}, {self.MAX_SCORE}]"
                    )
