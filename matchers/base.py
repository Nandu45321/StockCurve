"""
matchers/base.py — Abstract base class for all shape matchers.

Every matcher must inherit BaseMatcher and implement the match() method.
The interface is non-negotiable — see AGENTS.md Matcher Interface section.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseMatcher(ABC):
    """Abstract base for all shape-matching strategies."""

    name: str  # shown in the UI progress tracker

    @abstractmethod
    def match(self, sketch: np.ndarray, candidates: list[dict]) -> list[dict]:
        """
        Score each candidate window against the sketch and return sorted results.

        Args:
            sketch: z-normalized 50-point numpy array from user drawing
            candidates: list of window dicts (see AGENTS.md Data Schema)

        Returns:
            A NEW list (never modifies in-place) with 'score' key added to each
            candidate dict. Lower score = better match. Sorted best-first.
        """
        pass
