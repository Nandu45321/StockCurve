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


def vertical_mae(sketch: np.ndarray, curve: np.ndarray) -> float:
    """
    Mean absolute error between z-normalized sketch and match curve.
    Both must be 50-point z-normalized arrays.
    Lower is better. Units: standard deviations (sigma).
    """
    return float(np.mean(np.abs(sketch - curve)))


def mae_to_pct(mae: float, max_mae: float = 1.5) -> int:
    """
    Convert MAE to a human-readable match percentage.
    max_mae=1.5 means anything above 1.5 sigma avg error = 0% match.
    0 sigma error = 100% match.
    """
    return max(0, int((1 - mae / max_mae) * 100))


def shape_score(sketch: np.ndarray, curve: np.ndarray) -> float:
    """
    Combined ranking score: MAE + std of pointwise absolute errors.
    Lower is better. Penalises both high average error and high variance
    of errors (a curve with a consistent small error beats one with a
    low mean but large spikes).
    """
    errors = np.abs(sketch - curve)
    return float(np.mean(errors) + np.std(errors))
