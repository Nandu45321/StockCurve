"""
matchers/euclidean.py — Stage 2: Smoothed Euclidean distance matcher.

Compares the smoothed sketch against smoothed windows, keeps the top 20%
of candidates by distance. Lower score = better match.
"""

import numpy as np
from .base import BaseMatcher


class EuclideanMatcher(BaseMatcher):
    """Smoothed Euclidean distance matcher (Stage 2)."""

    name = "Euclidean"

    def __init__(self, sigma: float = 2.0):
        self.sigma = sigma

    def match(self, sketch: np.ndarray, candidates: list[dict]) -> list[dict]:
        """
        Compute Euclidean distance between sketch.smooth and each window.smooth.

        Keeps the top 20% of candidates (lowest distance = best match).
        Adds 'euclidean_score' and 'score' keys to each returned candidate.
        Returns a new sorted list; never modifies inputs in-place.
        """
        from scipy.ndimage import gaussian_filter1d

        # Smooth the sketch with the requested sigma
        sketch_smooth = gaussian_filter1d(sketch.astype(np.float64), sigma=self.sigma)

        scored: list[dict] = []
        for cand in candidates:
            # If sigma is default 2.0, use precomputed smooth. Otherwise dynamically smooth the raw normalized curve.
            if abs(self.sigma - 2.0) < 1e-5:
                window_smooth = cand["smooth"]
            else:
                window_smooth = gaussian_filter1d(cand["norm"].astype(np.float64), sigma=self.sigma)
                
            dist = float(np.linalg.norm(sketch_smooth - window_smooth))
            entry = dict(cand)          # shallow copy — never modify in-place
            entry["euclidean_score"] = dist
            entry["score"] = dist
            scored.append(entry)

        # Sort ascending (lowest distance = best)
        scored.sort(key=lambda x: x["euclidean_score"])

        # Keep top 20%
        keep_n = max(1, int(len(scored) * 0.20))
        return scored[:keep_n]
