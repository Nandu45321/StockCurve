"""
matchers/fourier.py — Stage 3a: Fourier magnitude matcher.

Compares the first 8 FFT magnitude components of the normalized sketch
against each window's normalized curve. Lower score = better match.
"""

import numpy as np
from .base import BaseMatcher


class FourierMatcher(BaseMatcher):
    """FFT magnitude matcher using first 8 frequency components (Stage 3a)."""

    name = "Fourier"

    def match(self, sketch: np.ndarray, candidates: list[dict]) -> list[dict]:
        """
        Compute Euclidean distance between FFT fingerprints of sketch and windows.

        Uses |FFT(arr)|[:8] as the fingerprint (canonical from AGENTS.md).
        Adds 'fourier_score' and 'score' keys. Returns new sorted list.
        """
        # FFT fingerprint of the sketch (first 8 magnitudes)
        fft_sketch = np.abs(np.fft.fft(sketch.astype(np.float64)))[:8]

        scored: list[dict] = []
        for cand in candidates:
            fft_window = np.abs(np.fft.fft(cand["norm"].astype(np.float64)))[:8]
            dist = float(np.linalg.norm(fft_sketch - fft_window))
            entry = dict(cand)          # shallow copy — never modify in-place
            entry["fourier_score"] = dist
            entry["score"] = dist
            scored.append(entry)

        # Sort ascending (lowest distance = best)
        scored.sort(key=lambda x: x["fourier_score"])
        return scored
