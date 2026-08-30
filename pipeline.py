"""
pipeline.py — Three-stage search pipeline with SSE streaming.

Stage 1: Hard filters (cap + std thresholds)
Stage 2: Smoothed Euclidean (via EuclideanMatcher)
Stage 3: Fourier matching (via FourierMatcher)
Convergence: sharpness-weighted blend → top 5 results

Each stage yields an SSE event dict. The caller (main.py) is responsible
for serializing and streaming these to the frontend.
"""

import math
import json
import numpy as np
from typing import AsyncGenerator

from matchers.euclidean import EuclideanMatcher
from matchers.fourier import FourierMatcher


# ---------------------------------------------------------------------------
# Canonical normalization helpers (from AGENTS.md — copied verbatim)
# ---------------------------------------------------------------------------

def resample(arr: np.ndarray, target: int = 50) -> np.ndarray:
    """Resample arr to `target` points using linear interpolation."""
    x_old = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, target)
    return np.interp(x_new, x_old, arr)


def znorm(arr: np.ndarray) -> np.ndarray:
    """Z-normalize arr to zero mean and unit std (epsilon prevents div-by-zero)."""
    mean = arr.mean()
    std = arr.std() + 1e-8
    return (arr - mean) / std


def smooth(arr: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Apply Gaussian smoothing with given sigma."""
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(arr, sigma=sigma)


def fingerprint(arr: np.ndarray, k: int = 8) -> np.ndarray:
    """Return first k magnitudes of the FFT of arr."""
    return np.abs(np.fft.fft(arr))[:k]


def sharpness(arr: np.ndarray) -> float:
    """Compute second-order difference mean as a sharpness metric."""
    return float(np.mean(np.abs(np.diff(arr, n=2))))


def sigmoid(x: float) -> float:
    """Standard sigmoid function."""
    return 1 / (1 + math.exp(-x))


# ---------------------------------------------------------------------------
# Stage 1 — Hard filters
# ---------------------------------------------------------------------------

def stage1_filter(
    windows: list[dict],
    filters: dict,
    std_thresh_low: float = 0.01,
    std_thresh_high: float = 3.0,
) -> list[dict]:
    """
    Apply hard filters to the full window list.

    Filters by cap category (large/mid/small booleans) and std thresholds
    to remove flat or pure-noise windows.
    """
    cap_filter = {
        "large": filters.get("large", True),
        "mid":   filters.get("mid", True),
        "small": filters.get("small", False),
    }

    candidates = []
    for w in windows:
        # Cap filter
        if not cap_filter.get(w["cap"], False):
            continue
        # Std filter: kick flat and pure-noise windows
        if not (std_thresh_low < w["std"] < std_thresh_high):
            continue
        candidates.append(w)

    return candidates


# ---------------------------------------------------------------------------
# Convergence layer
# ---------------------------------------------------------------------------

def convergence(
    candidates: list[dict],
    sketch: np.ndarray,
    meta_map: dict[str, dict],
) -> list[dict]:
    """
    Blend euclidean_score and fourier_score using sharpness-weighted alpha.

    alpha = sigmoid(sharpness(sketch) * 5)
    final_score = alpha * euclidean_score + (1 - alpha) * fourier_score

    Returns top 5 enriched result dicts.
    """
    sharp = sharpness(sketch)
    alpha = sigmoid(sharp * 5)

    results = []
    for cand in candidates:
        eu = cand.get("euclidean_score", 0.0)
        fo = cand.get("fourier_score", 0.0)
        final = alpha * eu + (1 - alpha) * fo

        symbol = cand["symbol"]
        meta = meta_map.get(symbol, {})

        results.append({
            "symbol": symbol,
            "name": meta.get("name", symbol),
            "date_end": cand["date_end"],
            "score": round(final, 6),
            "curve": cand["norm"].tolist(),
            "matcher_scores": {
                "Euclidean": round(eu, 6),
                "Fourier": round(fo, 6),
            },
        })

    results.sort(key=lambda x: x["score"])
    return results[:5]


# ---------------------------------------------------------------------------
# Main pipeline — async generator yielding SSE event dicts
# ---------------------------------------------------------------------------

async def run_pipeline(
    sketch: np.ndarray,
    filters: dict,
    all_windows: list[dict],
    meta_map: dict[str, dict],
    search_id: str,
) -> AsyncGenerator[dict, None]:
    """
    Run the full 3-stage search pipeline and yield SSE event dicts.

    Yields one dict per stage. The caller must serialize to SSE format.

    Args:
        sketch:      z-normalized 50-point array from the user's drawing
        filters:     dict with keys large/mid/small (bools) and window_days (int)
        all_windows: preloaded list of window dicts from data/windows.npy
        meta_map:    dict mapping symbol -> metadata dict
        search_id:   unique search ID for research logging
    """
    total = len(all_windows)

    # ------------------------------------------------------------------
    # Stage 1 — Hard filters
    # ------------------------------------------------------------------
    candidates = stage1_filter(all_windows, filters)
    stage1_count = len(candidates)

    yield {
        "stage": 1,
        "label": "Hard filters",
        "status": "done",
        "remaining": stage1_count,
        "total": total,
        "results": [],
    }

    # ------------------------------------------------------------------
    # Stage 2 — Smoothed Euclidean
    # ------------------------------------------------------------------
    smoothing = filters.get("smoothing", 2.0)
    euclidean_matcher = EuclideanMatcher(sigma=smoothing)
    candidates = euclidean_matcher.match(sketch, candidates)
    stage2_count = len(candidates)

    yield {
        "stage": 2,
        "label": "Smoothed Euclidean",
        "status": "done",
        "remaining": stage2_count,
        "total": total,
        "results": [],
    }

    # ------------------------------------------------------------------
    # Stage 3a — Fourier
    # ------------------------------------------------------------------
    fourier_matcher = FourierMatcher()
    candidates = fourier_matcher.match(sketch, candidates)
    stage3_count = len(candidates)

    yield {
        "stage": 3,
        "label": "Fourier",
        "status": "done",
        "remaining": stage3_count,
        "total": total,
        "results": [],
    }

    # ------------------------------------------------------------------
    # Convergence layer → top 5
    # ------------------------------------------------------------------
    sharp = sharpness(sketch)
    alpha = sigmoid(sharp * 5)

    top5 = convergence(candidates, sketch, meta_map)

    # Research log (stdout as required by AGENTS.md)
    euclidean_top5 = [c["symbol"] for c in sorted(
        candidates, key=lambda x: x.get("euclidean_score", 9e9))[:5]]
    fourier_top5 = [c["symbol"] for c in sorted(
        candidates, key=lambda x: x.get("fourier_score", 9e9))[:5]]
    final_top5 = [r["symbol"] for r in top5]

    research_log = {
        "search_id": search_id,
        "sharpness": round(sharp, 6),
        "alpha": round(alpha, 6),
        "stage1_count": stage1_count,
        "stage2_count": stage2_count,
        "stage3_count": stage3_count,
        "euclidean_top5": euclidean_top5,
        "fourier_top5": fourier_top5,
        "final_top5": final_top5,
    }
    print(f"[RESEARCH] {json.dumps(research_log)}", flush=True)

    yield {
        "stage": 4,
        "label": "Final results",
        "status": "done",
        "remaining": len(top5),
        "total": total,
        "results": top5,
    }
