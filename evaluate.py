"""
evaluate.py — Offline evaluation of all shape matchers.

Protocol:
  1. Hold out 100 random windows as test sketches.
  2. Remove those windows from the candidate pool so a sketch cannot match itself.
  3. For each test sketch, run each matcher against the full candidate pool.
  4. Record top-1 and top-5 same-stock accuracy, mean MAE of top-1, and avg latency.
  5. Print a formatted table and save it to data/evaluation_results.txt.

Run: python evaluate.py
"""

import os
import sys
import time
import random
import numpy as np

# Make matchers importable from the project root
sys.path.insert(0, os.path.dirname(__file__))

from matchers.euclidean import EuclideanMatcher
from matchers.fourier   import FourierMatcher
from matchers.base      import vertical_mae

WINDOWS_PATH = "data/windows.npy"
RESULTS_PATH = "data/evaluation_results.txt"
N_TEST       = 100
RANDOM_SEED  = 99
TOP_K        = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def score_all(matcher, sketch: np.ndarray, candidates: list[dict]) -> list[dict]:
    """
    Run matcher on ALL candidates and return the full sorted list.

    EuclideanMatcher internally truncates to 20% (pipeline behaviour).
    For fair evaluation we need all scores, so we replicate its distance
    computation without the cut.
    """
    name = matcher.name

    if name == "Euclidean":
        from scipy.ndimage import gaussian_filter1d
        sketch_smooth = gaussian_filter1d(sketch.astype(np.float64), sigma=2.0)
        scored = []
        for cand in candidates:
            dist = float(np.linalg.norm(sketch_smooth - cand["smooth"]))
            entry = dict(cand)
            entry["score"] = dist
            scored.append(entry)
        scored.sort(key=lambda x: x["score"])
        return scored

    elif name in ("Fourier", "Neural Net"):
        return matcher.match(sketch, candidates)

    else:
        return matcher.match(sketch, candidates)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Load windows, evaluate matchers, print and save results table."""
    rng = random.Random(RANDOM_SEED)

    print(f"[INFO] Loading {WINDOWS_PATH} ...")
    raw     = np.load(WINDOWS_PATH, allow_pickle=True)
    windows = list(raw)
    print(f"[INFO] {len(windows)} windows from "
          f"{len(set(w['symbol'] for w in windows))} symbols")

    # -- Hold-out 100 test sketches ------------------------------------------
    rng.shuffle(windows)
    test_sketches = windows[:N_TEST]
    test_ids      = set(id(w) for w in test_sketches)
    candidates    = [w for w in windows if id(w) not in test_ids]
    print(f"[INFO] Test sketches: {len(test_sketches)}  |  Candidate pool: {len(candidates)}")

    # -- Instantiate matchers -------------------------------------------------
    matchers = [EuclideanMatcher(), FourierMatcher()]

    try:
        from matchers.nn import NeuralMatcher
        matchers.append(NeuralMatcher())
        print("[INFO] NeuralMatcher loaded")
    except FileNotFoundError as e:
        print(f"[WARN] Skipping NeuralMatcher — {e}")

    # Also evaluate a Blended pseudo-matcher using a simple average of scores.
    # We record it separately after the loop.

    # -- Evaluate single matchers ---------------------------------------------
    # Results: { matcher_name -> {"top1": int, "top5": int, "maes": [float], "latencies": [float]} }
    stats: dict[str, dict] = {
        m.name: {"top1": 0, "top5": 0, "maes": [], "latencies": []}
        for m in matchers
    }

    # For Blended we need per-sketch all scores — collect them here.
    blended_inputs: list[tuple[np.ndarray, str, list[dict]]] = []  # (sketch, symbol, candidates)

    for i, sketch_win in enumerate(test_sketches):
        sketch = sketch_win["norm"]   # z-normalized 50-point array
        symbol = sketch_win["symbol"]

        matcher_results: dict[str, list[dict]] = {}

        for matcher in matchers:
            t0      = time.perf_counter()
            results = score_all(matcher, sketch, candidates)
            elapsed = (time.perf_counter() - t0) * 1000  # ms

            top5_symbols = [r["symbol"] for r in results[:TOP_K]]
            matcher_results[matcher.name] = results

            stats[matcher.name]["latencies"].append(elapsed)
            if top5_symbols and top5_symbols[0] == symbol:
                stats[matcher.name]["top1"] += 1
            if symbol in top5_symbols:
                stats[matcher.name]["top5"] += 1

            # MAE of the top-1 result vs the sketch
            if results:
                mae = vertical_mae(sketch, results[0]["norm"])
                stats[matcher.name]["maes"].append(mae)

        blended_inputs.append((sketch, symbol, matcher_results))

        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{N_TEST} sketches evaluated")

    # -- Evaluate Blended (simple Euclidean+Fourier score average) -----------
    blended_stats = {"top1": 0, "top5": 0, "maes": [], "latencies": []}

    eu_name = "Euclidean"
    fo_name = "Fourier"
    has_eu  = eu_name in stats
    has_fo  = fo_name in stats

    if has_eu and has_fo:
        for sketch, symbol, m_results in blended_inputs:
            t0 = time.perf_counter()

            eu_res = m_results.get(eu_name, [])
            fo_res = m_results.get(fo_name, [])

            # Build score map for each matcher
            eu_map = {r["symbol"]: r["score"] for r in eu_res}
            fo_map = {r["symbol"]: r["score"] for r in fo_res}

            all_syms = list({r["symbol"] for r in eu_res + fo_res})
            blended: list[tuple[float, str, dict]] = []
            for r in eu_res:
                sym = r["symbol"]
                eu_s = eu_map.get(sym, 9e9)
                fo_s = fo_map.get(sym, 9e9)
                blended.append(((eu_s + fo_s) / 2.0, sym, r))

            blended.sort(key=lambda x: x[0])
            elapsed = (time.perf_counter() - t0) * 1000

            # Deduplicate
            seen: set[str] = set()
            top5: list[tuple[float, str, dict]] = []
            for score, sym, r in blended:
                if sym not in seen:
                    seen.add(sym)
                    top5.append((score, sym, r))
                if len(top5) == TOP_K:
                    break

            top5_symbols = [sym for _, sym, _ in top5]
            blended_stats["latencies"].append(elapsed)
            if top5_symbols and top5_symbols[0] == symbol:
                blended_stats["top1"] += 1
            if symbol in top5_symbols:
                blended_stats["top5"] += 1
            if top5:
                mae = vertical_mae(sketch, top5[0][2]["norm"])
                blended_stats["maes"].append(mae)

    # -- Build result table ---------------------------------------------------
    header  = (
        f"{'Matcher':<14}  {'Top-1 Acc':>10}  {'Top-5 Acc':>10}"
        f"  {'Mean MAE':>9}  {'Avg Latency':>12}"
    )
    divider = "-" * len(header)
    rows    = []

    def fmt_row(name: str, s: dict) -> str:
        """Format one row of the results table."""
        n       = N_TEST
        top1    = s["top1"] / n * 100
        top5    = s["top5"] / n * 100
        mean_mae = sum(s["maes"]) / len(s["maes"]) if s["maes"] else float("nan")
        avg_ms  = sum(s["latencies"]) / len(s["latencies"]) if s["latencies"] else 0.0
        return (
            f"{name:<14}  {top1:>9.1f}%  {top5:>9.1f}%"
            f"  {mean_mae:>7.3f}s  {avg_ms:>10.1f}ms"
        )

    for m in matchers:
        rows.append(fmt_row(m.name, stats[m.name]))

    if has_eu and has_fo:
        rows.append(fmt_row("Blended", blended_stats))

    table_lines = [
        "",
        "=" * len(header),
        "  Evaluation Results",
        f"  N test sketches = {N_TEST}  |  Candidate pool = {len(candidates)}",
        "=" * len(header),
        header,
        divider,
        *rows,
        divider,
        "",
    ]
    table = "\n".join(table_lines)

    print(table)

    with open(RESULTS_PATH, "w") as f:
        f.write(table)
    print(f"[INFO] Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
