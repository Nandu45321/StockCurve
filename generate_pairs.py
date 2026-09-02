"""
generate_pairs.py — Build training pairs for the NN matcher.

Pair types:
  Positive     (label=1): same symbol, date_end within 60 days
  Negative     (label=0): different sectors, randomly sampled
  Hard negative(label=0): different symbols, but euclidean distance
                           in bottom 10% (looks similar to euclidean)

Output: data/pairs.npy — list of dicts:
  { "curve_a": np.ndarray(50,), "curve_b": np.ndarray(50,), "label": int }

Run: python generate_pairs.py
"""

import numpy as np
from datetime import datetime
import random

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WINDOWS_PATH   = "data/windows.npy"
PAIRS_PATH     = "data/pairs.npy"
TARGET_POS     = 5000
TARGET_NEG     = 5000
TARGET_HARD    = 2000
DATE_WINDOW    = 60   # days — max gap between positive pair date_ends
HARD_NEG_PCT   = 0.10 # bottom 10% euclidean distances = "hard"
RANDOM_SEED    = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """L2 distance between two norm vectors."""
    return float(np.sqrt(np.sum((a - b) ** 2)))


def date_gap_days(d1: str, d2: str) -> int:
    """Return absolute calendar day difference between two YYYY-MM-DD strings."""
    fmt = "%Y-%m-%d"
    return abs((datetime.strptime(d1, fmt) - datetime.strptime(d2, fmt)).days)


def make_pair(a: dict, b: dict, label: int) -> dict:
    """Create a flat pair dict from two window dicts."""
    return {
        "curve_a": a["norm"].copy(),
        "curve_b": b["norm"].copy(),
        "label":   label,
    }


# ---------------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------------

def build_positive_pairs(windows: list[dict], target: int, rng: random.Random) -> list[dict]:
    """Same-symbol windows whose date_end values are within DATE_WINDOW days."""
    by_symbol: dict[str, list[dict]] = {}
    for w in windows:
        by_symbol.setdefault(w["symbol"], []).append(w)

    # Collect all valid candidate pairs per symbol first
    candidates: list[tuple[dict, dict]] = []
    for sym, wins in by_symbol.items():
        if len(wins) < 2:
            continue
        for i in range(len(wins)):
            for j in range(i + 1, len(wins)):
                if date_gap_days(wins[i]["date_end"], wins[j]["date_end"]) <= DATE_WINDOW:
                    candidates.append((wins[i], wins[j]))

    rng.shuffle(candidates)
    chosen = candidates[:target]
    return [make_pair(a, b, 1) for a, b in chosen]


def build_negative_pairs(windows: list[dict], target: int, rng: random.Random) -> list[dict]:
    """Random pairs from DIFFERENT sectors."""
    by_sector: dict[str, list[dict]] = {}
    for w in windows:
        sector = w.get("sector", "Unknown")
        by_sector.setdefault(sector, []).append(w)

    sectors = list(by_sector.keys())
    if len(sectors) < 2:
        # Fallback: different symbols
        pairs = []
        wins = windows[:]
        for _ in range(target):
            a, b = rng.sample(wins, 2)
            if a["symbol"] != b["symbol"]:
                pairs.append(make_pair(a, b, 0))
        return pairs

    pairs: list[dict] = []
    attempts = 0
    while len(pairs) < target and attempts < target * 10:
        attempts += 1
        s1, s2 = rng.sample(sectors, 2)
        a = rng.choice(by_sector[s1])
        b = rng.choice(by_sector[s2])
        pairs.append(make_pair(a, b, 0))

    return pairs


def build_hard_negative_pairs(windows: list[dict], target: int, rng: random.Random) -> list[dict]:
    """Different-symbol pairs whose euclidean distance is in the bottom HARD_NEG_PCT."""
    # Sample a tractable pool of cross-symbol pairs and compute distances.
    # Full O(N^2) is too slow for large N — sample ~50k random cross-symbol pairs instead.
    pool_size = max(target * 25, 50_000)
    pool: list[tuple[float, dict, dict]] = []

    wins = windows[:]
    n = len(wins)
    attempts = 0
    while len(pool) < pool_size and attempts < pool_size * 3:
        attempts += 1
        i = rng.randrange(n)
        j = rng.randrange(n)
        if wins[i]["symbol"] == wins[j]["symbol"]:
            continue
        dist = euclidean(wins[i]["norm"], wins[j]["norm"])
        pool.append((dist, wins[i], wins[j]))

    if not pool:
        print("[WARN] Could not build hard negative pool.")
        return []

    pool.sort(key=lambda t: t[0])
    cutoff_idx = max(1, int(len(pool) * HARD_NEG_PCT))
    hard_pool  = pool[:cutoff_idx]

    rng.shuffle(hard_pool)
    chosen = hard_pool[:target]
    return [make_pair(a, b, 0) for _, a, b in chosen]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Load windows, generate pairs, save to data/pairs.npy."""
    rng = random.Random(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print(f"[INFO] Loading {WINDOWS_PATH} ...")
    raw     = np.load(WINDOWS_PATH, allow_pickle=True)
    windows = list(raw)
    print(f"[INFO] Loaded {len(windows)} windows from {len(set(w['symbol'] for w in windows))} symbols")

    print("[INFO] Building positive pairs ...")
    pos_pairs  = build_positive_pairs(windows, TARGET_POS, rng)
    print(f"       {len(pos_pairs)} positive pairs")

    print("[INFO] Building negative pairs ...")
    neg_pairs  = build_negative_pairs(windows, TARGET_NEG, rng)
    print(f"       {len(neg_pairs)} negative pairs")

    print("[INFO] Building hard negative pairs (this may take ~30s) ...")
    hard_pairs = build_hard_negative_pairs(windows, TARGET_HARD, rng)
    print(f"       {len(hard_pairs)} hard negative pairs")

    all_pairs = pos_pairs + neg_pairs + hard_pairs
    rng.shuffle(all_pairs)

    np.save(PAIRS_PATH, np.array(all_pairs, dtype=object), allow_pickle=True)

    print()
    print("=" * 40)
    print(f"  Positive pairs      : {len(pos_pairs):>6}")
    print(f"  Negative pairs      : {len(neg_pairs):>6}")
    print(f"  Hard negative pairs : {len(hard_pairs):>6}")
    print(f"  Total saved         : {len(all_pairs):>6}")
    print(f"  Output              : {PAIRS_PATH}")
    print("=" * 40)


if __name__ == "__main__":
    main()
