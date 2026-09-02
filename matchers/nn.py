"""
matchers/nn.py — Stage 3b: Neural network embedding matcher.

Loads the trained SiameseNet from data/siamese.pt and scores each candidate
by the L2 distance between its embedding and the sketch embedding.

Requires: python train.py has been run to produce data/siamese.pt
"""

import os
import numpy as np
import torch

from .base import BaseMatcher
from .siamese import SiameseNet

MODEL_PATH = os.path.join("data", "siamese.pt")


class NeuralMatcher(BaseMatcher):
    """Embedding-based matcher using a trained Siamese network."""

    name = "Neural Net"

    def __init__(self) -> None:
        """Load the trained SiameseNet weights from data/siamese.pt."""
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Train the model first: python train.py  (expected {MODEL_PATH})"
            )
        self.model = SiameseNet()
        self.model.load_state_dict(
            torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        )
        self.model.eval()

    def match(self, sketch: np.ndarray, candidates: list[dict]) -> list[dict]:
        """
        Score each candidate by L2 distance between embeddings.

        Args:
            sketch:     z-normalized 50-point numpy array
            candidates: list of window dicts (must have 'norm' field)

        Returns:
            New list of candidate dicts with 'score' added, sorted best-first.
        """
        with torch.no_grad():
            # Encode the sketch once: (1, 1, 50)
            sketch_t = torch.tensor(sketch, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            sketch_emb = self.model.encode(sketch_t)  # (1, 64)

            results = []
            for cand in candidates:
                curve_t = torch.tensor(
                    cand["norm"], dtype=torch.float32
                ).unsqueeze(0).unsqueeze(0)  # (1, 1, 50)
                cand_emb = self.model.encode(curve_t)  # (1, 64)

                dist = float(
                    torch.nn.functional.pairwise_distance(sketch_emb, cand_emb).item()
                )
                result = dict(cand)
                result["score"] = dist
                results.append(result)

        results.sort(key=lambda x: x["score"])
        return results

