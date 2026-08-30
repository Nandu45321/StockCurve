"""
matchers/nn.py — Stage 3b: Neural network matcher stub.

NOT implemented yet. Will be a learned embedding-based matcher for the
research comparison paper. Raises NotImplementedError as required by AGENTS.md.
"""

import numpy as np
from .base import BaseMatcher


class NNMatcher(BaseMatcher):
    """Neural network embedding matcher — stub only (Stage 3b, future work)."""

    name = "NeuralNet"

    def match(self, sketch: np.ndarray, candidates: list[dict]) -> list[dict]:
        """
        Stub: will match via learned embeddings. Not yet implemented.

        Raises:
            NotImplementedError: always, until Stage 3b is built.
        """
        raise NotImplementedError(
            "NNMatcher is not implemented yet. "
            "This will use learned embeddings for the research comparison paper."
        )
