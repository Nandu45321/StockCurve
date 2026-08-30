"""
matchers/__init__.py — Matcher registry.

To add a new matcher: create a file in matchers/, inherit BaseMatcher,
then append an instance to MATCHERS. Never touch pipeline.py to add a matcher.
"""

from .euclidean import EuclideanMatcher
from .fourier import FourierMatcher
from .nn import NNMatcher

# Registered matchers used by the pipeline (in stage order)
# NNMatcher is listed but will raise NotImplementedError until implemented.
MATCHERS = [
    EuclideanMatcher(),
    FourierMatcher(),
    # NNMatcher(),  # uncomment when Stage 3b is implemented
]
