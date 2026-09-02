"""
matchers/__init__.py — Matcher registry.

To add a new matcher: create a file in matchers/, inherit BaseMatcher,
then append an instance to MATCHERS. Never touch pipeline.py to add a matcher.
"""

from .euclidean import EuclideanMatcher
from .fourier import FourierMatcher

# Registered matchers used by the pipeline (in stage order)
MATCHERS = [
    EuclideanMatcher(),
    FourierMatcher(),
]

try:
    from .nn import NeuralMatcher
    MATCHERS.append(NeuralMatcher())
except FileNotFoundError:
    pass

