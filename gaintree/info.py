"""Information-theory helpers: information, entropy and information gain.

All logs are base 2, so entropy is measured in bits:
    - a fair coin has entropy 1 bit
    - a pure node (one class only) has entropy 0
"""

from __future__ import annotations

import numpy as np


def surprise(p: float) -> float:
    """Information (surprise) carried by an event of probability p: log2(1/p)."""
    if p <= 0 or p > 1:
        raise ValueError("p must be in (0, 1]")
    return float(np.log2(1.0 / p))


def entropy_of_counts(counts: np.ndarray) -> np.ndarray:
    """Entropy H = -sum p * log2(p) computed from class counts.

    `counts` can be 1-D (one distribution) or 2-D (one distribution per row),
    which lets us score every candidate threshold of a feature in one go.
    Convention: 0 * log2(0) = 0.
    """
    counts = np.asarray(counts, dtype=float)
    totals = counts.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(totals > 0, counts / totals, 0.0)
        logp = np.where(p > 0, np.log2(p), 0.0)
    return -(p * logp).sum(axis=-1) + 0.0  # "+ 0.0" turns -0.0 into 0.0


def shannon_entropy(y) -> float:
    """Entropy of a label vector."""
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    return float(entropy_of_counts(counts))


def gain_of_split(y_parent, children) -> float:
    """IG(T, a) = H(T) - sum_k (|T_k| / |T|) * H(T_k).

    The children entropies are weighted by their share of the samples
    ("averaged by the class importance" in the course slides).
    """
    y_parent = np.asarray(y_parent)
    n = y_parent.size
    if n == 0:
        return 0.0
    weighted = sum(len(c) / n * shannon_entropy(c) for c in children)
    return shannon_entropy(y_parent) - weighted
