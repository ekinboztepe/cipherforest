"""Oblivious decision trees.

Every node on a level shares one (feature, threshold) test, so a sample is routed
by exactly D comparisons and the comparison bits are the leaf index.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


def bits_to_index(bits: BoolArray) -> IntArray:
    """(n, D) comparison bits -> (n,) leaf indices. Level m has weight 2**m."""
    weights = np.left_shift(1, np.arange(bits.shape[1], dtype=np.int64))
    return bits.astype(np.int64) @ weights


@dataclass(frozen=True)
class ObliviousTree:
    features: IntArray  # (D,), a feature may appear on several levels
    thresholds: FloatArray  # (D,), bit m is x[features[m]] > thresholds[m]
    leaf_values: FloatArray  # (2**D,)

    def __post_init__(self) -> None:
        depth = self.features.shape[0]
        if self.features.ndim != 1 or self.thresholds.shape != (depth,):
            raise ValueError("features and thresholds must both have shape (D,)")
        if self.leaf_values.shape != (1 << depth,):
            raise ValueError(f"expected {1 << depth} leaf values, got {self.leaf_values.shape}")

    @property
    def depth(self) -> int:
        return int(self.features.shape[0])

    def bits(self, x: FloatArray) -> BoolArray:
        return x[:, self.features] > self.thresholds

    def leaf_index(self, x: FloatArray) -> IntArray:
        return bits_to_index(self.bits(x))

    def predict(self, x: FloatArray) -> FloatArray:
        return self.leaf_values[self.leaf_index(x)]


@dataclass(frozen=True)
class ObliviousEnsemble:
    """f(x) = bias + sum of the tree outputs."""

    trees: Sequence[ObliviousTree]
    bias: float = 0.0

    def predict(self, x: FloatArray) -> FloatArray:
        out = np.full(x.shape[0], self.bias, dtype=np.float64)
        for tree in self.trees:
            out += tree.predict(x)
        return out
