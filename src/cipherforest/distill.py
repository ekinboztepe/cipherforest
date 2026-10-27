"""Greedy distillation of a teacher into a small oblivious ensemble.

Each level picks the (feature, threshold) that minimises squared error across all
current leaves at once. Candidate thresholds are the quantiser edges, which is what
keeps the encrypted comparisons exact (see quantize.py).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from cipherforest.oblivious import FloatArray, ObliviousEnsemble, ObliviousTree


def fit_oblivious_tree(x: FloatArray, y: FloatArray, depth: int, thresholds: Sequence[FloatArray]) -> ObliviousTree:
    """Leaves that no training row reaches get mean(y)."""
    n = x.shape[0]
    cell = np.zeros(n, dtype=np.int64)
    features = np.empty(depth, dtype=np.int64)
    chosen = np.empty(depth, dtype=np.float64)

    for level in range(depth):
        n_cells = 1 << level
        best_gain, best_feature, best_threshold = -np.inf, -1, 0.0
        for f, cand in enumerate(thresholds):
            if cand.size == 0:
                continue
            # bin b holds cand[b-1] < x <= cand[b], so splitting after bin k means x > cand[k]
            bins = np.searchsorted(cand, x[:, f], side="left")
            slots = cand.size + 1
            flat = cell * slots + bins
            counts = np.bincount(flat, minlength=n_cells * slots).reshape(n_cells, slots)
            sums = np.bincount(flat, weights=y, minlength=n_cells * slots).reshape(n_cells, slots)
            left_n = np.cumsum(counts, axis=1)[:, :-1]
            left_s = np.cumsum(sums, axis=1)[:, :-1]
            right_n = counts.sum(axis=1, keepdims=True) - left_n
            right_s = sums.sum(axis=1, keepdims=True) - left_s
            # min SSE == max sum over children of sum^2 / count
            with np.errstate(divide="ignore", invalid="ignore"):
                gain = np.where(left_n > 0, left_s**2 / left_n, 0.0) + np.where(right_n > 0, right_s**2 / right_n, 0.0)
            per_threshold = gain.sum(axis=0)
            k = int(np.argmax(per_threshold))
            if per_threshold[k] > best_gain:
                best_gain, best_feature, best_threshold = float(per_threshold[k]), f, float(cand[k])
        if best_feature < 0:
            raise ValueError("no candidate thresholds: every feature is constant")
        features[level] = best_feature
        chosen[level] = best_threshold
        cell += (x[:, best_feature] > best_threshold).astype(np.int64) << level

    counts = np.bincount(cell, minlength=1 << depth)
    sums = np.bincount(cell, weights=y, minlength=1 << depth)
    leaf_values = np.where(counts > 0, sums / np.maximum(counts, 1), y.mean())
    return ObliviousTree(features=features, thresholds=chosen, leaf_values=leaf_values)


def distill_ensemble(
    x: FloatArray, target: FloatArray, depth: int, n_trees: int, thresholds: Sequence[FloatArray]
) -> ObliviousEnsemble:
    """Residual boosting with learning rate 1; thresholds[f] are the candidate splits of feature f."""
    bias = float(target.mean())
    residual = target - bias
    trees: list[ObliviousTree] = []
    for _ in range(n_trees):
        tree = fit_oblivious_tree(x, residual, depth, thresholds)
        residual = residual - tree.predict(x)
        trees.append(tree)
    return ObliviousEnsemble(trees=trees, bias=bias)
