"""Exact interventional SHAP for oblivious trees, precomputed per leaf.

With v_x(S) = mean_b f(x_S, b_~S), replacing the features outside S by a
background row only replaces the bits of the levels that test them. So v_x(S),
and every Shapley value, depends on x only through its D comparison bits: one
SHAP vector per leaf index, computed once in the clear.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import factorial

import numpy as np

from cipherforest.oblivious import BoolArray, FloatArray, IntArray, ObliviousEnsemble, ObliviousTree, bits_to_index


@dataclass(frozen=True)
class ShapTable:
    players: IntArray  # (k,) distinct features the tree tests; every other feature gets 0
    values: FloatArray  # (2**D, k), row i holds the SHAP values of inputs that land in leaf i
    expected_value: float  # mean prediction over the background set


def _all_bit_vectors(depth: int) -> BoolArray:
    """Row i holds the comparison bits of leaf i."""
    idx = np.arange(1 << depth, dtype=np.int64)
    return ((idx[:, None] >> np.arange(depth, dtype=np.int64)) & 1).astype(bool)


def build_shap_table(tree: ObliviousTree, background: FloatArray) -> ShapTable:
    """O(2**D * 2**k * |B| * D) with k <= D, which is nothing at the depths a TFHE lookup allows."""
    depth = tree.depth
    players = np.unique(tree.features)
    k = players.shape[0]
    level_player = np.searchsorted(players, tree.features)

    x_bits = _all_bit_vectors(depth)
    bg_bits = tree.bits(background)

    # v[s, i]: value of coalition s (a bitmask over players) for an input in leaf i
    v = np.empty((1 << k, 1 << depth), dtype=np.float64)
    for mask in range(1 << k):
        from_x = ((mask >> level_player) & 1).astype(bool)
        hybrid = np.where(from_x, x_bits[:, None, :], bg_bits[None, :, :])
        leaves = bits_to_index(hybrid.reshape(-1, depth)).reshape(hybrid.shape[:2])
        v[mask] = tree.leaf_values[leaves].mean(axis=1)

    values = np.zeros((1 << depth, k), dtype=np.float64)
    weights = [factorial(s) * factorial(k - s - 1) / factorial(k) for s in range(k)]
    for j in range(k):
        bit = 1 << j
        for mask in range(1 << k):
            if mask & bit:
                continue
            values[:, j] += weights[mask.bit_count()] * (v[mask | bit] - v[mask])

    return ShapTable(players=players, values=values, expected_value=float(v[0, 0]))


def explain(tree: ObliviousTree, table: ShapTable, x: FloatArray, n_features: int) -> FloatArray:
    out = np.zeros((x.shape[0], n_features), dtype=np.float64)
    out[:, table.players] = table.values[tree.leaf_index(x)]
    return out


def explain_ensemble(
    ensemble: ObliviousEnsemble, tables: Sequence[ShapTable], x: FloatArray, n_features: int
) -> FloatArray:
    """SHAP is linear in the model, so per-tree values add up."""
    out = np.zeros((x.shape[0], n_features), dtype=np.float64)
    for tree, table in zip(ensemble.trees, tables, strict=True):
        out += explain(tree, table, x, n_features)
    return out
