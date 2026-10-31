"""Integer form of an oblivious ensemble, as the TFHE circuit consumes it.

Inputs are exact. The distiller only splits on quantiser edges, and a value is
encoded as the number of edges strictly below it, so x > e_k  <=>  code(x) > k.
The circuit routes every sample to the same leaf as the float model.

Outputs are bounded. SHAP values are scaled by s and rounded so that each leaf's
integers still sum to round(s * (leaf value - tree base)). The client rebuilds the
prediction as base + sum(SHAP), so it always agrees with the explanation and each
tree adds at most 1 / (2s) error.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from cipherforest.oblivious import FloatArray, IntArray, ObliviousEnsemble, bits_to_index
from cipherforest.shap_table import ShapTable


@dataclass(frozen=True)
class FeatureQuantizer:
    edges: tuple[FloatArray, ...]  # strictly increasing, one array per feature

    @classmethod
    def fit(cls, x: FloatArray, n_bins: int) -> FeatureQuantizer:
        """Distinct inner n_bins-quantiles of each training column."""
        qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
        return cls(tuple(np.unique(np.quantile(x[:, f], qs)) for f in range(x.shape[1])))

    @property
    def n_features(self) -> int:
        return len(self.edges)

    @property
    def input_bits(self) -> int:
        # codes run from 0 to len(edges)
        return max(int(e.size).bit_length() for e in self.edges)

    def transform(self, x: FloatArray) -> IntArray:
        codes = [np.searchsorted(e, x[:, f], side="left") for f, e in enumerate(self.edges)]
        return np.column_stack(codes).astype(np.int64)

    def threshold_index(self, feature: int, threshold: float) -> int:
        edges = self.edges[feature]
        k = int(np.searchsorted(edges, threshold, side="left"))
        if k == edges.size or edges[k] != threshold:
            raise ValueError(f"threshold {threshold!r} is not an edge of feature {feature}")
        return k


@dataclass(frozen=True)
class QuantizedTree:
    features: IntArray  # (D,)
    thresholds: IntArray  # (D,), bit m is code[features[m]] > thresholds[m]
    players: IntArray  # (k,) global feature of each SHAP column
    shap: IntArray  # (2**D, k) scaled and rounded

    def leaf_index(self, codes: IntArray) -> IntArray:
        return bits_to_index(codes[:, self.features] > self.thresholds)


@dataclass(frozen=True)
class QuantizedModel:
    quantizer: FeatureQuantizer
    trees: tuple[QuantizedTree, ...]
    scale: float
    base_value: float  # ensemble bias plus every tree's expected value
    output_bits: int  # signed width every summed output fits in

    def simulate(self, codes: IntArray) -> IntArray:
        """What the circuit computes, in the clear: summed integer SHAP, (n, d)."""
        out = np.zeros((codes.shape[0], self.quantizer.n_features), dtype=np.int64)
        for tree in self.trees:
            out[:, tree.players] += tree.shap[tree.leaf_index(codes)]
        return out

    def decode(self, shap_codes: IntArray) -> tuple[FloatArray, FloatArray]:
        shap = shap_codes / self.scale
        return self.base_value + shap.sum(axis=1), shap

    def to_json(self) -> str:
        return json.dumps(
            {
                "edges": [e.tolist() for e in self.quantizer.edges],
                "trees": [
                    {
                        "features": t.features.tolist(),
                        "thresholds": t.thresholds.tolist(),
                        "players": t.players.tolist(),
                        "shap": t.shap.tolist(),
                    }
                    for t in self.trees
                ],
                "scale": self.scale,
                "base_value": self.base_value,
                "output_bits": self.output_bits,
            }
        )

    @classmethod
    def from_json(cls, text: str) -> QuantizedModel:
        doc = json.loads(text)

        def ints(values: object) -> IntArray:
            return np.asarray(values, dtype=np.int64)

        trees = tuple(
            QuantizedTree(
                features=ints(t["features"]),
                thresholds=ints(t["thresholds"]),
                players=ints(t["players"]),
                shap=ints(t["shap"]).reshape(-1, len(t["players"])),
            )
            for t in doc["trees"]
        )
        return cls(
            quantizer=FeatureQuantizer(tuple(np.asarray(e, dtype=np.float64) for e in doc["edges"])),
            trees=trees,
            scale=float(doc["scale"]),
            base_value=float(doc["base_value"]),
            output_bits=int(doc["output_bits"]),
        )


def to_grid(model: QuantizedModel, low: FloatArray, high: FloatArray, bits: int) -> QuantizedModel:
    """The same model on a public grid of 2**bits even cells per feature, edges hidden.

    The client only learns the range and the bit width. Each threshold moves up
    to the next grid edge, so the model changes for inputs between the two.
    Up, not to the nearest: data often sit exactly on an edge (discrete
    features), and x = e must stay below the threshold.
    """
    inner = np.arange(1, 1 << bits) / (1 << bits)
    grid = FeatureQuantizer(tuple(np.unique(a + inner * (b - a)) if b > a else np.zeros(0) for a, b in zip(low, high, strict=True)))
    trees = []
    for tree in model.trees:
        edges = [model.quantizer.edges[f][t] for f, t in zip(tree.features, tree.thresholds, strict=True)]
        thresholds = [min(int(np.searchsorted(grid.edges[f], e)), grid.edges[f].size - 1) for f, e in zip(tree.features, edges, strict=True)]
        trees.append(replace(tree, thresholds=np.array(thresholds, dtype=np.int64)))
    return replace(model, quantizer=grid, trees=tuple(trees))


def _sum_preserving_round(values: FloatArray, totals: IntArray) -> IntArray:
    """Rounds each row to integers summing to totals, moving no entry by 1 or more.

    Floor everything, then hand the missing units to the largest fractional parts
    (or take them from the smallest when the remainder is negative).
    """
    floor = np.floor(values)
    frac = values - floor
    remainder = totals - floor.sum(axis=1).astype(np.int64)
    if np.any(np.abs(remainder) > values.shape[1]):
        raise ValueError("row totals are inconsistent with the values being rounded")
    rank_desc = np.argsort(np.argsort(-frac, axis=1, kind="stable"), axis=1, kind="stable")
    rank_asc = np.argsort(np.argsort(frac, axis=1, kind="stable"), axis=1, kind="stable")
    up = rank_desc < np.maximum(remainder, 0)[:, None]
    down = rank_asc < np.maximum(-remainder, 0)[:, None]
    return (floor + up - down).astype(np.int64)


def quantize_model(
    ensemble: ObliviousEnsemble,
    tables: Sequence[ShapTable],
    quantizer: FeatureQuantizer,
    output_bits: int = 8,
) -> QuantizedModel:
    """Picks the largest scale for which every summed output provably fits in output_bits."""
    limit = (1 << (output_bits - 1)) - 1
    d = quantizer.n_features
    magnitude = np.zeros(d)  # sum over trees of max |SHAP| per feature
    uses = np.zeros(d)  # trees in which the feature is a player; each may add one rounding unit
    for table in tables:
        magnitude[table.players] += np.abs(table.values).max(axis=0)
        uses[table.players] += 1
    active = magnitude > 0
    if np.any(uses[active] >= limit):
        raise ValueError(f"output_bits={output_bits} is too small for {int(uses.max())} trees")
    scale = float(np.min((limit - uses[active]) / magnitude[active])) if active.any() else 1.0

    trees = []
    for tree, table in zip(ensemble.trees, tables, strict=True):
        totals = np.rint((tree.leaf_values - table.expected_value) * scale).astype(np.int64)
        thresholds = [quantizer.threshold_index(int(f), float(t)) for f, t in zip(tree.features, tree.thresholds)]
        trees.append(
            QuantizedTree(
                features=tree.features.copy(),
                thresholds=np.array(thresholds, dtype=np.int64),
                players=table.players.copy(),
                shap=_sum_preserving_round(table.values * scale, totals),
            )
        )
    base_value = ensemble.bias + sum(table.expected_value for table in tables)
    return QuantizedModel(
        quantizer=quantizer, trees=tuple(trees), scale=scale, base_value=base_value, output_bits=output_bits
    )
