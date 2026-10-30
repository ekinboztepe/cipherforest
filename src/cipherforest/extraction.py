"""Model extraction by a client that only holds client.json and a query budget.

The client knows the quantiser edges, so it can hit every input code of a
feature on purpose: code k is the edge e_k itself, the last code anything above
the last edge. Every threshold is one of these edges and the model is constant
between thresholds, so sweeping one feature while the others stay put changes
the output exactly at thresholds the current point makes visible. No false
positives are possible, only misses, which more starting points reduce.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from cipherforest.oblivious import FloatArray
from cipherforest.quantize import FeatureQuantizer, QuantizedModel

# (n, d) inputs -> (n,) predictions and (n, d) SHAP values
Oracle = Callable[[FloatArray], tuple[FloatArray, FloatArray]]


def plaintext_oracle(model: QuantizedModel) -> Oracle:
    """What the client decrypts, computed in the clear; the circuit matches it bit for bit."""

    def query(x: FloatArray) -> tuple[FloatArray, FloatArray]:
        return model.decode(model.simulate(model.quantizer.transform(x)))

    return query


@dataclass
class CountingOracle:
    oracle: Oracle
    queries: int = field(default=0, init=False)

    def __call__(self, x: FloatArray) -> tuple[FloatArray, FloatArray]:
        self.queries += x.shape[0]
        return self.oracle(x)


def code_values(edges: FloatArray) -> FloatArray:
    """One input per code: value i has exactly i edges strictly below it."""
    return np.append(edges, edges[-1] + 1.0) if edges.size else np.zeros(1)


def true_thresholds(model: QuantizedModel) -> set[tuple[int, int]]:
    """(feature, k) for every split x > e_k in the model."""
    return {(int(f), int(t)) for tree in model.trees for f, t in zip(tree.features, tree.thresholds, strict=True)}


def find_thresholds(
    oracle: Oracle,
    quantizer: FeatureQuantizer,
    n_starts: int,
    rng: np.random.Generator,
    use_shap: bool = True,
) -> set[tuple[int, int]]:
    """Sweeps every feature through all its codes from n_starts random points.

    Costs n_starts * sum(len(edges) + 1) queries. With use_shap=False only the
    prediction is compared, as against an API without explanations.
    """
    values = [code_values(e) for e in quantizer.edges]
    found: set[tuple[int, int]] = set()
    for _ in range(n_starts):
        start = np.array([rng.choice(v) for v in values])
        for f, v in enumerate(values):
            if v.size < 2:
                continue
            rows = np.tile(start, (v.size, 1))
            rows[:, f] = v
            pred, shap = oracle(rows)
            out = np.column_stack([pred, shap]) if use_shap else pred[:, None]
            changed = np.any(out[1:] != out[:-1], axis=1)
            # codes k and k+1 differ only in the bit of x > e_k
            found.update((f, int(k)) for k in np.flatnonzero(changed))
    return found


def recall(found: set[tuple[int, int]], truth: set[tuple[int, int]]) -> float:
    return len(found & truth) / len(truth) if truth else 1.0
