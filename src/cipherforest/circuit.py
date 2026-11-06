"""Compiles a QuantizedModel into a TFHE circuit with Concrete.

Per tree: D bootstraps for the comparisons, a free weighted sum for the leaf
index, one bootstrapped lookup per SHAP player on that same index, and a free
scatter-add (clear 0/1 matrix) into the output vector. The server never computes
the prediction; the client gets it as base + sum(SHAP).

Two smaller outputs drop the explanation. "score" looks up each tree's leaf total
once and returns their sum, so the client gets the prediction only. "decision"
also compares that sum with a clear cut on the server and returns one bit, as an
approve/decline API would.

Outputs are hinted to the width the model guarantees, not the width the inputset
happened to reach: a rare input must not wrap around.

With noise_bound > 0 the circuit takes a second, clear argument from the server
and adds it to the outputs: no bootstrapping, only wider outputs.

Concrete only ships Linux and macOS wheels. On Windows use docker/Dockerfile.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from concrete import fhe

from cipherforest.oblivious import IntArray
from cipherforest.quantize import QuantizedModel


# Concrete's own default allows about 1e-5 error per lookup, so with a few hundred
# lookups roughly one request in 300 to 3000 decrypts wrong. These are Concrete-ML's
# settings: 2-3x slower, no wrong result seen in 7500 requests. Compression is free.
SAFE = fhe.Configuration(compress_evaluation_keys=True, compress_input_ciphertexts=True, p_error=2**-40)


OUTPUTS = ("shap", "score", "decision")
DECISION_BITS = 4  # signed width left for the final comparison


@dataclass(frozen=True)
class CompiledModel:
    circuit: fhe.Circuit
    active_features: IntArray  # global feature behind each output slot
    n_features: int
    noise_bound: int = 0  # the clear noise argument, if any, lies in [-bound, bound]
    output: str = "shap"
    cut: int | None = None  # decision: 1 means score > cut

    def expand(self, outputs: IntArray) -> IntArray:
        """(n, A) circuit outputs -> (n, d), zeros for features no tree tests."""
        outputs = np.atleast_2d(outputs)
        full = np.zeros((outputs.shape[0], self.n_features), dtype=np.int64)
        full[:, self.active_features] = outputs
        return full


def active_features(model: QuantizedModel) -> IntArray:
    return np.unique(np.concatenate([tree.players for tree in model.trees]))


def score_bound(model: QuantizedModel) -> int:
    """Largest |score| any input can reach: every partial sum stays within it."""
    return int(sum(np.abs(tree.shap.sum(axis=1)).max() for tree in model.trees))


def circuit_function(model: QuantizedModel, noise: bool = False, output: str = "shap", cut: int | None = None) -> Callable[..., Any]:
    if output not in OUTPUTS:
        raise ValueError(f"output must be one of {OUTPUTS}, not {output!r}")
    if output != "shap" and noise:
        raise ValueError("server noise is only implemented for SHAP outputs")
    if (output == "decision") != (cut is not None):
        raise ValueError("a decision needs a cut, and only a decision takes one")
    if output != "shap":
        return _score_function(model, output, cut)
    active = active_features(model)
    limit = (1 << (model.output_bits - 1)) - 1
    slot = {int(f): i for i, f in enumerate(active)}
    plan = []
    for tree in model.trees:
        weights = np.left_shift(1, np.arange(tree.features.size, dtype=np.int64))
        tables = fhe.LookupTable([fhe.LookupTable([int(v) for v in column]) for column in tree.shap.T])
        scatter = np.zeros((tree.players.size, active.size), dtype=np.int64)
        scatter[np.arange(tree.players.size), [slot[int(p)] for p in tree.players]] = 1
        plan.append((tree.features, tree.thresholds, weights, tables, scatter))

    # Concrete cannot lower the vectorised comparison on inputs this wide; one
    # comparison at a time works, but needs larger keys on narrow inputs
    scalar = model.quantizer.input_bits > 11

    def run(codes: Any) -> Any:
        total = None
        for features, thresholds, weights, tables, scatter in plan:
            if scalar:
                index = sum((codes[int(f)] > int(t)) * int(w) for f, t, w in zip(features, thresholds, weights, strict=True))
            else:
                index = np.sum((codes[features] > thresholds) * weights)
            fanned = index + np.zeros(scatter.shape[0], dtype=np.int64)  # one copy per player table
            contribution = tables[fanned] @ scatter
            total = contribution if total is None else total + contribution
            if not noise:  # with noise, noisy_inputs already reaches the widest values
                total = fhe.hint(total, can_store=np.array([-limit, limit]))
        return total

    def noisy(codes: Any, noise: Any) -> Any:
        return run(codes) + noise

    return noisy if noise else run


def _score_function(model: QuantizedModel, output: str, cut: int | None) -> Callable[..., Any]:
    bound = score_bound(model)
    reach = bound + abs(cut or 0) + 1  # |score - (cut + 1)|
    drop = max(0, reach.bit_length() + 1 - DECISION_BITS)
    plan = []
    for tree in model.trees:
        weights = np.left_shift(1, np.arange(tree.features.size, dtype=np.int64))
        plan.append((tree.features, tree.thresholds, weights, fhe.LookupTable([int(v) for v in tree.shap.sum(axis=1)])))
    scalar = model.quantizer.input_bits > 11

    def run(codes: Any) -> Any:
        score = None
        for features, thresholds, weights, table in plan:
            if scalar:
                index = sum((codes[int(f)] > int(t)) * int(w) for f, t, w in zip(features, thresholds, weights, strict=True))
            else:
                index = np.sum((codes[features] > thresholds) * weights)
            score = table[index] if score is None else score + table[index]
            score = fhe.hint(score, can_store=np.array([-bound, bound]))
        if cut is None:
            return score
        # score > cut  <=>  diff >= 0. Flooring diff to a multiple of 2**k keeps its sign,
        # so the comparison only needs the top bits: a narrow bootstrap, not a wide one.
        diff = fhe.hint(score - (cut + 1), can_store=np.array([-reach, reach]))
        if drop:
            diff = fhe.truncate_bit_pattern(diff, lsbs_to_remove=drop)
        return diff >= 0

    return run


def representative_inputs(model: QuantizedModel, n: int, seed: int = 0) -> list[IntArray]:
    """All-min and all-max codes plus random ones, so Concrete sizes every bit width right."""
    upper = np.array([e.size for e in model.quantizer.edges], dtype=np.int64)
    rng = np.random.default_rng(seed)
    return [np.zeros_like(upper), upper.copy()] + [rng.integers(0, upper + 1) for _ in range(n)]


def noisy_inputs(model: QuantizedModel, noise_bound: int, n: int, seed: int = 0) -> list[tuple[IntArray, IntArray]]:
    """Random noise, plus two cases that put every output at +-(limit + bound).

    Concrete sizes the outputs from what it sees, and random inputs rarely hit
    the widest sum. The extra noise values lie outside the bound; they only
    size the circuit.
    """
    active = active_features(model)
    rng = np.random.default_rng(seed)
    codes = representative_inputs(model, n, seed)
    cases = [(c, rng.integers(-noise_bound, noise_bound + 1, size=active.size)) for c in codes]
    widest = (1 << (model.output_bits - 1)) - 1 + noise_bound
    clean = model.simulate(codes[0][None])[0, active]
    cases += [(codes[0], sign * widest - clean) for sign in (1, -1)]
    return cases


def compile_model(
    model: QuantizedModel,
    inputset_size: int = 64,
    noise_bound: int = 0,
    configuration: fhe.Configuration | None = None,
    output: str = "shap",
    cut: int | None = None,
) -> CompiledModel:
    """configuration=None means SAFE; pass fhe.Configuration() for Concrete's defaults.

    output is "shap" (prediction and explanation), "score" (prediction only) or
    "decision" (one bit: score > cut; see QuantizedModel.cut_for).
    """
    configuration = SAFE if configuration is None else configuration
    if output != "shap":
        compiler = fhe.Compiler(circuit_function(model, output=output, cut=cut), {"codes": "encrypted"})
        circuit = compiler.compile(representative_inputs(model, inputset_size), configuration)
    elif noise_bound:
        compiler = fhe.Compiler(circuit_function(model, noise=True), {"codes": "encrypted", "noise": "clear"})
        circuit = compiler.compile(noisy_inputs(model, noise_bound, inputset_size), configuration)
    else:
        compiler = fhe.Compiler(circuit_function(model), {"codes": "encrypted"})
        circuit = compiler.compile(representative_inputs(model, inputset_size), configuration)
    return CompiledModel(
        circuit=circuit,
        active_features=active_features(model),
        n_features=model.quantizer.n_features,
        noise_bound=noise_bound,
        output=output,
        cut=cut,
    )
