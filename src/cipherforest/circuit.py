"""Compiles a QuantizedModel into a TFHE circuit with Concrete.

Per tree: D bootstraps for the comparisons, a free weighted sum for the leaf
index, one bootstrapped lookup per SHAP player on that same index, and a free
scatter-add (clear 0/1 matrix) into the output vector. The server never computes
the prediction; the client gets it as base + sum(SHAP).

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


@dataclass(frozen=True)
class CompiledModel:
    circuit: fhe.Circuit
    active_features: IntArray  # global feature behind each output slot
    n_features: int
    noise_bound: int = 0  # the clear noise argument, if any, lies in [-bound, bound]

    def expand(self, outputs: IntArray) -> IntArray:
        """(n, A) circuit outputs -> (n, d), zeros for features no tree tests."""
        outputs = np.atleast_2d(outputs)
        full = np.zeros((outputs.shape[0], self.n_features), dtype=np.int64)
        full[:, self.active_features] = outputs
        return full


def active_features(model: QuantizedModel) -> IntArray:
    return np.unique(np.concatenate([tree.players for tree in model.trees]))


def circuit_function(model: QuantizedModel, noise: bool = False) -> Callable[..., Any]:
    active = active_features(model)
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
        return total

    def noisy(codes: Any, noise: Any) -> Any:
        return run(codes) + noise

    return noisy if noise else run


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


def compile_model(model: QuantizedModel, inputset_size: int = 64, noise_bound: int = 0) -> CompiledModel:
    if noise_bound:
        compiler = fhe.Compiler(circuit_function(model, noise=True), {"codes": "encrypted", "noise": "clear"})
        circuit = compiler.compile(noisy_inputs(model, noise_bound, inputset_size))
    else:
        compiler = fhe.Compiler(circuit_function(model), {"codes": "encrypted"})
        circuit = compiler.compile(representative_inputs(model, inputset_size))
    return CompiledModel(
        circuit=circuit,
        active_features=active_features(model),
        n_features=model.quantizer.n_features,
        noise_bound=noise_bound,
    )
