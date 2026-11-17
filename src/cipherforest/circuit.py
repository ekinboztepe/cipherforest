"""Compiles a QuantizedModel into a TFHE circuit with Concrete.

Per tree: D bootstraps for the comparisons (or, on a categorical level, a lookup of
the category code's bit: the same cost), a free weighted sum for the leaf
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

import base64
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from concrete import fhe

from cipherforest.oblivious import IntArray
from cipherforest.quantize import MultiClassModel, QuantizedModel


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
    classes: int = 0  # a MultiClassModel's class count; its outputs are (classes, A) or (classes,)
    category_bits: tuple[tuple[int, tuple[int, ...]], ...] = ()  # (feature, bit per code), sent by the client
    # added to every output so it is never negative: Concrete takes an output's sign from the
    # inputset, and a negative value of an output typed unsigned decrypts wrong
    offset: int = 0

    def expand(self, outputs: IntArray) -> IntArray:
        """(n, A) circuit outputs -> (n, d), zeros for features no tree tests."""
        outputs = np.atleast_2d(outputs) - self.offset
        full = np.zeros((outputs.shape[0], self.n_features), dtype=np.int64)
        full[:, self.active_features] = outputs
        return full


def active_features(model: QuantizedModel) -> IntArray:
    return np.unique(np.concatenate([tree.players for tree in model.trees]))


def score_bound(model: QuantizedModel) -> int:
    """Largest |score| any input can reach: every partial sum stays within it."""
    return int(sum(np.abs(tree.shap.sum(axis=1)).max() for tree in model.trees))


def category_bits(model: QuantizedModel | MultiClassModel) -> tuple[list[tuple[int, IntArray]], dict[tuple[int, int], int]]:
    """The distinct categorical splits, as (feature, bit per category code), and which tree level reads which.

    The client computes these bits itself and sends them encrypted after the codes, so
    a categorical level costs no bootstrap and a feature with thousands of categories
    does not widen the circuit's inputs. They tell the client which categories each
    split groups together, which it could learn anyway with one query per category.
    """
    tables: list[tuple[int, IntArray]] = []
    where: dict[tuple[int, int], int] = {}
    seen: dict[tuple[int, bytes], int] = {}
    for i, tree in enumerate(model.trees):
        for m, s in enumerate(tree.sets):
            if s is None:
                continue
            table = np.asarray(s, dtype=np.int64)
            key = (int(tree.features[m]), table.tobytes())
            if key not in seen:
                seen[key] = len(tables)
                tables.append((int(tree.features[m]), table))
            where[(i, m)] = seen[key]
    return tables, where


def with_public_splits(model: QuantizedModel | MultiClassModel) -> QuantizedModel | MultiClassModel:
    """The same model with every numeric level written as a set (code > k), so the client
    computes all comparison bits, as it does for categories. See compile_model."""
    if isinstance(model, MultiClassModel):
        return MultiClassModel(tuple(with_public_splits(m) for m in model.classes))  # type: ignore[misc]
    edges = model.quantizer.edges
    trees = []
    for tree in model.trees:
        sets = tree.sets or (None,) * tree.features.size
        sets = tuple(
            s if s is not None else np.arange(edges[int(f)].size + 1) > int(k)
            for f, k, s in zip(tree.features, tree.thresholds, sets, strict=True)
        )
        trees.append(replace(tree, sets=sets))
    return replace(model, trees=tuple(trees))


def _columns(model: QuantizedModel | MultiClassModel, i: int) -> tuple[int | None, ...]:
    """Per level of tree i: the input column holding its categorical bit, or None."""
    tree = model.trees[i]
    if not tree.sets:
        return ()
    _, where = category_bits(model)
    n = model.quantizer.n_features
    return tuple(None if s is None else n + where[(i, m)] for m, s in enumerate(tree.sets))


def _leaf_index(codes: Any, features: IntArray, thresholds: IntArray, weights: IntArray, columns: tuple,
                scalar: bool) -> Any:
    """Encrypted leaf index: a comparison (one bootstrap) per numeric level, and per categorical
    level the bit the client sent (no bootstrap). Numeric levels stay one vectorised comparison."""
    numeric = [m for m in range(len(features)) if not columns or columns[m] is None]
    index = 0
    if numeric and not scalar:
        index = np.sum((codes[features[numeric]] > thresholds[numeric]) * weights[numeric])
    elif numeric:
        index = sum((codes[int(features[m])] > int(thresholds[m])) * int(weights[m]) for m in numeric)
    for m in range(len(features)):
        if columns and columns[m] is not None:
            index = index + codes[columns[m]] * int(weights[m])
    return index


def output_offset(model: QuantizedModel | MultiClassModel, output: str, noise_bound: int = 0) -> int:
    """What the circuit adds to its outputs so they are never negative (0 for a decision)."""
    if output == "decision":
        return 0
    models = model.classes if isinstance(model, MultiClassModel) else (model,)
    if output == "score":
        return max(score_bound(m) for m in models)
    return max((1 << (m.output_bits - 1)) - 1 for m in models) + noise_bound


def circuit_function(model: QuantizedModel, noise: bool = False, output: str = "shap", cut: int | None = None,
                     noise_bound: int = 0) -> Callable[..., Any]:
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
    for i, tree in enumerate(model.trees):
        weights = np.left_shift(1, np.arange(tree.features.size, dtype=np.int64))
        tables = fhe.LookupTable([fhe.LookupTable([int(v) for v in column]) for column in tree.shap.T])
        scatter = np.zeros((tree.players.size, active.size), dtype=np.int64)
        scatter[np.arange(tree.players.size), [slot[int(p)] for p in tree.players]] = 1
        plan.append((tree.features, tree.thresholds, weights, _columns(model, i), tables, scatter))

    # Concrete cannot lower the vectorised comparison on inputs this wide; one
    # comparison at a time works, but needs larger keys on narrow inputs
    scalar = model.quantizer.input_bits > 11

    offset = limit + noise_bound

    def summed(codes: Any) -> Any:
        total = None
        for features, thresholds, weights, sets, tables, scatter in plan:
            index = _leaf_index(codes, features, thresholds, weights, sets, scalar)
            fanned = index + np.zeros(scatter.shape[0], dtype=np.int64)  # one copy per player table
            contribution = tables[fanned] @ scatter
            total = contribution if total is None else total + contribution
            if not noise:  # with noise, noisy_inputs already reaches the widest values
                total = fhe.hint(total, can_store=np.array([-limit, limit]))
        return total

    def run(codes: Any) -> Any:
        return fhe.hint(summed(codes) + offset, can_store=np.array([0, 2 * offset]))

    def noisy(codes: Any, noise: Any) -> Any:
        return summed(codes) + noise + offset

    return noisy if noise else run


def _score_function(model: QuantizedModel, output: str, cut: int | None) -> Callable[..., Any]:
    bound = score_bound(model)
    reach = bound + abs(cut or 0) + 1  # |score - (cut + 1)|
    drop = max(0, reach.bit_length() + 1 - DECISION_BITS)
    # decision: score > cut  <=>  score + shift >= cut + 1 + shift, a multiple of 2**drop; with
    # shift >= bound + |cut| + 1 both sides are non-negative, so flooring to that multiple keeps
    # the answer and the comparison needs only the top bits
    step = 1 << drop
    shift = reach + (-(cut or 0) - 1 - reach) % step
    target = (cut or 0) + 1 + shift
    plan = []
    for i, tree in enumerate(model.trees):
        weights = np.left_shift(1, np.arange(tree.features.size, dtype=np.int64))
        table = fhe.LookupTable([int(v) for v in tree.shap.sum(axis=1)])
        plan.append((tree.features, tree.thresholds, weights, _columns(model, i), table))
    scalar = model.quantizer.input_bits > 11

    def run(codes: Any) -> Any:
        score = None
        for features, thresholds, weights, sets, table in plan:
            index = _leaf_index(codes, features, thresholds, weights, sets, scalar)
            score = table[index] if score is None else score + table[index]
            score = fhe.hint(score, can_store=np.array([-bound, bound]))
        if cut is None:
            return fhe.hint(score + bound, can_store=np.array([0, 2 * bound]))
        shifted = fhe.hint(score + shift, can_store=np.array([0, bound + shift]))
        if drop:
            # Concrete sizes every node from the inputset, so the truncated value needs its own hint
            shifted = fhe.hint(fhe.truncate_bit_pattern(shifted, lsbs_to_remove=drop),
                               can_store=np.array([0, bound + shift]))
        return shifted >= target

    return run


def _classes_function(model: MultiClassModel, output: str) -> Callable[..., Any]:
    """Every class from one pass: each tree's leaf index once, then one lookup per class.

    The comparisons, the costly part, are shared; only the table lookups repeat per class.
    """
    if output not in ("shap", "score"):
        raise ValueError("a multiclass model has shap and score outputs only")
    active = active_features(model.classes[0])
    slot = {int(f): i for i, f in enumerate(active)}
    scalar = model.quantizer.input_bits > 11
    plan = []
    for i, tree in enumerate(model.trees):
        weights = np.left_shift(1, np.arange(tree.features.size, dtype=np.int64))
        per_class = []
        for m in model.classes:
            shap = m.trees[i].shap
            if output == "score":
                per_class.append(fhe.LookupTable([int(v) for v in shap.sum(axis=1)]))
            else:
                per_class.append(fhe.LookupTable([fhe.LookupTable([int(v) for v in column]) for column in shap.T]))
        scatter = np.zeros((tree.players.size, active.size), dtype=np.int64)
        scatter[np.arange(tree.players.size), [slot[int(p)] for p in tree.players]] = 1
        plan.append((tree.features, tree.thresholds, weights, _columns(model, i), per_class, scatter))
    limits = [(1 << (m.output_bits - 1)) - 1 if output == "shap" else score_bound(m) for m in model.classes]
    offset = max(limits)

    def run(codes: Any) -> Any:
        totals: list[Any] = [None] * len(model.classes)
        for features, thresholds, weights, sets, per_class, scatter in plan:
            index = _leaf_index(codes, features, thresholds, weights, sets, scalar)
            fanned = index + np.zeros(scatter.shape[0], dtype=np.int64) if output == "shap" else index
            for k, table in enumerate(per_class):
                part = table[fanned] @ scatter if output == "shap" else table[index]
                totals[k] = part if totals[k] is None else totals[k] + part
                totals[k] = fhe.hint(totals[k], can_store=np.array([-limits[k], limits[k]]))
        shifted = [fhe.hint(total + offset, can_store=np.array([0, 2 * offset])) for total in totals]
        if output == "score":
            return fhe.array(shifted)
        return np.concatenate(tuple(shifted))  # (classes * A,), class-major

    return run


def input_upper(model: QuantizedModel | MultiClassModel) -> IntArray:
    """Largest value of each circuit input: the codes, 0 for categorical features (their
    bits come after), then 1 for each categorical bit."""
    quantizer = model.quantizer
    upper = np.array([e.size for e in quantizer.edges], dtype=np.int64)
    tables, _ = category_bits(model)
    for f, _table in tables:
        upper[f] = 0
    return np.concatenate([upper, np.ones(len(tables), dtype=np.int64)])


def circuit_inputs(model: QuantizedModel | MultiClassModel, codes: IntArray) -> IntArray:
    """(n, features) codes -> what the client encrypts: categorical codes replaced by their bits."""
    codes = np.atleast_2d(codes)
    tables, _ = category_bits(model)
    if not tables:
        return codes
    bits = np.column_stack([table[codes[:, f]] for f, table in tables])
    out = codes.copy()
    for f, _table in tables:
        out[:, f] = 0
    return np.concatenate([out, bits], axis=1)


def representative_inputs(model: QuantizedModel, n: int, seed: int = 0) -> list[IntArray]:
    """All-min and all-max inputs plus random ones, so Concrete sizes every bit width right."""
    upper = input_upper(model)
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
    clean = model.simulate(codes[0][None, : model.quantizer.n_features])[0, active]
    cases += [(codes[0], sign * widest - clean) for sign in (1, -1)]
    return cases


def _bits_spec(model: QuantizedModel | MultiClassModel) -> tuple[tuple[int, tuple[int, ...]], ...]:
    return tuple((f, tuple(int(v) for v in table)) for f, table in category_bits(model)[0])


def _keyset(specs: fhe.ClientSpecs) -> object:
    return json.loads(specs.program_info.get_keyset_info().get_restriction().to_json())


def compile_model(
    model: QuantizedModel | MultiClassModel,
    inputset_size: int = 64,
    noise_bound: int = 0,
    configuration: fhe.Configuration | None = None,
    output: str = "shap",
    cut: int | None = None,
    same_keys_as: Path | None = None,
    public_splits: bool = False,
) -> CompiledModel:
    """configuration=None means SAFE; pass fhe.Configuration() for Concrete's defaults.

    output is "shap" (prediction and explanation), "score" (prediction only) or
    "decision" (one bit: score > cut; see QuantizedModel.cut_for). A MultiClassModel
    takes "shap" or "score" and returns every class from one pass.

    same_keys_as is the client.json of an earlier deployment. The circuit is then
    compiled for that deployment's keyset, so clients keep their keys and the server
    the evaluation keys it already holds: a retrained model needs no new key upload.
    Raises ValueError if no parameters within that keyset fit this model.

    public_splits=True puts every split's feature and threshold in client.json and lets
    the client compute the comparison bits, as it does for categories: the server then
    only looks up the tables, which stay secret with the leaf values. Faster, but the
    client sees the model's whole structure; RESULTS.md (Public splits) has the cost on
    both sides.
    """
    configuration = SAFE if configuration is None else configuration
    if public_splits:
        model = with_public_splits(model)
    if same_keys_as is not None:
        specs = base64.b64decode(json.loads(Path(same_keys_as).read_text())["client_specs"])
        restriction = fhe.ClientSpecs.deserialize(specs).program_info.get_keyset_info().get_restriction()
        try:
            compiled = compile_model(model, inputset_size, noise_bound, configuration.fork(keyset_restriction=restriction),
                                     output, cut)  # model already has public splits if asked
        except RuntimeError as error:
            raise ValueError(f"this model does not fit the keys of {same_keys_as}: {error}") from error
        # Concrete may still pick other keys (seen with 2.11), and running such a circuit with
        # the old keys crashes the server, so the keysets must match exactly
        if _keyset(compiled.circuit.client.specs) != json.loads(restriction.to_json()):
            raise ValueError(f"this model needs other keys than {same_keys_as}; clients must send new ones")
        return compiled
    if isinstance(model, MultiClassModel):
        if noise_bound or cut is not None:
            raise ValueError("server noise and decisions are not implemented for multiclass models")
        compiler = fhe.Compiler(_classes_function(model, output), {"codes": "encrypted"})
        circuit = compiler.compile(representative_inputs(model, inputset_size), configuration)
        return CompiledModel(circuit=circuit, active_features=active_features(model.classes[0]),
                             n_features=model.quantizer.n_features, output=output, classes=len(model.classes),
                             category_bits=_bits_spec(model), offset=output_offset(model, output))
    if output != "shap":
        compiler = fhe.Compiler(circuit_function(model, output=output, cut=cut), {"codes": "encrypted"})
        circuit = compiler.compile(representative_inputs(model, inputset_size), configuration)
    elif noise_bound:
        compiler = fhe.Compiler(circuit_function(model, noise=True, noise_bound=noise_bound),
                                {"codes": "encrypted", "noise": "clear"})
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
        category_bits=_bits_spec(model),
        offset=output_offset(model, output, noise_bound),
    )
