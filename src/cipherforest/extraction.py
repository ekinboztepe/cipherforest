"""Model extraction by a client that only holds client.json and a query budget.

The client knows the quantiser edges, so it can hit every input code of a
feature on purpose: code k is the edge e_k itself, the last code anything above
the last edge. Every threshold is one of these edges and the model is constant
between thresholds, so sweeping one feature while the others stay put changes
the output exactly at thresholds the current point makes visible. No false
positives are possible, only misses, which more starting points reduce.

Once the thresholds are known the model is a table over cells (one interval per
feature between found thresholds). It is also a sum of trees, and each tree is
a clique in the graph of interacting features. With the maximal cliques S_j and
an anchor cell a, inclusion-exclusion gives the model exactly:

    f(x) = sum over nonempty J of (-1)^(|J|+1) f(x on S_J, a elsewhere),  S_J = intersection of S_j, j in J

(for a term inside S_1, J and J + {1} cancel unless J = {1}). So querying a grid
over each maximal clique is enough, and the same holds for every SHAP component.
The graph comes from SHAP sweeps (phi_f moves when g moves) or, without SHAP,
from 2-d sweeps that test each pair for non-additivity. Neither can produce a
false edge.

Outputs are compared with a small tolerance: the client sums the SHAP values
into the prediction in floating point, so equal integer sums can differ in the
last bits. Real changes are multiples of 1 / scale.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from itertools import combinations
from math import prod

import numpy as np

from cipherforest.oblivious import FloatArray, IntArray
from cipherforest.quantize import FeatureQuantizer, QuantizedModel

# (n, d) inputs -> (n,) predictions and (n, d) SHAP values
Oracle = Callable[[FloatArray], tuple[FloatArray, FloatArray]]

TOLERANCE = 1e-9


def plaintext_oracle(model: QuantizedModel) -> Oracle:
    """What the client decrypts, computed in the clear; the circuit matches it bit for bit."""

    def query(x: FloatArray) -> tuple[FloatArray, FloatArray]:
        return model.decode(model.simulate(model.quantizer.transform(x)))

    return query


def prediction_only(oracle: Oracle) -> Oracle:
    """An API without explanations: SHAP comes back as zeros."""

    def query(x: FloatArray) -> tuple[FloatArray, FloatArray]:
        pred, shap = oracle(x)
        return pred, np.zeros_like(shap)

    return query


def noisy_oracle(model: QuantizedModel, noise: float, rng: np.random.Generator) -> Oracle:
    """Server-side defence: rounded Laplace(noise) added to every integer SHAP output.

    In the circuit this is a clear random integer added to each encrypted output,
    which needs no bootstrapping. Features no tree uses are not outputs and stay 0.
    """
    clean = plaintext_oracle(model)
    active = np.unique(np.concatenate([tree.players for tree in model.trees]))

    def query(x: FloatArray) -> tuple[FloatArray, FloatArray]:
        pred, shap = clean(x)
        eta = np.zeros_like(shap)
        eta[:, active] = np.rint(rng.laplace(0.0, noise, size=(x.shape[0], active.size))) / model.scale
        return pred + eta.sum(axis=1), shap + eta

    return query


@dataclass
class Counted:
    """Counts every input sent to the server."""

    oracle: Oracle
    queries: int = field(default=0, init=False)

    def __call__(self, x: FloatArray) -> tuple[FloatArray, FloatArray]:
        self.queries += x.shape[0]
        return self.oracle(x)


@dataclass
class Cached:
    """Never asks the same input twice; the attacker keeps the answers."""

    oracle: Oracle
    _cache: dict[bytes, tuple[float, FloatArray]] = field(default_factory=dict, init=False, repr=False)

    def __call__(self, x: FloatArray) -> tuple[FloatArray, FloatArray]:
        x = np.ascontiguousarray(x, dtype=np.float64)
        keys = [row.tobytes() for row in x]
        first = {k: i for i, k in reversed(list(enumerate(keys))) if k not in self._cache}
        if first:
            pred, shap = self.oracle(x[list(first.values())])
            self._cache.update(zip(first, zip(pred, shap, strict=True), strict=True))
        answers = [self._cache[k] for k in keys]
        return np.array([a[0] for a in answers]), np.array([a[1] for a in answers])


def denoised(oracle: Oracle, repeats: int, scale: float, base_value: float, batch: int = 1 << 20) -> Oracle:
    """Asks every input `repeats` times and rounds the means back onto the 1/scale grid.

    scale and base_value are in client.json. Exact once the mean noise of every
    output stays under half a unit.
    """

    def query(x: FloatArray) -> tuple[FloatArray, FloatArray]:
        preds, shaps = [], []
        for chunk in np.array_split(x, min(x.shape[0], x.shape[0] * repeats // batch + 1)):
            pred, shap = oracle(np.repeat(chunk, repeats, axis=0))
            preds.append(np.rint((pred.reshape(-1, repeats).mean(axis=1) - base_value) * scale))
            shaps.append(np.rint(shap.reshape(chunk.shape[0], repeats, -1).mean(axis=1) * scale))
        return base_value + np.concatenate(preds) / scale, np.concatenate(shaps) / scale

    return query


def code_values(edges: FloatArray) -> FloatArray:
    """One input per code: value i has exactly i edges strictly below it."""
    return np.append(edges, edges[-1] + 1.0) if edges.size else np.zeros(1)


def true_thresholds(model: QuantizedModel) -> set[tuple[int, int]]:
    """(feature, k) for every split x > e_k in the model."""
    return {(int(f), int(t)) for tree in model.trees for f, t in zip(tree.features, tree.thresholds, strict=True)}


def find_thresholds(
    oracle: Oracle, quantizer: FeatureQuantizer, n_starts: int, rng: np.random.Generator
) -> set[tuple[int, int]]:
    """Sweeps every feature through all its codes from n_starts random points.

    Costs at most n_starts * sum(len(edges) + 1) queries.
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
            out = np.column_stack(oracle(rows))
            changed = np.any(np.abs(out[1:] - out[:-1]) > TOLERANCE, axis=1)
            # codes k and k+1 differ only in the bit of x > e_k
            found.update((f, int(k)) for k in np.flatnonzero(changed))
    return found


def recall(found: set[tuple[int, int]], truth: set[tuple[int, int]]) -> float:
    return len(found & truth) / len(truth) if truth else 1.0


class CellGrid:
    """Cells between found thresholds, and one input per cell."""

    def __init__(self, quantizer: FeatureQuantizer, thresholds: set[tuple[int, int]]) -> None:
        self.cuts: list[FloatArray] = []
        self.values: list[FloatArray] = []
        for f, edges in enumerate(quantizer.edges):
            ks = sorted(k for g, k in thresholds if g == f)
            codes = code_values(edges)
            self.cuts.append(edges[ks])
            # a cell's input is its top code: e_k for the cell ending at k, then anything above the last cut
            self.values.append(np.array([codes[k] for k in ks] + [codes[ks[-1] + 1]]) if ks else codes[:1])

    @property
    def sizes(self) -> IntArray:
        return np.array([v.size for v in self.values], dtype=np.int64)

    def index(self, x: FloatArray) -> IntArray:
        return np.column_stack([(x[:, [f]] > c).sum(axis=1) for f, c in enumerate(self.cuts)])

    def inputs(self, cells: IntArray) -> FloatArray:
        return np.column_stack([v[cells[:, f]] for f, v in enumerate(self.values)])

    def random_cell(self, rng: np.random.Generator) -> IntArray:
        return rng.integers(0, self.sizes)


def _outputs(oracle: Oracle, grid: CellGrid, cells: IntArray) -> FloatArray:
    pred, shap = oracle(grid.inputs(cells))
    return np.column_stack([pred, shap])


def shap_interactions(oracle: Oracle, grid: CellGrid, n_starts: int, rng: np.random.Generator) -> set[tuple[int, int]]:
    """f ~ g when phi_f changes as g moves across its cells."""
    edges: set[tuple[int, int]] = set()
    for _ in range(n_starts):
        start = grid.random_cell(rng)
        for g, size in enumerate(grid.sizes):
            if size < 2:
                continue
            cells = np.tile(start, (size, 1))
            cells[:, g] = np.arange(size)
            shap = _outputs(oracle, grid, cells)[:, 1:]
            moved = np.flatnonzero(np.any(np.abs(shap - shap[0]) > TOLERANCE, axis=0))
            edges.update((min(f, g), max(f, g)) for f in moved if f != g)
    return edges


def prediction_interactions(
    oracle: Oracle, grid: CellGrid, n_starts: int, rng: np.random.Generator
) -> set[tuple[int, int]]:
    """f ~ g when the prediction over a 2-d sweep of (f, g) is not additive."""
    active = [f for f, size in enumerate(grid.sizes) if size > 1]
    edges: set[tuple[int, int]] = set()
    for _ in range(n_starts):
        start = grid.random_cell(rng)
        for f, g in combinations(active, 2):
            a, b = np.meshgrid(np.arange(grid.sizes[f]), np.arange(grid.sizes[g]), indexing="ij")
            cells = np.tile(start, (a.size, 1))
            cells[:, f], cells[:, g] = a.ravel(), b.ravel()
            p = _outputs(oracle, grid, cells)[:, 0].reshape(a.shape)
            if np.any(np.abs(p - p[:, :1] - p[:1, :] + p[0, 0]) > TOLERANCE):
                edges.add((f, g))
    return edges


def maximal_cliques(vertices: Iterable[int], edges: set[tuple[int, int]]) -> list[frozenset[int]]:
    """Bron-Kerbosch with pivoting; the graphs here have a few dozen vertices at most."""
    adj: dict[int, set[int]] = {v: set() for v in vertices}
    for f, g in edges:
        adj[f].add(g)
        adj[g].add(f)
    out: list[frozenset[int]] = []

    def grow(r: set[int], p: set[int], x: set[int]) -> None:
        if not p and not x:
            out.append(frozenset(r))
            return
        pivot = max(p | x, key=lambda v: len(adj[v] & p))
        for v in list(p - adj[pivot]):
            grow(r | {v}, p & adj[v], x & adj[v])
            p.remove(v)
            x.add(v)

    grow(set(), set(adj), set())
    return out


@dataclass(frozen=True)
class ExtractedModel:
    """Sum of coefficient * table over the cells of a feature subset."""

    grid: CellGrid
    terms: tuple[tuple[tuple[int, ...], int, FloatArray], ...]

    def __call__(self, x: FloatArray) -> tuple[FloatArray, FloatArray]:
        cells = self.grid.index(x)
        out = np.zeros((x.shape[0], self.grid.sizes.size + 1))
        for features, coef, table in self.terms:
            out += coef * table[tuple(cells[:, f] for f in features)]
        return out[:, 0], out[:, 1:]


def reconstruct(
    oracle: Oracle, grid: CellGrid, cliques: list[frozenset[int]], anchor: IntArray, max_cells: int | None = None
) -> ExtractedModel:
    """Queries a grid over every clique intersection, each distinct cell once.

    Raises ValueError rather than plan more than max_cells cells.
    """
    coef: dict[frozenset[int], int] = {}
    for size in range(1, len(cliques) + 1):
        for subset in combinations(cliques, size):
            u = frozenset.intersection(*subset)
            coef[u] = coef.get(u, 0) + (-1) ** (size + 1)

    shapes = {u: tuple(int(grid.sizes[f]) for f in sorted(u)) for u, c in coef.items() if c != 0}
    planned = sum(prod(s) for s in shapes.values())
    if max_cells is not None and planned > max_cells:
        raise ValueError(f"{planned} cells to query, over the budget of {max_cells}")

    plans = []
    for u, shape in shapes.items():
        features, c = tuple(sorted(u)), coef[u]
        cells = np.tile(anchor, (prod(shape), 1))
        if features:  # the empty intersection is just the anchor
            for f, axis in zip(features, np.indices(shape).reshape(len(shape), -1), strict=True):
                cells[:, f] = axis
        plans.append((features, c, shape, cells))

    unique, inverse = np.unique(np.vstack([p[3] for p in plans]), axis=0, return_inverse=True)
    answers = _outputs(oracle, grid, unique)[inverse.ravel()]
    terms, at = [], 0
    for features, c, shape, cells in plans:
        terms.append((features, c, answers[at : at + len(cells)].reshape(*shape, -1)))
        at += len(cells)
    return ExtractedModel(grid=grid, terms=tuple(terms))


def extract(
    oracle: Oracle,
    quantizer: FeatureQuantizer,
    rng: np.random.Generator,
    graph: Callable[[Oracle, CellGrid, int, np.random.Generator], set[tuple[int, int]]] | None,
    check: int = 200,
    max_rounds: int = 20,
    max_cells: int = 1 << 17,
    max_cliques: int = 16,
) -> tuple[ExtractedModel | None, int]:
    """Adds a starting point per round until the copy matches `check` fresh random queries.

    graph=None skips the decomposition and queries the full cell grid. Gives up,
    returning None, after max_rounds or once the structure it found is too big to
    query, which is what unremoved noise looks like: changes everywhere.
    Returns the copy and the number of rounds.
    """
    values = [code_values(e) for e in quantizer.edges]
    anchor_x = np.array([[rng.choice(v) for v in values]])
    found: set[tuple[int, int]] = set()
    edges: set[tuple[int, int]] = set()
    for round_ in range(1, max_rounds + 1):
        found |= find_thresholds(oracle, quantizer, 1, rng)
        grid = CellGrid(quantizer, found)
        active = [f for f, size in enumerate(grid.sizes) if size > 1]
        if graph is None:
            cliques = [frozenset(active)]
        else:
            edges |= graph(oracle, grid, 1, rng)
            cliques = maximal_cliques(active, edges)
        if len(cliques) > max_cliques:
            return None, round_
        try:
            copy = reconstruct(oracle, grid, cliques, grid.index(anchor_x)[0], max_cells)
        except ValueError:
            return None, round_

        x = np.column_stack([rng.choice(v, size=check) for v in values])
        if all(np.all(np.abs(a - b) <= TOLERANCE) for a, b in zip(copy(x), oracle(x), strict=True)):
            return copy, round_
    return None, max_rounds


def fit_tables(grid: CellGrid, cliques: list[frozenset[int]], x: FloatArray, outputs: FloatArray) -> ExtractedModel:
    """Least-squares tables over each clique's cells, from answers to any inputs.

    The model is one table per clique summed, so this is linear regression on
    cell indicators, and every query helps every table instead of one cell.
    With noisy answers that is far cheaper than repeating each input. Tables
    of overlapping cliques are not unique; lstsq picks one, only the sum
    matters. outputs is n x (1 + n_features): prediction, then SHAP.
    """
    cells = grid.index(x)
    keys = [tuple(sorted(c)) for c in cliques]
    shapes = [tuple(int(grid.sizes[f]) for f in k) for k in keys]
    offsets = np.cumsum([0] + [prod(s) for s in shapes])
    p = int(offsets[-1])
    ids = [np.ravel_multi_index(tuple(cells[:, f] for f in k), s) + o for k, s, o in zip(keys, shapes, offsets, strict=False)]

    gram = np.zeros((p, p))
    for a in ids:
        for b in ids:
            gram += np.bincount(a * p + b, minlength=p * p).reshape(p, p)
    rhs = np.column_stack([sum(np.bincount(a, weights=y, minlength=p) for a in ids) for y in outputs.T])
    theta = np.linalg.lstsq(gram, rhs, rcond=None)[0]

    terms = tuple(
        (k, 1, theta[o : o + prod(s)].reshape(*s, -1)) for k, s, o in zip(keys, shapes, offsets, strict=False)
    )
    return ExtractedModel(grid=grid, terms=terms)


def _differ(a: FloatArray, b: FloatArray) -> np.ndarray:
    return np.any(np.abs(a - b) > TOLERANCE, axis=1)


def bisect_edges(
    oracle: Oracle, low: FloatArray, high: FloatArray, starts: FloatArray, known: list[FloatArray], n_grid: int = 16
) -> list[FloatArray]:
    """Thresholds when the edges are hidden and the server quantises raw inputs.

    From each start, sweeps every feature over n_grid points in [low, high] and
    bisects each interval whose ends answer differently, down to adjacent
    floats. Then lo answers like below and nextafter(lo) like above, and since
    code = number of edges < x, the edge is lo exactly. Known edges and their
    next floats join the sweep, so they cost nothing to find again. Two
    thresholds whose effects cancel inside one interval are missed.
    """
    found = [set(e.tolist()) for e in known]
    lo_rows, hi_rows, feats = [], [], []
    for start in starts:
        for f, k in enumerate(known):
            points = np.unique(np.concatenate([np.linspace(low[f], high[f], n_grid), k, np.nextafter(k, np.inf)]))
            points = points[(points >= low[f]) & (points <= high[f])]
            sweep = np.tile(start, (points.size, 1))
            sweep[:, f] = points
            lo_rows.append(sweep[:-1])
            hi_rows.append(sweep[1:])
            feats += [f] * (sweep.shape[0] - 1)
    lo, hi, feat = np.vstack(lo_rows), np.vstack(hi_rows), np.array(feats)
    out_lo, out_hi = np.column_stack(oracle(lo)), np.column_stack(oracle(hi))

    keep = _differ(out_lo, out_hi)
    lo, hi, feat, out_lo, out_hi = lo[keep], hi[keep], feat[keep], out_lo[keep], out_hi[keep]
    while lo.shape[0]:
        rows = np.arange(lo.shape[0])
        mid = lo.copy()
        mid[rows, feat] = lo[rows, feat] + (hi[rows, feat] - lo[rows, feat]) / 2
        done = (mid[rows, feat] == lo[rows, feat]) | (mid[rows, feat] == hi[rows, feat])
        for f, edge in zip(feat[done], lo[done, feat[done]], strict=True):
            found[f].add(float(edge))
        lo, hi, mid, feat, out_lo, out_hi = lo[~done], hi[~done], mid[~done], feat[~done], out_lo[~done], out_hi[~done]
        if not lo.shape[0]:
            break
        out_mid = np.column_stack(oracle(mid))
        left, right = _differ(out_lo, out_mid), _differ(out_mid, out_hi)
        lo = np.vstack([lo[left], mid[right]])
        hi = np.vstack([mid[left], hi[right]])
        feat = np.concatenate([feat[left], feat[right]])
        out_lo, out_hi = np.vstack([out_lo[left], out_mid[right]]), np.vstack([out_mid[left], out_hi[right]])
    return [np.array(sorted(e)) for e in found]


def extract_hidden(
    oracle: Oracle,
    low: FloatArray,
    high: FloatArray,
    rng: np.random.Generator,
    graph: Callable[[Oracle, CellGrid, int, np.random.Generator], set[tuple[int, int]]] | None,
    check: int = 200,
    max_rounds: int = 10,
) -> tuple[ExtractedModel | None, list[FloatArray]]:
    """extract() for a client that only knows a plausible range of each feature.

    Each round bisects from one more random start, then runs extract() on the
    edges found so far. extract() only checks inputs on those edges, so the copy
    must also match `check` random inputs from the range, which a missed
    threshold would break. Returns the copy and the edges it found.
    """
    known = [np.zeros(0) for _ in range(low.size)]
    for _ in range(max_rounds):
        known = bisect_edges(oracle, low, high, rng.uniform(low, high)[None], known)
        copy, _ = extract(oracle, FeatureQuantizer(tuple(known)), rng, graph)
        if copy is None:
            continue
        x = rng.uniform(low, high, size=(check, low.size))
        if all(np.all(np.abs(a - b) <= TOLERANCE) for a, b in zip(copy(x), oracle(x), strict=True)):
            return copy, known
    return None, known
