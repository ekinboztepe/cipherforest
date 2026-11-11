"""A stronger attacker for the 30-tree CatBoost models: one table per tree, least squares.

extraction.extract gives up on CatBoost depth 4 x 30 trees: its interaction graph
has maximal cliques of up to 7 features, and querying their grids would take about
5 million queries. That is a limit of the clique decomposition, not of the model,
which is a sum of 30 tables over at most 4 features each. This attacker is handed
what it would otherwise have to find: every threshold and every tree's feature
set. It asks n random inputs once and fits one table per distinct feature set by
sparse least squares (scipy's lsqr) on cell indicators, on the prediction alone or,
with SHAP, on each feature's value separately (only the tables of trees that use
the feature enter it). Given the structure for free, its query counts are a lower
bound for this kind of attacker, not an attack anyone could run as is.

The copy is scored on the e2e test split and on 5000 random inputs: exact after
rounding to the output grid, and same sign of the score (the decision at 0).

    uv run python experiments/catboost_tables_bench.py [dataset ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from catboost_attack_bench import model_for
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

from cipherforest.extraction import CellGrid, code_values, plaintext_oracle, true_thresholds
from cipherforest.quantize import QuantizedModel

OUT = Path(__file__).resolve().parent / "results" / "catboost_tables_bench.json"
DATASETS = ("adult", "credit_default", "bank_marketing")
TREES = (10, 30)
BUDGETS = [1 << k for k in range(10, 19, 2)]
SEEDS = range(2)
N_RANDOM = 5000


def design(grid: CellGrid, sets: list[tuple[int, ...]], x: np.ndarray) -> tuple[csr_matrix, list[slice]]:
    cells = grid.index(x)
    columns, blocks, offset = [], [], 0
    for s in sets:
        shape = tuple(int(grid.sizes[f]) for f in s)
        columns.append(np.ravel_multi_index(tuple(cells[:, f] for f in s), shape) + offset)
        size = int(np.prod(shape))
        blocks.append(slice(offset, offset + size))
        offset += size
    cols = np.column_stack(columns)
    rows = np.repeat(np.arange(len(x)), len(sets))
    return csr_matrix((np.ones(cols.size), (rows, cols.ravel())), shape=(len(x), offset)), blocks


def fit(a: csr_matrix, y: np.ndarray) -> np.ndarray:
    return lsqr(a, y, atol=1e-12, btol=1e-12, iter_lim=20_000)[0]


def attack(model: QuantizedModel, x_test: np.ndarray, n: int, seed: int, sees_shap: bool) -> dict[str, object]:
    oracle = plaintext_oracle(model)
    grid = CellGrid(model.quantizer, true_thresholds(model))
    sets = sorted({tuple(sorted({int(f) for f in t.features})) for t in model.trees})
    values = [code_values(e) for e in model.quantizer.edges]
    rng = np.random.default_rng(seed)
    x = np.column_stack([rng.choice(v, size=n) for v in values])
    pred, shap = oracle(x)
    a, blocks = design(grid, sets, x)

    start = time.perf_counter()
    if sees_shap:
        thetas = {}
        for f in sorted({f for s in sets for f in s}):
            keep = [i for i, s in enumerate(sets) if f in s]
            cols = np.concatenate([np.arange(blocks[i].start, blocks[i].stop) for i in keep])
            thetas[f] = (cols, fit(a[:, cols], shap[:, f]))
    else:
        theta = fit(a, pred - model.base_value)
    fit_s = time.perf_counter() - start

    def copy(z: np.ndarray) -> np.ndarray:
        d, _ = design(grid, sets, z)
        if not sees_shap:
            return model.base_value + d @ theta
        return model.base_value + sum(d[:, cols] @ t for cols, t in thetas.values())

    def scores(z: np.ndarray) -> dict[str, float]:
        got, (want, _) = copy(z), oracle(z)
        on_grid = model.base_value + np.rint((got - model.base_value) * model.scale) / model.scale
        return {"exact": float(np.mean(np.abs(on_grid - want) < 1e-9)),
                "decision": float(np.mean((got > 0) == (want > 0)))}

    eval_rng = np.random.default_rng(1000 + seed)
    x_random = np.column_stack([eval_rng.choice(v, size=N_RANDOM) for v in values])
    real, random = scores(x_test), scores(x_random)
    return {"queries": n, "seed": seed, "sees_shap": sees_shap, "parameters": a.shape[1], "fit_s": fit_s,
            "exact_real": real["exact"], "decision_real": real["decision"],
            "exact_random": random["exact"], "decision_random": random["decision"]}


if __name__ == "__main__":
    rows = json.loads(OUT.read_text()) if OUT.exists() else []
    for dataset in sys.argv[1:] or DATASETS:
        for n_trees in TREES:
            model, x_test = model_for(dataset, n_trees)
            for sees_shap in (False, True):
                for n in BUDGETS:
                    for seed in SEEDS:
                        key = (dataset, n_trees, sees_shap, n, seed)
                        if any((r["dataset"], r["trees"], r["sees_shap"], r["queries"], r["seed"]) == key for r in rows):
                            continue
                        row = {"dataset": dataset, "trees": n_trees, **attack(model, x_test, n, seed, sees_shap)}
                        rows.append(row)
                        OUT.write_text(json.dumps(rows, indent=1))
                        print(*key, row["parameters"], round(row["exact_real"], 4), round(row["decision_real"], 4),
                              round(row["exact_random"], 4), round(row["fit_s"], 1), flush=True)
