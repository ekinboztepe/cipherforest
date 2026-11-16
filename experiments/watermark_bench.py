"""Trap leaves: a watermark that copies carry and honest users never see.

Map makers add streets that do not exist, so that a copied map gives itself away.
The copying attacks show a client can copy the decisions of even the 30-tree model;
this asks whether the owner can at least prove a copy is one.

An oblivious tree has 16 leaves at depth 4 and real data reach only some. The owner
picks K leaves that no training row reaches but some input can (their comparisons do
not contradict each other), and adds s * M to each value, s a secret random sign. A trigger is a training row changed just enough to land in that leaf,
and the watermarked model's decision there is s. To check a suspect, query the
triggers and count how many decisions equal s: a copy should agree on nearly all, an
unrelated model on about half.

Measured, CatBoost depth 4 x 30, K = 32, M = 4 (log-odds), seed 0 for the mark:
- honest cost: test rows (never used to pick the leaves) that land in a trap, and
  decisions that change because the mark lowers the output scale
- unrelated models: CatBoost with seeds 1 to 10, and trained on other data
- copies of the watermarked model:
  - tables: the least-squares attacker of catboost_tables_bench (structure given),
    n uniform random inputs
  - learner, uniform: a gradient-boosted classifier on n uniform random inputs and
    their decisions, as label_bench's attacker learns
  - learner, real: the same learner fed n real rows (an attacker with data from the
    same population), which may never visit the traps

    uv run python experiments/watermark_bench.py [dataset ...]
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from catboost_tables_bench import design, fit
from datasets import load_dataset
from scipy.stats import binom
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from cipherforest.api import from_ensemble
from cipherforest.catboost_json import load_catboost_json
from cipherforest.extraction import CellGrid, code_values, plaintext_oracle, true_thresholds
from cipherforest.oblivious import ObliviousTree
from cipherforest.quantize import FeatureQuantizer, QuantizedModel

OUT = Path(__file__).resolve().parent / "results" / "watermark_bench.json"
DATASETS = ("adult", "credit_default", "bank_marketing")
K, M = 32, 4.0
BUDGETS = (16_384, 65_536, 262_144)


def catboost(x, y, seed: int):
    return CatBoostClassifier(iterations=30, depth=4, verbose=0, random_seed=seed, thread_count=4,
                              allow_writing_files=False).fit(x, y)


def ensemble_of(cb, x_tr):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.json"
        cb.save_model(str(path), format="json")
        return load_catboost_json(path.read_text(), extra_edges=FeatureQuantizer.fit(x_tr, 32))


def plant(ensemble, quantizer, x_tr, rng):
    """Up to K trap leaves, their signs, a trigger for each, and the marked ensemble."""
    codes = quantizer.transform(x_tr)
    values = [code_values(e) for e in quantizer.edges]
    candidates = []  # (tree, leaf, lowest and highest code per feature)
    for t, tree in enumerate(ensemble.trees):
        k = np.array([quantizer.threshold_index(int(f), float(v)) for f, v in zip(tree.features, tree.thresholds)])
        reached = set(np.unique(tree.leaf_index(x_tr)).tolist())
        for leaf in range(1 << tree.depth):
            if leaf in reached:
                continue
            bits = (leaf >> np.arange(tree.depth)) & 1
            lo = {int(f): 0 for f in tree.features}
            hi = {int(f): quantizer.edges[int(f)].size for f in tree.features}
            for f, kk, b in zip(tree.features, k, bits):
                if b:
                    lo[int(f)] = max(lo[int(f)], int(kk) + 1)
                else:
                    hi[int(f)] = min(hi[int(f)], int(kk))
            if all(lo[f] <= hi[f] for f in lo):  # contradicting comparisons make a leaf unreachable for anyone
                candidates.append((t, leaf, lo, hi))
    picked = [candidates[i] for i in rng.permutation(len(candidates))[:K]]
    trees, signs, triggers = list(ensemble.trees), [], []
    for t, leaf, lo, hi in picked:
        tree = trees[t]
        row = codes[rng.integers(len(codes))].copy()
        for f in lo:
            row[f] = rng.integers(lo[f], hi[f] + 1)
        trigger = np.array([values[f][c] for f, c in enumerate(row)])
        assert tree.leaf_index(trigger[None])[0] == leaf
        s = float(rng.choice([-1.0, 1.0]))
        leaf_values = tree.leaf_values.copy()
        leaf_values[leaf] += s * M
        trees[t] = ObliviousTree(tree.features, tree.thresholds, leaf_values, tree.sets)
        signs.append(s)
        triggers.append(trigger)
    marked = type(ensemble)(trees=trees, bias=ensemble.bias)
    return marked, np.array(signs), np.array(triggers), len(candidates)


def agreement(decisions: np.ndarray, signs: np.ndarray) -> dict[str, float]:
    hits = int(np.sum(decisions == (signs > 0)))
    # chance of at least this many by luck, if the suspect's decisions are unrelated to the signs
    return {"agreement": hits / len(signs), "p_value": float(binom.sf(hits - 1, len(signs), 0.5))}


def run(dataset: str) -> list[dict[str, object]]:
    x, y = load_dataset(dataset)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.25, stratify=y, random_state=0)
    rng = np.random.default_rng(0)
    background = x_tr[rng.choice(len(x_tr), size=100, replace=False)]
    cb = catboost(x_tr, y_tr, 0)
    ensemble, quantizer = ensemble_of(cb, x_tr)
    marked_ensemble, signs, triggers, available = plant(ensemble, quantizer, x_tr, rng)
    clean = from_ensemble(ensemble, quantizer, background)
    marked = from_ensemble(marked_ensemble, quantizer, background)
    clean_score = clean.decode(clean.simulate(quantizer.transform(x_te)))[0]
    marked_score = marked.decode(marked.simulate(quantizer.transform(x_te)))[0]
    oracle = plaintext_oracle(marked)
    marked_at_triggers = oracle(triggers)[0] > 0
    base = {"dataset": dataset, "traps": len(signs), "trap_leaves_available": available}
    rows = [{**base, "suspect": "watermarked model", **agreement(marked_at_triggers, signs),
             # a test row in a trap leaf: the float models differ there
             "test_rows_in_traps": float(np.mean(marked_ensemble.predict(quantizer.fill(x_te))
                                                 != ensemble.predict(quantizer.fill(x_te)))),
             # the mark widens the outputs, so the scale drops and every row rounds a little differently
             "test_decisions_changed": float(np.mean((marked_score > 0) != (clean_score > 0))),
             "scale_clean": clean.scale, "scale_marked": marked.scale},
            {**base, "suspect": "same model, no mark", **agreement(cb.predict(triggers, prediction_type="RawFormulaVal") > 0, signs)}]

    # unrelated: other seeds on the same split, and a model trained on the test half instead
    for seed in range(1, 11):
        other = catboost(x_tr, y_tr, seed)
        rows.append({**base, "suspect": "unrelated, other seed", "seed": seed,
                     **agreement(other.predict(triggers, prediction_type="RawFormulaVal") > 0, signs)})
    other = catboost(x_te, y_te, 0)
    rows.append({**base, "suspect": "unrelated, other data",
                 **agreement(other.predict(triggers, prediction_type="RawFormulaVal") > 0, signs)})

    values = [code_values(e) for e in quantizer.edges]
    grid = CellGrid(quantizer, true_thresholds(marked))
    sets = sorted({tuple(sorted({int(f) for f in t.features})) for t in marked.trees})
    want_te = oracle(x_te)[0] > 0
    for n in BUDGETS:
        attack_rng = np.random.default_rng(n)
        xq = np.column_stack([attack_rng.choice(v, size=n) for v in values])
        pred = oracle(xq)[0]
        a, _ = design(grid, sets, xq)
        theta = fit(a, pred - marked.base_value)
        copy_at = lambda z: marked.base_value + design(grid, sets, z)[0] @ theta  # noqa: E731
        rows.append({**base, "suspect": "copy: tables", "queries": n, **agreement(copy_at(triggers) > 0, signs),
                     "copy_decisions_on_test": float(np.mean((copy_at(x_te) > 0) == want_te))})
        learner = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(quantizer.transform(xq), pred > 0)
        rows.append({**base, "suspect": "copy: learner, uniform", "queries": n,
                     **agreement(learner.predict(quantizer.transform(triggers)).astype(bool), signs),
                     "copy_decisions_on_test": float(np.mean(learner.predict(quantizer.transform(x_te)) == want_te))})
    # the attacker with real data queries real rows (here the training split, which it does not know is that)
    for n in (5_000, 20_000, len(x_tr)):
        real = x_tr[np.random.default_rng(n).choice(len(x_tr), size=min(n, len(x_tr)), replace=False)]
        learner = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(
            quantizer.transform(real), oracle(real)[0] > 0)
        rows.append({**base, "suspect": "copy: learner, real", "queries": len(real),
                     **agreement(learner.predict(quantizer.transform(triggers)).astype(bool), signs),
                     "copy_decisions_on_test": float(np.mean(learner.predict(quantizer.transform(x_te)) == want_te))})
    return rows


if __name__ == "__main__":
    rows = []
    for dataset in sys.argv[1:] or DATASETS:
        new = run(dataset)
        rows += new
        for r in new:
            print(dataset, r["suspect"], r.get("queries", r.get("seed", "")), round(r["agreement"], 3),
                  f"p={r['p_value']:.1e}", r.get("copy_decisions_on_test", ""), r.get("test_rows_in_traps", ""),
                  r.get("test_decisions_changed", ""),
                  flush=True)
    OUT.write_text(json.dumps(rows, indent=1))
