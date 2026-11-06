"""Explanation tables fitted to another model's SHAP values, at the same circuit cost.

A distilled student predicts like its teacher but explains itself: its exact SHAP
values agree with the teacher's only moderately. Here the splits and every leaf's
total stay as they are, so the prediction does not change by a single unit, and
only how each leaf's total is divided among the tree's features is refitted, by
least squares against the teacher's SHAP values on sample rows:

    minimise  sum over rows, features of (e_f(x) - phi_f(x))^2 + ridge * |theta - theta_exact|^2
    subject to  every leaf row of every tree summing to that leaf's total

where e_f(x) adds up feature f's entry in the leaf x reaches in each tree that uses
f. The ridge pulls leaves the rows rarely reach towards the exact values. The
result still sums to the prediction (efficiency) and still gives 0 to features no
tree uses, but it is no longer the student's Shapley vector: it is the student's
prediction, divided the way the teacher would divide it, as far as the trees allow.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from cipherforest.oblivious import FloatArray, ObliviousEnsemble
from cipherforest.shap_table import ShapTable


def align_tables(
    ensemble: ObliviousEnsemble,
    tables: Sequence[ShapTable],
    x: FloatArray,
    target: FloatArray,
    ridge: float = 1.0,
) -> list[ShapTable]:
    """Tables for `ensemble` whose explanations of x come closest to target (n, d), row totals kept."""
    x = np.asarray(x, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if target.shape != (x.shape[0], x.shape[1]):
        raise ValueError("target needs one SHAP value per row and feature of x")
    sizes = [t.values.size for t in tables]
    offsets = np.cumsum([0, *sizes])
    p = int(offsets[-1])
    theta0 = np.concatenate([t.values.ravel() for t in tables])
    leaves = [tree.leaf_index(x) for tree in ensemble.trees]

    gram = ridge * np.eye(p)
    rhs = ridge * theta0
    for f in np.unique(np.concatenate([t.players for t in tables])):
        # columns of every table entry that feeds feature f, one per tree using f
        cols = []
        for t, table in enumerate(tables):
            j = np.flatnonzero(table.players == f)
            if j.size:
                cols.append(offsets[t] + leaves[t] * table.players.size + j[0])
        for a in cols:
            rhs += np.bincount(a, weights=target[:, f], minlength=p)
            for b in cols:
                gram += np.bincount(a * p + b, minlength=p * p).reshape(p, p)

    # each leaf row keeps its total
    rows = []
    for t, table in enumerate(tables):
        k = table.players.size
        for leaf in range(table.values.shape[0]):
            c = np.zeros(p)
            c[offsets[t] + leaf * k : offsets[t] + (leaf + 1) * k] = 1.0
            rows.append(c)
    constraint = np.array(rows)
    totals = constraint @ theta0

    m = constraint.shape[0]
    kkt = np.block([[gram, constraint.T], [constraint, np.zeros((m, m))]])
    solution = np.linalg.lstsq(kkt, np.concatenate([rhs, totals]), rcond=None)[0][:p]

    return [
        ShapTable(players=table.players, values=solution[offsets[t] : offsets[t + 1]].reshape(table.values.shape),
                  expected_value=table.expected_value)
        for t, table in enumerate(tables)
    ]
