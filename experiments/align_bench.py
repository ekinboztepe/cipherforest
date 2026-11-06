"""Explanation tables fitted to the teacher's SHAP values, against the student's exact ones.

Teacher: the 100-tree forest of distill_bench. Student: our distilled oblivious
ensemble, depth 6, T trees. Both are explained against the same 50 background rows
(interventional SHAP; shap.TreeExplainer for the forest). align_tables refits the
student's tables to the teacher's SHAP values on 1000 training rows; agreement is
measured on 200 test rows. The prediction is the same for both tables.

    uv run python experiments/align_bench.py [dataset ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import shap
from datasets import load_dataset
from distill_bench import class1_shap, shap_fidelity
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from cipherforest.align import align_tables
from cipherforest.distill import distill_ensemble
from cipherforest.quantize import FeatureQuantizer
from cipherforest.shap_table import build_shap_table, explain_ensemble

OUT = Path(__file__).resolve().parent / "results" / "align_bench.json"
DATASETS = ("adult", "credit_default", "bank_marketing", "breast_cancer", "credit_g")
SEEDS = (0, 1, 2)
TREES = (3, 10)
N_BACKGROUND, N_FIT, N_EVAL = 50, 1000, 200
RIDGES = (0.1, 1.0, 10.0)


def run(dataset: str, seed: int) -> list[dict[str, object]]:
    x, y = load_dataset(dataset)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.25, stratify=y, random_state=seed)
    rng = np.random.default_rng(seed)
    background = x_tr[rng.choice(len(x_tr), size=N_BACKGROUND, replace=False)]
    fit_rows = x_tr[rng.choice(len(x_tr), size=min(N_FIT, len(x_tr)), replace=False)]
    eval_rows = x_te[rng.choice(len(x_te), size=min(N_EVAL, len(x_te)), replace=False)]

    teacher = RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=4, random_state=seed).fit(x_tr, y_tr)
    explainer = shap.TreeExplainer(teacher, data=background, feature_perturbation="interventional")
    phi_fit = class1_shap(explainer.shap_values(fit_rows, check_additivity=False))
    phi_eval = class1_shap(explainer.shap_values(eval_rows, check_additivity=False))
    edges = FeatureQuantizer.fit(x_tr, n_bins=32).edges

    rows = []
    for n_trees in TREES:
        student = distill_ensemble(x_tr, teacher.predict_proba(x_tr)[:, 1], 6, n_trees, edges)
        tables = [build_shap_table(tree, background) for tree in student.trees]
        own = explain_ensemble(student, tables, eval_rows, x.shape[1])
        base = {"dataset": dataset, "seed": seed, "trees": n_trees,
                "auc_teacher": float(roc_auc_score(y_te, teacher.predict_proba(x_te)[:, 1])),
                "auc_student": float(roc_auc_score(y_te, student.predict(x_te)))}
        rows.append({**base, "tables": "exact", **shap_fidelity(own, phi_eval)})
        for ridge in RIDGES:
            aligned = align_tables(student, tables, fit_rows, phi_fit, ridge=ridge)
            new = explain_ensemble(student, aligned, eval_rows, x.shape[1])
            assert np.allclose(new.sum(axis=1), own.sum(axis=1)), "prediction moved"
            rows.append({**base, "tables": f"aligned, ridge {ridge}", **shap_fidelity(new, phi_eval)})
    return rows


if __name__ == "__main__":
    rows = json.loads(OUT.read_text()) if OUT.exists() else []
    for dataset in sys.argv[1:] or DATASETS:
        for seed in SEEDS:
            if any(r["dataset"] == dataset and r["seed"] == seed for r in rows):
                continue
            new = run(dataset, seed)
            rows += new
            OUT.write_text(json.dumps(rows, indent=1))
            for r in new:
                print(dataset, seed, r["trees"], r["tables"], round(r["shap_cosine"], 3), round(r["shap_top3"], 3), flush=True)
