"""How much does a TFHE-sized oblivious student lose against a 100-tree forest?

Models, all on the same held-out split:
  teacher     random forest, 100 trees, max_depth 12
  cb_full     CatBoost, 500 trees (strong reference, far too big for TFHE)
  obl_distill our distiller on teacher probabilities
  obl_votes   our distiller on per-tree hard-vote fractions
  cb_distill  CatBoost regressor (also oblivious) on teacher probabilities
  cb_labels   CatBoost with the same D/T on the labels: "why distil at all?"
  dt_labels   plain depth-D decision tree on the labels

SHAP fidelity compares obl_distill with the teacher's interventional SHAP on the
same background rows.

    uv run python experiments/distill_bench.py --datasets breast_cancer credit_g --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier, CatBoostRegressor
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from cipherforest.distill import distill_ensemble
from cipherforest.quantize import FeatureQuantizer
from cipherforest.shap_table import build_shap_table, explain_ensemble

RESULTS = Path(__file__).resolve().parent / "results" / "distill_bench.json"
DEPTHS = (4, 5, 6)
TREE_COUNTS = (1, 3)
N_BACKGROUND = 50
N_SHAP_ROWS = 100


def class1_shap(values: object) -> np.ndarray:
    # shap returns a list or an (n, d, 2) array for binary classifiers depending on version
    if isinstance(values, list):
        return np.asarray(values[1], dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64)
    return arr[:, :, 1] if arr.ndim == 3 else arr


def shap_fidelity(student: np.ndarray, teacher: np.ndarray) -> dict[str, float]:
    eps = 1e-12
    cos = (student * teacher).sum(1) / (np.linalg.norm(student, axis=1) * np.linalg.norm(teacher, axis=1) + eps)
    rel_l1 = np.abs(student - teacher).sum(1) / (np.abs(teacher).sum(1) + eps)
    top_s = np.argsort(-np.abs(student), axis=1)[:, :3]
    top_t = np.argsort(-np.abs(teacher), axis=1)[:, :3]
    top3 = [len(set(a) & set(b)) / 3 for a, b in zip(top_s, top_t, strict=True)]
    return {"shap_cosine": float(cos.mean()), "shap_rel_l1": float(rel_l1.mean()), "shap_top3": float(np.mean(top3))}


def run_dataset(name: str, seed: int) -> list[dict[str, object]]:
    x, y = load_dataset(name)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.25, stratify=y, random_state=seed)
    rng = np.random.default_rng(seed)
    background = x_tr[rng.choice(len(x_tr), size=min(N_BACKGROUND, len(x_tr)), replace=False)]
    shap_rows = x_te[rng.choice(len(x_te), size=min(N_SHAP_ROWS, len(x_te)), replace=False)]

    teacher = RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=seed).fit(x_tr, y_tr)
    p_tr = teacher.predict_proba(x_tr)[:, 1]
    p_te = teacher.predict_proba(x_te)[:, 1]
    votes_tr = np.mean([est.predict(x_tr) for est in teacher.estimators_], axis=0)
    explainer = shap.TreeExplainer(teacher, data=background, feature_perturbation="interventional")
    teacher_phi = class1_shap(explainer.shap_values(shap_rows, check_additivity=False))
    edges = FeatureQuantizer.fit(x_tr, n_bins=32).edges

    rows: list[dict[str, object]] = []

    def record(model: str, depth: int | None, trees: int | None, score: np.ndarray, **extra: float) -> None:
        rows.append(
            {
                "dataset": name,
                "seed": seed,
                "model": model,
                "depth": depth,
                "trees": trees,
                "auc": float(roc_auc_score(y_te, score)),
                "acc": float(np.mean((score > 0.5) == y_te)),
                "fidelity": float(np.mean((score > 0.5) == (p_te > 0.5))),
                **extra,
            }
        )

    def catboost(make: Callable[[], CatBoostClassifier | CatBoostRegressor], target: np.ndarray) -> np.ndarray:
        model = make().fit(x_tr, target)
        if isinstance(model, CatBoostClassifier):
            return model.predict_proba(x_te)[:, 1]
        return np.asarray(model.predict(x_te), dtype=np.float64)

    record("teacher", None, 100, p_te)
    record("cb_full", 6, 500, catboost(lambda: CatBoostClassifier(iterations=500, random_seed=seed, verbose=0), y_tr))

    for depth in DEPTHS:
        tree = DecisionTreeClassifier(max_depth=depth, random_state=seed).fit(x_tr, y_tr)
        record("dt_labels", depth, 1, tree.predict_proba(x_te)[:, 1])
        for n_trees in TREE_COUNTS:
            student = distill_ensemble(x_tr, p_tr, depth, n_trees, edges)
            tables = [build_shap_table(t, background) for t in student.trees]
            phi = explain_ensemble(student, tables, shap_rows, x.shape[1])
            used = len({int(f) for t in student.trees for f in t.features})
            record("obl_distill", depth, n_trees, student.predict(x_te), features_used=used, **shap_fidelity(phi, teacher_phi))

            votes = distill_ensemble(x_tr, votes_tr, depth, n_trees, edges)
            record("obl_votes", depth, n_trees, votes.predict(x_te))

            common = {"depth": depth, "iterations": n_trees, "learning_rate": 1.0, "border_count": 32}
            record("cb_distill", depth, n_trees, catboost(lambda: CatBoostRegressor(**common, random_seed=seed, verbose=0), p_tr))
            record("cb_labels", depth, n_trees, catboost(lambda: CatBoostClassifier(**common, random_seed=seed, verbose=0), y_tr))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["breast_cancer", "credit_g"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    args = parser.parse_args()

    rows = [row for name in args.datasets for seed in args.seeds for row in run_dataset(name, seed)]
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(rows, indent=2))

    frame = pd.DataFrame(rows)
    metrics = [c for c in ("auc", "acc", "fidelity", "shap_cosine", "shap_top3", "features_used") if c in frame]
    print(frame.groupby(["dataset", "model", "depth", "trees"], dropna=False)[metrics].mean().round(3).to_string())


if __name__ == "__main__":
    main()
