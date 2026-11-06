"""Serve CatBoost directly, or distil? Accuracy against circuit size, in the clear.

CatBoost trained on the labels and imported with from_catboost, against our
distilled student (forest teacher), at several tree counts and depths. AUC is that
of the integer model the circuit serves (cipherforest.explain), so output rounding
is included. PBS is the circuit's bootstrap count for the SHAP output: per tree
depth + the features it uses, which is exact for this circuit and tracks server time.

    uv run python experiments/catboost_bench.py [dataset ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import cipherforest

OUT = Path(__file__).resolve().parent / "results" / "catboost_bench.json"
DATASETS = ("adult", "credit_default", "bank_marketing", "breast_cancer", "credit_g")
SEEDS = (0, 1, 2)
CATBOOST = [(depth, trees) for depth in (4, 6) for trees in (3, 10, 30, 100)]
DISTILLED = [(6, trees) for trees in (3, 10, 30)]


def pbs(model: cipherforest.QuantizedModel) -> int:
    return int(sum(t.features.size + t.players.size for t in model.trees))


def run(dataset: str, seed: int) -> list[dict[str, object]]:
    x, y = load_dataset(dataset)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.25, stratify=y, random_state=seed)
    background = x_tr[np.random.default_rng(seed).choice(len(x_tr), size=100, replace=False)]
    rows = []
    base = {"dataset": dataset, "seed": seed}

    teacher = RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=4, random_state=seed).fit(x_tr, y_tr)
    rows.append({**base, "model": "forest teacher", "auc": float(roc_auc_score(y_te, teacher.predict_proba(x_te)[:, 1]))})
    target = teacher.predict_proba(x_tr)[:, 1]
    for depth, trees in DISTILLED:
        model = cipherforest.distill(x_tr, target, depth=depth, n_trees=trees, background=background)
        score, _ = cipherforest.explain(model, x_te)
        rows.append({**base, "model": "distilled", "depth": depth, "trees": trees, "pbs": pbs(model), "auc": float(roc_auc_score(y_te, score))})

    for depth, trees in CATBOOST:
        cb = CatBoostClassifier(iterations=trees, depth=depth, verbose=0, random_seed=seed, thread_count=4,
                                allow_writing_files=False).fit(x_tr, y_tr)
        row = {**base, "model": "catboost", "depth": depth, "trees": trees,
               "auc_float": float(roc_auc_score(y_te, cb.predict(x_te, prediction_type="RawFormulaVal")))}
        try:
            model = cipherforest.from_catboost(cb, background=background, x_train=x_tr)
        except ValueError as error:  # too many trees for the output width
            rows.append({**row, "error": str(error)})
            continue
        score, _ = cipherforest.explain(model, x_te)
        rows.append({**row, "pbs": pbs(model), "auc": float(roc_auc_score(y_te, score))})
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
                print(dataset, seed, r["model"], r.get("depth"), r.get("trees"), r.get("pbs"), round(r.get("auc", float("nan")), 4), flush=True)
