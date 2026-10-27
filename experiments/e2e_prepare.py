"""End-to-end demo, part 1 (any OS): train, distil, quantise, export.

Writes experiments/e2e/<dataset>/{model.json, samples.npz, plain.json}.
Part 2, the encrypted run, is e2e_fhe.py inside the Docker image.

    uv run python experiments/e2e_prepare.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from cipherforest.distill import distill_ensemble
from cipherforest.quantize import FeatureQuantizer, quantize_model
from cipherforest.shap_table import build_shap_table, explain_ensemble

OUT = Path(__file__).resolve().parent / "e2e"
DATASETS = ("adult", "credit_default")
DEPTH, TREES, OUTPUT_BITS, N_BINS = 6, 3, 8, 32
N_BACKGROUND, N_SAMPLES, SEED = 100, 20, 0


def prepare(name: str) -> dict[str, float]:
    x, y = load_dataset(name)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.25, stratify=y, random_state=SEED)
    rng = np.random.default_rng(SEED)

    teacher = RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=SEED).fit(x_tr, y_tr)
    quantizer = FeatureQuantizer.fit(x_tr, N_BINS)
    ensemble = distill_ensemble(x_tr, teacher.predict_proba(x_tr)[:, 1], DEPTH, TREES, quantizer.edges)
    background = x_tr[rng.choice(len(x_tr), size=N_BACKGROUND, replace=False)]
    tables = [build_shap_table(tree, background) for tree in ensemble.trees]
    model = quantize_model(ensemble, tables, quantizer, OUTPUT_BITS)

    codes = quantizer.transform(x_te)
    for float_tree, int_tree in zip(ensemble.trees, model.trees, strict=True):
        if not np.array_equal(float_tree.leaf_index(x_te), int_tree.leaf_index(codes)):
            raise AssertionError("integer routing diverged from float routing")
    student = ensemble.predict(x_te)
    quant_pred, quant_shap = model.decode(model.simulate(codes))
    exact_shap = explain_ensemble(ensemble, tables, x_te, x.shape[1])

    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.json").write_text(model.to_json())
    pick = rng.choice(len(x_te), size=N_SAMPLES, replace=False)
    np.savez(
        out / "samples.npz",
        x=x_te[pick],
        student_pred=student[pick],
        student_shap=exact_shap[pick],
        teacher_prob=teacher.predict_proba(x_te[pick])[:, 1],
    )
    metrics = {
        "n_test": int(len(y_te)),
        "n_features": int(x.shape[1]),
        "features_used": int(len({int(f) for t in ensemble.trees for f in t.features})),
        "input_bits": quantizer.input_bits,
        "scale": model.scale,
        "auc_teacher": float(roc_auc_score(y_te, teacher.predict_proba(x_te)[:, 1])),
        "auc_student_float": float(roc_auc_score(y_te, student)),
        "auc_student_quantized": float(roc_auc_score(y_te, quant_pred)),
        "max_pred_error_quantized": float(np.abs(quant_pred - student).max()),
        "max_shap_error_quantized": float(np.abs(quant_shap - exact_shap).max()),
        "max_abs_shap": float(np.abs(exact_shap).max()),
    }
    (out / "plain.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    for dataset in DATASETS:
        print(dataset, json.dumps(prepare(dataset), indent=2))
