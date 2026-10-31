"""Coarser outputs: fewer output bits per SHAP value.

The attacker copies the coarse deployed model exactly either way, so the
question is how close that copy is to the fine float model the owner trained,
and what the coarsening costs an honest client. Effects that round to zero are
invisible to both. Same pipeline and split as e2e_prepare.py.

    uv run python experiments/coarse_bench.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from e2e_prepare import N_BINS, OUTPUT_BITS, SEED, student
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from cipherforest.extraction import Cached, Counted, extract, plaintext_oracle, prediction_interactions, prediction_only, shap_interactions
from cipherforest.quantize import FeatureQuantizer, QuantizedModel, quantize_model
from cipherforest.shap_table import explain_ensemble

E2E = Path(__file__).resolve().parent / "e2e"
OUT = Path(__file__).resolve().parent / "results" / "coarse_bench.json"
DATASETS = ("adult", "credit_default")
BITS = (8, 6, 5, 4)  # 3 trees need at least 4: each may add a rounding unit
SEEDS = range(5)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    keep = norms > 0
    return float(np.mean(np.sum(a * b, axis=1)[keep] / norms[keep]))


def run(dataset: str, rows: list[dict[str, object]]) -> None:
    x, y = load_dataset(dataset)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.25, stratify=y, random_state=SEED)
    quantizer = FeatureQuantizer.fit(x_tr, N_BINS)
    _, ensemble, tables = student(x_tr, y_tr, quantizer, np.random.default_rng(SEED))
    fine_pred = ensemble.predict(x_te)
    fine_shap = explain_ensemble(ensemble, tables, x_te, x.shape[1])

    for bits in BITS:
        model = quantize_model(ensemble, tables, quantizer, bits)
        if bits == OUTPUT_BITS:
            deployed = QuantizedModel.from_json((E2E / dataset / "model.json").read_text())
            # base_value can differ in the last bit between runs, nothing else
            for a, b in zip(plaintext_oracle(model)(x_te), plaintext_oracle(deployed)(x_te), strict=True):
                assert np.allclose(a, b, rtol=0, atol=1e-12), "pipeline drifted from e2e_prepare.py"
        pred, shap = plaintext_oracle(model)(x_te)
        honest = {
            "scale": model.scale,
            "auc": float(roc_auc_score(y_te, pred)),
            "shap_cosine": cosine(shap, fine_shap),
            "label_agreement": float(np.mean((pred > 0.5) == (fine_pred > 0.5))),
            "pred_error": float(np.mean(np.abs(pred - fine_pred))),
        }
        for sees_shap in (True, False):
            view = plaintext_oracle(model) if sees_shap else prediction_only(plaintext_oracle(model))
            graph = shap_interactions if sees_shap else prediction_interactions
            for seed in SEEDS:
                counter = Counted(view)
                copy, _ = extract(Cached(counter), model.quantizer, np.random.default_rng(seed), graph)
                assert copy is not None, "attack did not converge"
                got, want = np.column_stack(copy(x_te)), np.column_stack(view(x_te))
                rows.append(
                    {
                        "dataset": dataset,
                        "bits": bits,
                        "attacker": "pred+shap" if sees_shap else "pred",
                        "seed": seed,
                        "queries": counter.queries,
                        "exact_test": float(np.mean(np.all(np.abs(got - want) < 1e-9, axis=1))),
                        **honest,
                    }
                )
        print(dataset, bits, honest, flush=True)
        OUT.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        run(dataset, rows)
    print("dataset         bits  AUC    SHAP cos  labels vs fine  pred+shap queries  pred queries  exact (min)")
    for dataset in DATASETS:
        for bits in BITS:
            sel = [r for r in rows if r["dataset"] == dataset and r["bits"] == bits]
            q = {a: int(np.median([r["queries"] for r in sel if r["attacker"] == a])) for a in ("pred+shap", "pred")}
            r = sel[0]
            print(
                f"{dataset:15} {bits:4}  {r['auc']:.3f}  {r['shap_cosine']:.3f}     {r['label_agreement']:.4f}"
                f"          {q['pred+shap']:8}  {q['pred']:12}  {min(float(s['exact_test']) for s in sel):.4f}"
            )
