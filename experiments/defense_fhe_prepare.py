"""What the defences cost under FHE, part 1 (any OS): the variant models.

- grid-p: hidden edges. The client encodes each feature on a public p-bit grid
  over the dataset's range and every threshold moves to its nearest grid edge.
- bits-k: coarser outputs, the same student quantised to k output bits.

Noise needs no new model; defense_fhe.py compiles it from model.json. Writes
experiments/e2e/<dataset>/variants/<name>.json and variants.json with the
honest-client cost against the deployed model on the test split. Needs
model.json from e2e_prepare.py.

    uv run python experiments/defense_fhe_prepare.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from e2e_prepare import N_BINS, SEED, student
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from cipherforest.quantize import FeatureQuantizer, QuantizedModel, quantize_model, to_grid

E2E = Path(__file__).resolve().parent / "e2e"
DATASETS = ("adult", "credit_default")
GRID_BITS = (8, 10, 12, 14, 16)  # Concrete looks up at most 16 bits
OUTPUT_BITS = (6, 4)


def run(dataset: str) -> dict[str, dict[str, float]]:
    x, y = load_dataset(dataset)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.25, stratify=y, random_state=SEED)
    deployed = QuantizedModel.from_json((E2E / dataset / "model.json").read_text())
    want_pred, want_shap = deployed.decode(deployed.simulate(deployed.quantizer.transform(x_te)))

    variants = {f"grid-{p}": to_grid(deployed, x.min(axis=0), x.max(axis=0), p) for p in GRID_BITS}
    quantizer = FeatureQuantizer.fit(x_tr, N_BINS)
    _, ensemble, tables = student(x_tr, y_tr, quantizer, np.random.default_rng(SEED))
    variants |= {f"bits-{k}": quantize_model(ensemble, tables, quantizer, k) for k in OUTPUT_BITS}

    out = E2E / dataset / "variants"
    out.mkdir(exist_ok=True)
    metrics = {}
    for name, model in variants.items():
        (out / f"{name}.json").write_text(model.to_json())
        pred, shap = model.decode(model.simulate(model.quantizer.transform(x_te)))
        metrics[name] = {
            "input_bits": model.quantizer.input_bits,
            "output_bits": model.output_bits,
            "auc": float(roc_auc_score(y_te, pred)),
            "label_agreement": float(np.mean((pred > 0.5) == (want_pred > 0.5))),
            "max_shap_change": float(np.abs(shap - want_shap).max()),
            # rows whose explanation changes at all; grid models keep the same tables
            "rows_changed": float(np.mean(np.any(np.abs(shap - want_shap) > 1e-12, axis=1))),
        }
    (E2E / dataset / "variants.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    for dataset in DATASETS:
        print(dataset, json.dumps(run(dataset), indent=2), flush=True)
