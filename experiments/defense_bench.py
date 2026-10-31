"""Server-side noise: what it costs an honest client and what it costs an attacker.

An honest client asks once and reads a noisy prediction and SHAP vector. The
attacker asks every input r times and rounds the means back onto the integer
grid, doubling r until its exact copy passes its own check; every query of every
attempt counts. Needs experiments/e2e/<dataset>/model.json from e2e_prepare.py.
About three hours on a laptop, almost all of it Taiwan at noise 4.

    uv run python experiments/defense_bench.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from cipherforest.extraction import Cached, Counted, code_values, denoised, extract, noisy_oracle, plaintext_oracle, shap_interactions
from cipherforest.quantize import QuantizedModel

E2E = Path(__file__).resolve().parent / "e2e"
OUT = Path(__file__).resolve().parent / "results" / "defense_bench.json"
DATASETS = ("adult", "credit_default")
NOISE = (0.0, 0.5, 1.0, 2.0, 4.0)  # Laplace scale in output units (1 / scale)
SEEDS = range(2)
MAX_REPEATS = 1 << 14
N_EVAL = 5000
SPLIT_SEED = 0  # as in e2e_prepare.py


def honest_client(model: QuantizedModel, x: np.ndarray, y: np.ndarray, noise: float, seed: int) -> dict[str, float]:
    clean_pred, clean = plaintext_oracle(model)(x)
    pred, shap = noisy_oracle(model, noise, np.random.default_rng(seed))(x)
    norms = np.linalg.norm(shap, axis=1) * np.linalg.norm(clean, axis=1)
    keep = norms > 0
    cosine = np.sum(shap * clean, axis=1)[keep] / norms[keep]
    top = lambda s: np.argsort(-np.abs(s), axis=1)[:, :3]  # noqa: E731
    overlap = [len(set(a) & set(b)) / 3 for a, b in zip(top(shap), top(clean), strict=True)]
    return {
        "auc": float(roc_auc_score(y, pred)),
        "auc_clean": float(roc_auc_score(y, clean_pred)),
        "shap_cosine": float(cosine.mean()),
        "top3_overlap": float(np.mean(overlap)),
        "max_pred_shift": float(np.abs(pred - clean_pred).max()),
    }


def attacker(model: QuantizedModel, noise: float, seed: int) -> dict[str, object]:
    total, repeats = 0, 1
    while repeats <= MAX_REPEATS:
        counter = Counted(noisy_oracle(model, noise, np.random.default_rng(10_000 + seed)))
        oracle = Cached(denoised(counter, repeats, model.scale, model.base_value))
        copy, _ = extract(oracle, model.quantizer, np.random.default_rng(seed), shap_interactions)
        total += counter.queries
        if copy is not None:
            rng = np.random.default_rng(1000 + seed)
            x = np.column_stack([rng.choice(code_values(e), size=N_EVAL) for e in model.quantizer.edges])
            got, want = np.column_stack(copy(x)), np.column_stack(plaintext_oracle(model)(x))
            exact = float(np.mean(np.all(np.abs(got - want) < 1e-9, axis=1)))
            return {"repeats": repeats, "queries": total, "exact_rows": exact}
        repeats *= 2
    return {"repeats": None, "queries": total, "exact_rows": 0.0}


def run(dataset: str, rows: list[dict[str, object]]) -> None:
    """Appends to rows and rewrites OUT after each one; the full run takes hours."""
    model = QuantizedModel.from_json((E2E / dataset / "model.json").read_text())
    x, y = load_dataset(dataset)
    _, x_te, _, y_te = train_test_split(x, y, test_size=0.25, stratify=y, random_state=SPLIT_SEED)
    for noise in NOISE:
        for seed in SEEDS:
            rows.append(
                {
                    "dataset": dataset,
                    "noise": noise,
                    "seed": seed,
                    **honest_client(model, x_te, y_te, noise, seed),
                    **attacker(model, noise, seed),
                }
            )
            OUT.write_text(json.dumps(rows, indent=1))
            print(dataset, noise, seed, rows[-1]["repeats"], rows[-1]["queries"], flush=True)


if __name__ == "__main__":
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        run(dataset, rows)
    print("dataset         noise  AUC (clean)     SHAP cos  top-3   attacker repeats  queries (median)  exact rows (min)")
    for dataset in DATASETS:
        for noise in NOISE:
            sel = [r for r in rows if r["dataset"] == dataset and r["noise"] == noise]
            mean = lambda k: np.mean([float(r[k]) for r in sel])  # noqa: E731
            reps = sorted({r["repeats"] for r in sel}, key=lambda v: (v is None, v or 0))
            print(
                f"{dataset:15} {noise:5}  {mean('auc'):.3f} ({mean('auc_clean'):.3f})   {mean('shap_cosine'):.3f}"
                f"   {mean('top3_overlap'):.2f}   {str(reps):16}  {int(np.median([r['queries'] for r in sel])):>12}"
                f"  {min(float(r['exact_rows']) for r in sel):.4f}"
            )
