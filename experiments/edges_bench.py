"""What the quantiser edges protect: training data, the model, neither?

- quantile edges, public: the e2e model as deployed.
- uniform edges, public: same pipeline, edges evenly spaced between the
  training min and max, so they say nothing about the distribution in between.
- quantile edges, hidden: the server quantises raw inputs and the client only
  knows a plausible range per feature (the dataset's min and max).

For each: test AUC, and the queries an attacker needs for an exact copy, with
and without SHAP. For hidden edges also how many used edges it recovers bit for
bit. Needs experiments/e2e/<dataset>/model.json from e2e_prepare.py.

    uv run python experiments/edges_bench.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from e2e_prepare import N_BINS, OUTPUT_BITS, SEED, student
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from cipherforest.extraction import (
    Cached,
    Counted,
    extract,
    extract_hidden,
    plaintext_oracle,
    prediction_interactions,
    prediction_only,
    shap_interactions,
    true_thresholds,
)
from cipherforest.quantize import FeatureQuantizer, QuantizedModel, quantize_model

E2E = Path(__file__).resolve().parent / "e2e"
OUT = Path(__file__).resolve().parent / "results" / "edges_bench.json"
DATASETS = ("adult", "credit_default")
SEEDS = range(5)
N_EVAL = 5000


def uniform_model(x_tr: np.ndarray, y_tr: np.ndarray) -> QuantizedModel:
    """e2e_prepare.py with evenly spaced edges."""
    inner = np.linspace(0.0, 1.0, N_BINS + 1)[1:-1]
    lo, hi = x_tr.min(axis=0), x_tr.max(axis=0)
    quantizer = FeatureQuantizer(tuple(np.unique(a + inner * (b - a)) if b > a else np.zeros(0) for a, b in zip(lo, hi, strict=True)))
    _, ensemble, tables = student(x_tr, y_tr, quantizer, np.random.default_rng(SEED))
    return quantize_model(ensemble, tables, quantizer, OUTPUT_BITS)


def exact_share(copy, view, x: np.ndarray) -> float:
    got, want = np.column_stack(copy(x)), np.column_stack(view(x))
    return float(np.mean(np.all(np.abs(got - want) < 1e-9, axis=1)))


def attack(model: QuantizedModel, hidden: bool, sees_shap: bool, seed: int, x_te: np.ndarray, low: np.ndarray, high: np.ndarray):
    view = plaintext_oracle(model) if sees_shap else prediction_only(plaintext_oracle(model))
    graph = shap_interactions if sees_shap else prediction_interactions
    counter = Counted(view)
    rng = np.random.default_rng(seed)
    row: dict[str, object] = {}
    if hidden:
        copy, found = extract_hidden(Cached(counter), low, high, rng, graph)
        used = {(f, float(model.quantizer.edges[f][k])) for f, k in true_thresholds(model)}
        got = {(f, float(e)) for f, es in enumerate(found) for e in es}
        row |= {"used_edges": len(used), "recovered_edges": len(got & used), "wrong_edges": len(got - used)}
    else:
        copy, _ = extract(Cached(counter), model.quantizer, rng, graph)
    row["queries"] = counter.queries
    if copy is None:
        return row | {"exact_test": 0.0, "exact_range": 0.0}
    x_range = np.random.default_rng(1000 + seed).uniform(low, high, size=(N_EVAL, low.size))
    return row | {"exact_test": exact_share(copy, view, x_te), "exact_range": exact_share(copy, view, x_range)}


def run(dataset: str, rows: list[dict[str, object]]) -> None:
    x, y = load_dataset(dataset)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.25, stratify=y, random_state=SEED)
    low, high = x.min(axis=0), x.max(axis=0)
    models = {
        "quantile": QuantizedModel.from_json((E2E / dataset / "model.json").read_text()),
        "uniform": uniform_model(x_tr, y_tr),
    }
    for edges, hidden in (("quantile", False), ("uniform", False), ("quantile", True)):
        model = models[edges]
        auc = float(roc_auc_score(y_te, plaintext_oracle(model)(x_te)[0]))
        for sees_shap in (True, False):
            for seed in SEEDS:
                rows.append(
                    {
                        "dataset": dataset,
                        "edges": edges + (", hidden" if hidden else ", public"),
                        "attacker": "pred+shap" if sees_shap else "pred",
                        "seed": seed,
                        "auc": auc,
                        "total_edges": int(sum(e.size for e in model.quantizer.edges)),
                        **attack(model, hidden, sees_shap, seed, x_te, low, high),
                    }
                )
                print({k: v for k, v in rows[-1].items() if k != "dataset"}, flush=True)
            OUT.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        run(dataset, rows)
    print("dataset         edges              attacker   AUC    queries (median, max)  exact test / range (min)")
    for dataset in DATASETS:
        for edges in ("quantile, public", "uniform, public", "quantile, hidden"):
            for name in ("pred+shap", "pred"):
                sel = [r for r in rows if r["dataset"] == dataset and r["edges"] == edges and r["attacker"] == name]
                q = [int(r["queries"]) for r in sel]
                print(
                    f"{dataset:15} {edges:18} {name:9}  {sel[0]['auc']:.3f}  {int(np.median(q)):8} {max(q):8}"
                    f"  {min(float(r['exact_test']) for r in sel):.4f} / {min(float(r['exact_range']) for r in sel):.4f}"
                )
