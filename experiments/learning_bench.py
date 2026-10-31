"""Attackers that learn from noisy answers instead of repeating each input.

Same server noise as defense_bench.py. Every attacker asks n random inputs once
and fits a copy:

- tables, pred+shap: least squares over the clique tables (fit_tables) on the
  SHAP outputs; prediction = base + sum of SHAP. Gets the thresholds and the
  interaction graph for free, which only makes it stronger.
- tables, pred: the same on the prediction alone.
- gbdt, pred: a gradient-boosted regressor on the prediction, no structure.

The copy is scored on the e2e test split against the clean model. Needs
experiments/e2e/<dataset>/model.json from e2e_prepare.py.

    uv run python experiments/learning_bench.py
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

from cipherforest.extraction import CellGrid, code_values, fit_tables, maximal_cliques, noisy_oracle, plaintext_oracle, true_thresholds
from cipherforest.quantize import QuantizedModel

E2E = Path(__file__).resolve().parent / "e2e"
OUT = Path(__file__).resolve().parent / "results" / "learning_bench.json"
DATASETS = ("adult", "credit_default")
NOISE = (0.0, 0.5, 1.0, 2.0)
BUDGETS = [1 << k for k in range(10, 21, 2)]
SEEDS = range(2)
SPLIT_SEED = 0  # as in e2e_prepare.py


def on_grid(model: QuantizedModel, values: np.ndarray, offset: float = 0.0) -> np.ndarray:
    return offset + np.rint((values - offset) * model.scale) / model.scale


def score(model: QuantizedModel, x: np.ndarray, pred: np.ndarray, shap: np.ndarray | None) -> dict[str, float]:
    clean_pred, clean_shap = plaintext_oracle(model)(x)
    exact = np.abs(on_grid(model, pred, model.base_value) - clean_pred) < 1e-9
    if shap is not None:
        exact &= np.all(np.abs(on_grid(model, shap) - clean_shap) < 1e-9, axis=1)
    return {
        "label_agreement": float(np.mean((pred > 0.5) == (clean_pred > 0.5))),
        "pred_error": float(np.mean(np.abs(pred - clean_pred)) * model.scale),  # in output units
        "exact_rows": float(np.mean(exact)),
    }


def run(dataset: str, rows: list[dict[str, object]]) -> None:
    """Appends to rows and rewrites OUT after each noise level."""
    model = QuantizedModel.from_json((E2E / dataset / "model.json").read_text())
    x, y = load_dataset(dataset)
    _, x_te, _, _ = train_test_split(x, y, test_size=0.25, stratify=y, random_state=SPLIT_SEED)
    grid = CellGrid(model.quantizer, true_thresholds(model))
    graph = {(min(f, g), max(f, g)) for t in model.trees for f, g in combinations({int(v) for v in t.features}, 2)}
    cliques = maximal_cliques(range(grid.sizes.size), graph)
    n = max(BUDGETS)

    for noise in NOISE:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            server = noisy_oracle(model, noise, np.random.default_rng(10_000 + seed))
            # the table attackers know the cells; the gbdt one only the edges
            x_cells = grid.inputs(np.column_stack([rng.integers(0, s, size=n) for s in grid.sizes]))
            x_codes = np.column_stack([rng.choice(code_values(e), size=n) for e in model.quantizer.edges])
            pred_cells, shap_cells = server(x_cells)
            pred_codes, _ = server(x_codes)
            for budget in BUDGETS:
                copy = fit_tables(grid, cliques, x_cells[:budget], np.column_stack([pred_cells, shap_cells])[:budget])
                _, shap = copy(x_te)
                shap = on_grid(model, shap)
                fits = {"tables, pred+shap": score(model, x_te, model.base_value + shap.sum(axis=1), shap)}

                only = np.column_stack([pred_cells[:budget], np.zeros((budget, grid.sizes.size))])
                fits["tables, pred"] = score(model, x_te, fit_tables(grid, cliques, x_cells[:budget], only)(x_te)[0], None)

                gbdt = HistGradientBoostingRegressor(max_iter=300, random_state=seed)
                gbdt.fit(x_codes[:budget], pred_codes[:budget])
                fits["gbdt, pred"] = score(model, x_te, gbdt.predict(x_te), None)

                for name, metrics in fits.items():
                    rows.append({"dataset": dataset, "noise": noise, "seed": seed, "queries": budget, "attacker": name, **metrics})
            print(dataset, noise, seed, flush=True)
        OUT.write_text(json.dumps(rows, indent=1))


def first(sel: list[dict[str, object]], key: str, target: float) -> str:
    """Smallest budget at which every seed reaches target."""
    for budget in BUDGETS:
        if min(float(r[key]) for r in sel if r["queries"] == budget) >= target:
            return str(budget)
    return f">{max(BUDGETS)}"


if __name__ == "__main__":
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        run(dataset, rows)
    print("dataset         attacker           noise  99% labels  exact  labels / error at max budget")
    for dataset in DATASETS:
        for name in ("tables, pred+shap", "tables, pred", "gbdt, pred"):
            for noise in NOISE:
                sel = [r for r in rows if r["dataset"] == dataset and r["attacker"] == name and r["noise"] == noise]
                top = [r for r in sel if r["queries"] == max(BUDGETS)]
                print(
                    f"{dataset:15} {name:18} {noise:5}  {first(sel, 'label_agreement', 0.99):>10}"
                    f"  {first(sel, 'exact_rows', 1.0):>8}"
                    f"  {np.mean([r['label_agreement'] for r in top]):.4f} / {np.mean([r['pred_error'] for r in top]):.2f}"
                )
