"""How fast a client copies the decisions of a server that only says approve or decline.

The server is the e2e model behind decision_oracle, as a lender would deploy it
under ECOA Regulation B: the client learns approve or decline, and on a decline
optionally the 4 features that pushed hardest towards it (named and ranked, no
values). Adverse is a score at or below 0.5 on Adult and Bank marketing (no high
income, no subscription) and above 0.5 on Taiwan (default).

The attacker holds client.json, so it knows every quantiser edge and which
features the model uses, and works in rounds. It asks the points its copy is least
sure about, sweeps features through all their codes from them, adds the
least certain of fresh random inputs, and refits a gradient-boosted copy on every
answer so far. With reasons it sweeps a decline's named features; without, random
used features. A second attacker never looks at client.json: it draws inputs evenly
between each feature's lowest and highest edge, sweeps 33 evenly spaced values, and
fits on raw inputs, as it would if the server kept the edges to itself.

The copy is scored on the real test split and on random inputs over all codes, and
against the honest share of the majority answer, which a constant copy gets.

    uv run python experiments/label_bench.py [dataset ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from cipherforest.extraction import Counted, code_values, decision_oracle, plaintext_oracle
from cipherforest.quantize import QuantizedModel

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "label_bench.json"
DATASETS = {"adult": False, "credit_default": True, "bank_marketing": False}  # adverse above 0.5?
SEEDS = range(5)
REASONS = 4
BUDGET = 20_000
STARTS, SWEPT, UNSURE, POOL = 4, 4, 256, 8192
N_RANDOM = 5000


def random_inputs(values: list[np.ndarray], n: int, rng: np.random.Generator) -> np.ndarray:
    return np.column_stack([rng.choice(v, size=n) for v in values])


def fit(codes: np.ndarray, labels: np.ndarray, seed: int) -> HistGradientBoostingClassifier | None:
    if np.unique(labels).size < 2:
        return None
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=seed).fit(codes, labels)


def predict(copy: HistGradientBoostingClassifier | None, codes: np.ndarray, constant: float) -> np.ndarray:
    return np.full(len(codes), constant) if copy is None else copy.predict(codes)


def unsure(copy: HistGradientBoostingClassifier | None, x: np.ndarray, codes: np.ndarray, k: int, rng) -> np.ndarray:
    if copy is None:
        return x[rng.choice(len(x), size=k, replace=False)]
    margin = np.abs(copy.predict_proba(codes)[:, 1] - 0.5)
    return x[np.argsort(margin, kind="stable")[:k]]


def attack(
    model: QuantizedModel, adverse_above: bool, reasons: int, x_real: np.ndarray, seed: int, edges: bool = True
) -> dict[str, object]:
    server = decision_oracle(plaintext_oracle(model), adverse_above, reasons)
    counter = Counted(server)
    rng = np.random.default_rng(seed)
    quantizer = model.quantizer
    if edges:
        values = [code_values(e) for e in quantizer.edges]
        used = sorted({int(f) for tree in model.trees for f in tree.players})  # listed in client.json
        view = quantizer.transform
    else:
        values = [np.linspace(e[0], e[-1] + 1.0, 33) if e.size else np.zeros(1) for e in quantizer.edges]
        used = list(range(quantizer.n_features))
        view = np.asarray

    eval_rng = np.random.default_rng(1000 + seed)
    x_random = random_inputs(values, N_RANDOM, eval_rng)
    want_real, _ = server(x_real)
    want_random, _ = server(x_random)

    xs, labels = [], []
    copy, trace = None, []
    while counter.queries < BUDGET:
        pool = random_inputs(values, POOL, rng)
        starts = unsure(copy, pool, view(pool), STARTS, rng)
        start_labels, start_reasons = counter(starts)
        xs.append(starts)
        labels.append(start_labels)
        for start, named in zip(starts, start_reasons, strict=True):
            features = list(np.flatnonzero(named))[:SWEPT] if reasons else []
            if not features:
                features = list(rng.choice(used, size=min(SWEPT, len(used)), replace=False))
            for f in features:
                rows = np.tile(start, (values[f].size, 1))
                rows[:, f] = values[f]
                xs.append(rows)
                labels.append(counter(rows)[0])
        fresh = random_inputs(values, POOL, rng)
        picked = unsure(copy, fresh, view(fresh), UNSURE, rng)
        xs.append(picked)
        labels.append(counter(picked)[0])

        x_all, y_all = np.vstack(xs), np.concatenate(labels)
        copy = fit(view(x_all), y_all, seed)
        constant = float(np.round(y_all.mean()))
        trace.append(
            {
                "queries": counter.queries,
                "real": float(np.mean(predict(copy, view(x_real), constant) == want_real)),
                "random": float(np.mean(predict(copy, view(x_random), constant) == want_random)),
            }
        )

    def first(key: str, level: float) -> int | None:
        return next((t["queries"] for t in trace if t[key] >= level), None)

    return {
        "attacker": "edges" if edges else "no edges",
        "reasons": reasons,
        "seed": seed,
        "adverse_real": float(want_real.mean()),
        "adverse_random": float(want_random.mean()),
        "majority_real": float(max(want_real.mean(), 1 - want_real.mean())),
        "majority_random": float(max(want_random.mean(), 1 - want_random.mean())),
        "queries_99_real": first("real", 0.99),
        "queries_999_real": first("real", 0.999),
        "queries_99_random": first("random", 0.99),
        "queries_999_random": first("random", 0.999),
        "queries_all_real": first("real", 1.0),
        "queries_all_random": first("random", 1.0),
        "final_real": trace[-1]["real"],
        "final_random": trace[-1]["random"],
        "trace": trace,
    }


def run(dataset: str, rows: list[dict[str, object]]) -> None:
    """Appends each finished attack to rows and to OUT, so a run cut short resumes where it stopped."""
    model = QuantizedModel.from_json((HERE / "e2e" / dataset / "model.json").read_text())
    x, y = load_dataset(dataset)
    _, x_te, _, _ = train_test_split(x, y, test_size=0.25, stratify=y, random_state=0)  # as e2e_prepare
    for edges, reasons in ((True, 0), (True, REASONS), (False, 0)):
        attacker = "edges" if edges else "no edges"
        for seed in SEEDS:
            key = (dataset, attacker, reasons, seed)
            if any((r["dataset"], r["attacker"], r["reasons"], r["seed"]) == key for r in rows):
                continue
            row = {"dataset": dataset, **attack(model, DATASETS[dataset], reasons, x_te, seed, edges)}
            print(*key, row["queries_99_real"], row["queries_all_real"], round(row["final_real"], 4), flush=True)
            rows.append(row)
            OUT.write_text(json.dumps(rows, indent=1))


def median(values: list[int | None]) -> str:
    done = [v for v in values if v is not None]
    if len(done) < len(values) / 2:
        return f"> {BUDGET}"
    padded = sorted(done) + [BUDGET + 1] * (len(values) - len(done))
    return str(int(np.median(padded)))


if __name__ == "__main__":
    rows = json.loads(OUT.read_text()) if OUT.exists() else []
    for dataset in sys.argv[1:] or list(DATASETS):
        run(dataset, rows)
    print("dataset         attacker  reasons  adverse  majority  99% real  99.9% real  all real  all random  final real (min)")
    for dataset in DATASETS:
        for attacker, reasons in (("edges", 0), ("edges", REASONS), ("no edges", 0)):
            sel = [r for r in rows if r["dataset"] == dataset and r["attacker"] == attacker and r["reasons"] == reasons]
            if not sel:
                continue
            print(
                f"{dataset:15} {attacker:8}  {reasons:7}  {np.mean([r['adverse_real'] for r in sel]):7.3f}"
                f"  {np.mean([r['majority_real'] for r in sel]):8.3f}  {median([r['queries_99_real'] for r in sel]):>8}"
                f"  {median([r['queries_999_real'] for r in sel]):>10}  {median([r['queries_all_real'] for r in sel]):>8}"
                f"  {median([r['queries_all_random'] for r in sel]):>10}  {min(r['final_real'] for r in sel):.4f}"
            )
