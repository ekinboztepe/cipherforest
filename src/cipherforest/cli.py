"""Command line: a CatBoost model to server and client files without writing code.

    cipherforest prepare model.cbm --train x_train.csv --out model.json   # any OS
    cipherforest explain model.json --data x_test.csv                     # any OS
    cipherforest compile model.json --out artifacts                       # Linux or macOS

Data files are .csv (an optional header row is skipped, empty fields are missing
values, categorical columns are read as text) or .npy, one row per sample, columns
in the order the model was trained on.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from cipherforest.api import explain, from_catboost
from cipherforest.quantize import OUTPUT_BITS, MultiClassModel, QuantizedModel, model_from_json


def _is_header(line: str) -> bool:
    for field in line.split(","):
        field = field.strip()
        if field and field.lower() not in ("nan", "na"):
            try:
                float(field)
            except ValueError:
                return True
    return False


def load_rows(path: Path, categorical: frozenset[int] = frozenset()) -> np.ndarray:
    """Empty fields, "nan" and "NA" are missing values; categorical columns stay strings."""
    if path.suffix == ".npy":
        rows = np.load(path, allow_pickle=bool(categorical))
    elif categorical:
        import csv

        with path.open(newline="") as f:
            lines = [line for line in csv.reader(f) if line]
        if lines and _is_header(",".join(v for i, v in enumerate(lines[0]) if i not in categorical)):
            lines = lines[1:]

        def number(v: str) -> float:
            return float("nan") if v.strip() in ("", "NA") else float(v)

        rows = np.array([[v if i in categorical else number(v) for i, v in enumerate(line)] for line in lines],
                        dtype=object)
        return rows
    else:
        with path.open() as f:
            header = _is_header(f.readline())
        rows = np.genfromtxt(path, delimiter=",", skip_header=int(header), missing_values=("", "NA"),
                             filling_values=np.nan, ndmin=2)
    rows = np.asarray(rows, dtype=object if categorical else np.float64)
    if rows.ndim != 2:
        raise SystemExit(f"{path}: expected one row per sample")
    return rows


def categorical_columns(model: QuantizedModel) -> frozenset[int]:
    return frozenset(f for f, c in enumerate(model.quantizer.categories) if c is not None)


def load_catboost(path: Path) -> tuple[object, object]:
    """(what from_catboost takes, a CatBoost model to check against or None)."""
    if path.suffix == ".json":
        return path.read_text(), None
    try:
        from catboost import CatBoost
    except ImportError as error:
        raise SystemExit("reading .cbm needs catboost; pip install 'cipherforest[catboost]' or pass the JSON export") from error
    model = CatBoost()
    model.load_model(str(path))
    return model, model


def bootstraps(model: QuantizedModel | MultiClassModel) -> int:
    """Per request with the SHAP output: comparisons once, lookups once per class."""
    classes = len(model.classes) if isinstance(model, MultiClassModel) else 1
    return int(sum(t.features.size + classes * t.players.size for t in model.trees))


def rounding_bound(model: QuantizedModel | MultiClassModel) -> float:
    models = model.classes if isinstance(model, MultiClassModel) else (model,)
    return max(len(m.trees) / (2 * m.scale) for m in models)


def prepare(args: argparse.Namespace) -> None:
    source, catboost = load_catboost(args.model)
    cat = frozenset(int(i) for i in catboost.get_cat_feature_indices()) if catboost is not None else frozenset()
    x_train = load_rows(args.train, cat)
    rng = np.random.default_rng(args.seed)
    background = x_train[rng.choice(len(x_train), size=min(args.background, len(x_train)), replace=False)]
    model = from_catboost(source, background=background, x_train=x_train, n_bins=args.bins, output_bits=args.output_bits)
    args.out.write_text(model.to_json())

    depths = sorted({t.features.size for t in model.trees})
    used = sorted({int(f) for t in model.trees for f in t.features})
    classes = f", {len(model.classes)} classes" if isinstance(model, MultiClassModel) else ""
    print(f"{len(model.trees)} trees of depth {'/'.join(map(str, depths))}{classes}, {len(used)} of "
          f"{model.quantizer.n_features} features used, about {bootstraps(model)} bootstraps per request")
    if catboost is not None:
        sample = x_train[: min(1000, len(x_train))]
        score, _ = explain(model, sample)
        from catboost import Pool

        pool = Pool(sample.tolist(), cat_features=sorted(cat)) if cat else sample
        raw = np.asarray(catboost.predict(pool, prediction_type="RawFormulaVal"), dtype=np.float64)
        print(f"largest difference from CatBoost's raw score on {len(sample)} training rows: "
              f"{np.abs(score - raw).max():.2g} (bound {rounding_bound(model):.2g})")
    print(f"wrote {args.out}")


def explain_rows(args: argparse.Namespace) -> None:
    model = model_from_json(args.model.read_text())
    score, shap = explain(model, load_rows(args.data, categorical_columns(model)))
    if score.ndim == 1:
        score, shap = score[:, None], shap[:, None, :]
    classes = score.shape[1]
    suffix = [f"_{k}" if classes > 1 else "" for k in range(classes)]
    header = [f"score{s}" for s in suffix] + [f"shap{s}_{i}" for s in suffix for i in range(shap.shape[2])]
    out = sys.stdout
    out.write(",".join(header) + "\n")
    for s, row in zip(score, shap, strict=True):
        out.write(",".join(f"{v:.6g}" for v in (*s, *row.ravel())) + "\n")


def compile_files(args: argparse.Namespace) -> None:
    try:
        from cipherforest.circuit import compile_model
        from cipherforest.deploy import export
    except ImportError as error:
        raise SystemExit(f"compile needs concrete-python (Linux or macOS): {error}") from error
    model = model_from_json(args.model.read_text())
    if isinstance(model, MultiClassModel) and args.output == "decision":
        raise SystemExit("a multiclass model has shap and score outputs only")
    cut = model.cut_for(args.threshold) if args.output == "decision" else None
    export(compile_model(model, output=args.output, cut=cut, public_splits=args.public_splits), model, args.out)
    print(f"wrote {args.out / 'server.zip'} for the server and {args.out / 'client.json'} for clients")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cipherforest", description="Encrypted tree inference with exact SHAP.")
    commands = parser.add_subparsers(required=True, metavar="command")

    p = commands.add_parser("prepare", help="CatBoost model (.cbm or JSON export) to model.json")
    p.add_argument("model", type=Path)
    p.add_argument("--train", type=Path, required=True, help="training rows: edges and the SHAP background")
    p.add_argument("--out", type=Path, default=Path("model.json"))
    p.add_argument("--background", type=int, default=100, help="training rows SHAP compares against")
    p.add_argument("--bins", type=int, default=32, help="training quantiles added to the edges")
    p.add_argument("--output-bits", type=int, default=OUTPUT_BITS,
                   help="width of each output; 8 makes score and decision circuits faster and less precise")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(run=prepare)

    p = commands.add_parser("explain", help="score and SHAP values in the clear, as CSV")
    p.add_argument("model", type=Path)
    p.add_argument("--data", type=Path, required=True)
    p.set_defaults(run=explain_rows)

    p = commands.add_parser("compile", help="model.json to server.zip and client.json")
    p.add_argument("model", type=Path)
    p.add_argument("--out", type=Path, default=Path("artifacts"))
    p.add_argument("--output", choices=("shap", "score", "decision"), default="shap")
    p.add_argument("--threshold", type=float, default=0.0,
                   help="decision only: approve when the raw score (log-odds for a classifier) is above this")
    p.add_argument("--public-splits", action="store_true",
                   help="put the thresholds in client.json and let clients compare: faster, reveals the structure")
    p.set_defaults(run=compile_files)

    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
