"""Command line: a CatBoost model to server and client files without writing code.

    cipherforest prepare model.cbm --train x_train.csv --out model.json   # any OS
    cipherforest explain model.json --data x_test.csv                     # any OS
    cipherforest compile model.json --out artifacts                       # Linux or macOS

Data files are numeric .csv (an optional header row is skipped) or .npy, one row
per sample, columns in the order the model was trained on.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from cipherforest.api import explain, from_catboost
from cipherforest.quantize import QuantizedModel


def load_rows(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        rows = np.load(path)
    else:
        try:
            rows = np.loadtxt(path, delimiter=",", ndmin=2)
        except ValueError:
            rows = np.loadtxt(path, delimiter=",", ndmin=2, skiprows=1)
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2:
        raise SystemExit(f"{path}: expected one row per sample")
    if np.isnan(rows).any():
        raise SystemExit(f"{path}: missing values are not supported")
    return rows


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


def bootstraps(model: QuantizedModel) -> int:
    return int(sum(t.features.size + t.players.size for t in model.trees))


def prepare(args: argparse.Namespace) -> None:
    x_train = load_rows(args.train)
    source, catboost = load_catboost(args.model)
    rng = np.random.default_rng(args.seed)
    background = x_train[rng.choice(len(x_train), size=min(args.background, len(x_train)), replace=False)]
    model = from_catboost(source, background=background, x_train=x_train, n_bins=args.bins, output_bits=args.output_bits)
    args.out.write_text(model.to_json())

    depths = sorted({t.features.size for t in model.trees})
    used = sorted({int(f) for t in model.trees for f in t.features})
    print(f"{len(model.trees)} trees of depth {'/'.join(map(str, depths))}, {len(used)} of "
          f"{model.quantizer.n_features} features used, about {bootstraps(model)} bootstraps per request")
    if catboost is not None:
        sample = x_train[: min(1000, len(x_train))]
        score, _ = explain(model, sample)
        raw = np.asarray(catboost.predict(sample, prediction_type="RawFormulaVal"), dtype=np.float64)
        print(f"largest difference from CatBoost's raw score on {len(sample)} training rows: "
              f"{np.abs(score - raw).max():.2g} (bound {len(model.trees) / (2 * model.scale):.2g})")
    print(f"wrote {args.out}")


def explain_rows(args: argparse.Namespace) -> None:
    model = QuantizedModel.from_json(args.model.read_text())
    score, shap = explain(model, load_rows(args.data))
    out = sys.stdout
    out.write(",".join(["score", *(f"shap_{i}" for i in range(shap.shape[1]))]) + "\n")
    for s, row in zip(score, shap, strict=True):
        out.write(",".join(f"{v:.6g}" for v in (s, *row)) + "\n")


def compile_files(args: argparse.Namespace) -> None:
    try:
        from cipherforest.circuit import compile_model
        from cipherforest.deploy import export
    except ImportError as error:
        raise SystemExit(f"compile needs concrete-python (Linux or macOS): {error}") from error
    model = QuantizedModel.from_json(args.model.read_text())
    cut = model.cut_for(args.threshold) if args.output == "decision" else None
    export(compile_model(model, output=args.output, cut=cut), model, args.out)
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
    p.add_argument("--output-bits", type=int, default=8)
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
    p.set_defaults(run=compile_files)

    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
