"""Encrypted tree inference with exact SHAP explanations over TFHE.

The model preparation below works on any OS. compile_model, export, Client and
Server need concrete-python (Linux or macOS) and load it only when first used.
"""

from __future__ import annotations

from typing import Any

from cipherforest.api import distill, explain, from_catboost, from_ensemble
from cipherforest.quantize import FeatureQuantizer, QuantizedModel

__version__ = "0.2.0"

_FHE = {
    "compile_model": "cipherforest.circuit",
    "SAFE": "cipherforest.circuit",
    "export": "cipherforest.deploy",
    "Client": "cipherforest.deploy",
    "Server": "cipherforest.deploy",
}

__all__ = ["FeatureQuantizer", "QuantizedModel", "distill", "explain", "from_catboost", "from_ensemble", *_FHE]


def __getattr__(name: str) -> Any:
    if name in _FHE:
        import importlib

        return getattr(importlib.import_module(_FHE[name]), name)
    raise AttributeError(f"module 'cipherforest' has no attribute {name!r}")
