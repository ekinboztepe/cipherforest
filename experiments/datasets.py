"""Binary tabular benchmarks: sklearn's breast cancer plus four OpenML sets (cached in .data/)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer

DATA_HOME = Path(__file__).resolve().parent / ".data"
OPENML_IDS = {"adult": 1590, "credit_g": 31, "bank_marketing": 1461, "credit_default": 42477}


def _to_matrix(frame: pd.DataFrame) -> np.ndarray:
    # trees only need an order: ordinal codes for categoricals (-1 = missing), median fill for numbers
    columns = []
    for name in frame.columns:
        col = frame[name]
        if isinstance(col.dtype, pd.CategoricalDtype) or col.dtype == object:
            columns.append(col.astype("category").cat.codes.to_numpy(dtype=np.float64))
        else:
            values = col.to_numpy(dtype=np.float64)
            columns.append(np.where(np.isnan(values), np.nanmedian(values), values))
    return np.column_stack(columns)


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    if name == "breast_cancer":
        data = load_breast_cancer()
        return data.data.astype(np.float64), data.target.astype(np.int64)
    frame, y = _openml(name)
    return _to_matrix(frame), y


def load_raw(name: str) -> tuple[np.ndarray, list[int], np.ndarray]:
    """Rows as an object array: categories as strings ("missing" when absent), numbers with NaN kept."""
    frame, y = _openml(name)
    columns, categorical = [], []
    for i, name_ in enumerate(frame.columns):
        col = frame[name_]
        if isinstance(col.dtype, pd.CategoricalDtype) or col.dtype == object:
            categorical.append(i)
            columns.append(col.astype(object).where(col.notna(), "missing").astype(str).to_numpy(dtype=object))
        else:
            columns.append(col.to_numpy(dtype=np.float64).astype(object))
    return np.column_stack(columns), categorical, y


def _openml(name: str) -> tuple[pd.DataFrame, np.ndarray]:
    try:
        bunch = fetch_openml(data_id=OPENML_IDS[name], as_frame=True, data_home=str(DATA_HOME), parser="auto")
    except TypeError:
        # sklearn's pandas ARFF parser chokes on integer-valued nominal columns
        bunch = fetch_openml(data_id=OPENML_IDS[name], as_frame=True, data_home=str(DATA_HOME), parser="liac-arff")
    target = bunch.target.astype(str)
    positive = sorted(target.unique())[-1]
    return bunch.data, (target == positive).to_numpy(dtype=np.int64)
