"""The command line, end to end on small files."""

from pathlib import Path

import numpy as np
import pytest

import cipherforest
from cipherforest.cli import main

catboost = pytest.importorskip("catboost")


@pytest.fixture
def files(tmp_path: Path) -> tuple[Path, Path, np.ndarray, object]:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 4)).astype(np.float32).astype(np.float64)
    y = (x[:, 0] + x[:, 1] * x[:, 2] > 0).astype(int)
    cb = catboost.CatBoostClassifier(iterations=4, depth=3, verbose=0, random_seed=0, allow_writing_files=False).fit(x, y)
    model = tmp_path / "model.cbm"
    cb.save_model(str(model))
    train = tmp_path / "train.csv"
    np.savetxt(train, x, delimiter=",", header="a,b,c,d", comments="")
    return model, train, x, cb


def test_prepare_matches_the_library(files, tmp_path: Path, capsys) -> None:
    model, train, x, cb = files
    out = tmp_path / "model.json"
    main(["prepare", str(model), "--train", str(train), "--out", str(out), "--background", "50"])
    assert "4 trees of depth 3" in capsys.readouterr().out

    background = x[np.random.default_rng(0).choice(len(x), size=50, replace=False)]
    want = cipherforest.from_catboost(cb, background=background, x_train=x)
    assert cipherforest.QuantizedModel.from_json(out.read_text()).to_json() == want.to_json()


def test_explain_prints_score_and_shap(files, tmp_path: Path, capsys) -> None:
    model, train, x, _ = files
    out = tmp_path / "model.json"
    main(["prepare", str(model), "--train", str(train), "--out", str(out)])
    data = tmp_path / "rows.npy"
    np.save(data, x[:5])
    capsys.readouterr()
    main(["explain", str(out), "--data", str(data)])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "score,shap_0,shap_1,shap_2,shap_3"
    rows = np.array([[float(v) for v in line.split(",")] for line in lines[1:]])
    score, shap = cipherforest.explain(cipherforest.QuantizedModel.from_json(out.read_text()), x[:5])
    np.testing.assert_allclose(rows[:, 0], score, rtol=1e-5)
    np.testing.assert_allclose(rows[:, 1:], shap, rtol=1e-5, atol=1e-9)


@pytest.mark.fhe
def test_compile_writes_files_a_client_can_use(files, tmp_path: Path) -> None:
    pytest.importorskip("concrete")
    model, train, x, _ = files
    out = tmp_path / "model.json"
    main(["prepare", str(model), "--train", str(train), "--out", str(out)])
    main(["compile", str(out), "--out", str(tmp_path / "artifacts"), "--output", "score"])
    server = cipherforest.Server(tmp_path / "artifacts" / "server.zip")
    client = cipherforest.Client.load(tmp_path / "artifacts" / "client.json")
    keys = server.load_keys(client.evaluation_keys())
    score, _ = client.decrypt(server.run(client.encrypt(x[0]), keys))
    want, _ = cipherforest.explain(cipherforest.QuantizedModel.from_json(out.read_text()), x[:1])
    assert abs(score - want[0]) < 1e-9
