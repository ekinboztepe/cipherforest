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


def test_csv_with_empty_fields_reads_them_as_missing(tmp_path: Path) -> None:
    from cipherforest.cli import load_rows

    path = tmp_path / "rows.csv"
    path.write_text("a,b,c\n1,,3\nnan,2,NA\n")
    rows = load_rows(path)
    np.testing.assert_array_equal(np.isnan(rows), [[False, True, False], [True, False, True]])
    assert rows[0, 0] == 1 and rows[1, 1] == 2


def test_prepare_and_explain_a_model_with_categories(tmp_path: Path, capsys) -> None:
    rng = np.random.default_rng(3)
    city = rng.choice(["ankara", "izmir", "van"], 300)
    income = rng.normal(size=300)
    y = ((city == "van") ^ (income > 0)).astype(int)
    rows = [[c, float(v)] for c, v in zip(city, income)]
    cb = catboost.CatBoostClassifier(iterations=4, depth=2, verbose=0, cat_features=[0], one_hot_max_size=1,
                                     max_ctr_complexity=1, allow_writing_files=False).fit(rows, y)
    cb.save_model(str(tmp_path / "m.cbm"))
    train = tmp_path / "train.csv"
    train.write_text("city,income\n" + "".join(f"{c},{v}\n" for c, v in rows))
    main(["prepare", str(tmp_path / "m.cbm"), "--train", str(train), "--out", str(tmp_path / "model.json")])
    out = capsys.readouterr().out
    difference = float(out.split("training rows: ")[1].split(" ")[0])
    bound = float(out.split("(bound ")[1].split(")")[0])
    assert difference <= bound
    (tmp_path / "rows.csv").write_text("van,0.5\nistanbul,\n")
    main(["explain", str(tmp_path / "model.json"), "--data", str(tmp_path / "rows.csv")])
    assert len(capsys.readouterr().out.splitlines()) == 3


def test_explain_a_multiclass_model(tmp_path: Path, capsys) -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(300, 3))
    y = (x[:, 0] > 0).astype(int) + (x[:, 1] > 0).astype(int)
    cb = catboost.CatBoostClassifier(iterations=3, depth=2, verbose=0, loss_function="MultiClass",
                                     allow_writing_files=False).fit(x, y)
    cb.save_model(str(tmp_path / "m.cbm"))
    np.save(tmp_path / "train.npy", x)
    main(["prepare", str(tmp_path / "m.cbm"), "--train", str(tmp_path / "train.npy"), "--out", str(tmp_path / "m.json")])
    assert "3 classes" in capsys.readouterr().out
    np.save(tmp_path / "rows.npy", x[:2])
    main(["explain", str(tmp_path / "m.json"), "--data", str(tmp_path / "rows.npy")])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("score_0,score_1,score_2,shap_0_0") and len(lines) == 3
