"""A retrained model served with the clients' old keys."""

import numpy as np
import pytest

import cipherforest


def data(seed: int, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4))
    return x, (x[:, 0] + x[:, 1] * x[:, 2] > 0).astype(int)


@pytest.mark.fhe
def test_retrained_model_answers_with_the_old_keys(tmp_path) -> None:
    pytest.importorskip("concrete")
    x, y = data(0)
    old = cipherforest.distill(x, y, depth=3, n_trees=2, n_bins=8, background=40)
    cipherforest.export(cipherforest.compile_model(old), old, tmp_path / "old")
    first = cipherforest.Client.load(tmp_path / "old" / "client.json")
    evaluation_keys, secret_keys = first.evaluation_keys(), first.secret_keys()

    x2, y2 = data(1)
    new = cipherforest.distill(x2, y2, depth=3, n_trees=2, n_bins=8, background=40, seed=1)
    compiled = cipherforest.compile_model(new, same_keys_as=tmp_path / "old" / "client.json")
    cipherforest.export(compiled, new, tmp_path / "new")
    server = cipherforest.Server(tmp_path / "new" / "server.zip")
    keys = server.load_keys(evaluation_keys)  # what the server already held
    client = cipherforest.Client.load(tmp_path / "new" / "client.json", keys=secret_keys)
    want_pred, want_shap = cipherforest.explain(new, x2[:3])
    for row, p, s in zip(x2[:3], want_pred, want_shap, strict=True):
        pred, shap = client.decrypt(server.run(client.encrypt(row), keys))
        assert abs(pred - p) < 1e-9 and np.allclose(shap, s)


@pytest.mark.fhe
def test_a_model_needing_other_keys_is_refused(tmp_path) -> None:
    pytest.importorskip("concrete")
    x, y = data(2)
    old = cipherforest.distill(x, y, depth=2, n_trees=1, n_bins=4, background=40)
    cipherforest.export(cipherforest.compile_model(old), old, tmp_path / "old")
    bigger = cipherforest.distill(x, y, depth=6, n_trees=3, n_bins=32, background=40, output_bits=12)
    with pytest.raises(ValueError, match="keys"):
        cipherforest.compile_model(bigger, same_keys_as=tmp_path / "old" / "client.json")
