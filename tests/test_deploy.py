"""Client and server loaded only from exported files, talking in bytes. Needs concrete-python."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("concrete.fhe")

import cipherforest  # noqa: E402
from cipherforest.circuit import compile_model  # noqa: E402
from cipherforest.deploy import (  # noqa: E402
    CLIENT_FILE,
    SERVER_FILE,
    BudgetedServer,
    BudgetExceeded,
    Client,
    ClientBundle,
    Server,
    export,
    laplace_noise,
)
from cipherforest.distill import distill_ensemble  # noqa: E402
from cipherforest.quantize import FeatureQuantizer, quantize_model  # noqa: E402
from cipherforest.shap_table import build_shap_table  # noqa: E402

pytestmark = pytest.mark.fhe


@pytest.mark.parametrize("noise_bound", [0, 4])
def test_round_trip_through_exported_files(tmp_path: Path, noise_bound: int) -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(400, 6))
    y = (x[:, 0] > 0.1).astype(float) + np.tanh(x[:, 3] * x[:, 4])
    quantizer = FeatureQuantizer.fit(x, n_bins=16)
    ensemble = distill_ensemble(x, y, 4, 2, quantizer.edges)
    tables = [build_shap_table(tree, x[:30]) for tree in ensemble.trees]
    model = quantize_model(ensemble, tables, quantizer, output_bits=8)
    compiled = compile_model(model, noise_bound=noise_bound)
    export(compiled, model, tmp_path)

    server = Server(tmp_path / SERVER_FILE)
    client = Client.load(tmp_path / CLIENT_FILE, secret_seed=11)
    keys = server.load_keys(client.evaluation_keys())

    samples = x[rng.choice(400, size=4, replace=False)]
    want_pred, want_shap = model.decode(model.simulate(quantizer.transform(samples)))
    for sample, pred, shap in zip(samples, want_pred, want_shap, strict=True):
        noise = None
        if noise_bound:
            noise = laplace_noise(2.0, noise_bound, compiled.active_features.size, rng)
            shap = shap.copy()
            shap[compiled.active_features] += noise / model.scale
            pred = pred + noise.sum() / model.scale
        got_pred, got_shap = client.decrypt(server.run(client.encrypt(sample), keys, noise))
        np.testing.assert_allclose(got_shap, shap, rtol=0, atol=1e-12)
        assert abs(got_pred - pred) < 1e-12


def test_client_bundle_json_round_trip() -> None:
    bundle = ClientBundle(
        client_specs=b"\x00\x01specs",
        quantizer=FeatureQuantizer((np.array([0.5, 1.5]), np.array([], dtype=np.float64))),
        scale=12.5,
        base_value=-0.25,
        active_features=np.array([0], dtype=np.int64),
    )
    restored = ClientBundle.from_json(bundle.to_json())
    assert restored.client_specs == bundle.client_specs
    assert (restored.scale, restored.base_value) == (bundle.scale, bundle.base_value)
    np.testing.assert_array_equal(restored.active_features, bundle.active_features)
    for a, b in zip(restored.quantizer.edges, bundle.quantizer.edges, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("output", ["score", "decision"])
def test_round_trip_without_explanations(tmp_path: Path, output: str) -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(size=(400, 6))
    y = (x[:, 0] > 0.1).astype(float) + np.tanh(x[:, 3] * x[:, 4])
    quantizer = FeatureQuantizer.fit(x, n_bins=16)
    ensemble = distill_ensemble(x, y, 4, 2, quantizer.edges)
    model = quantize_model(ensemble, [build_shap_table(t, x[:30]) for t in ensemble.trees], quantizer, output_bits=8)
    threshold = float(np.median(model.decode(model.simulate(quantizer.transform(x)))[0]))
    cut = model.cut_for(threshold) if output == "decision" else None
    export(compile_model(model, output=output, cut=cut), model, tmp_path)
    server = Server(tmp_path / SERVER_FILE)
    client = Client.load(tmp_path / CLIENT_FILE, secret_seed=12)
    keys = server.load_keys(client.evaluation_keys())
    for sample in x[rng.choice(400, size=4, replace=False)]:
        pred, _ = model.decode(model.simulate(quantizer.transform(sample[None])))
        got, shap = client.decrypt(server.run(client.encrypt(sample), keys))
        assert shap is None
        want = float(pred[0] > threshold) if output == "decision" else pred[0]
        assert abs(got - want) < 1e-12


def test_budget_counts_requests_per_set_of_keys(tmp_path: Path) -> None:
    rng = np.random.default_rng(6)
    x = rng.normal(size=(300, 4))
    y = (x[:, 0] > 0).astype(float)
    quantizer = FeatureQuantizer.fit(x, n_bins=8)
    ensemble = distill_ensemble(x, y, 3, 1, quantizer.edges)
    model = quantize_model(ensemble, [build_shap_table(t, x[:20]) for t in ensemble.trees], quantizer, output_bits=8)
    export(compile_model(model, output="score"), model, tmp_path)
    server = BudgetedServer(tmp_path / SERVER_FILE, budget=2)
    alice, bob = (Client.load(tmp_path / CLIENT_FILE, secret_seed=s) for s in (1, 2))
    alice_keys, bob_keys = server.load_keys(alice.evaluation_keys()), server.load_keys(bob.evaluation_keys())
    for _ in range(2):
        server.run(alice.encrypt(x[0]), alice_keys)
    with pytest.raises(BudgetExceeded):
        server.run(alice.encrypt(x[0]), alice_keys)
    server.run(bob.encrypt(x[0]), bob_keys)  # someone else's keys have their own budget
    assert sorted(server.used.values()) == [1, 2]
    # the same keys uploaded again are the same identity
    again = server.load_keys(alice.evaluation_keys())
    with pytest.raises(BudgetExceeded):
        server.run(alice.encrypt(x[0]), again)


@pytest.mark.fhe
def test_key_store_keeps_few_loaded_and_survives_a_restart(tmp_path: Path) -> None:
    from cipherforest.deploy import KeyStore

    rng = np.random.default_rng(5)
    x = rng.normal(size=(300, 3))
    model = cipherforest.distill(x, (x[:, 0] > 0).astype(int), depth=2, n_trees=1, n_bins=4, background=30)
    export(compile_model(model), model, tmp_path / "art")
    server = Server(tmp_path / "art" / "server.zip")
    store = KeyStore(server, tmp_path / "keys", max_loaded=1)
    clients = [Client.load(tmp_path / "art" / "client.json") for _ in range(2)]
    ids = [store.add(c.evaluation_keys()) for c in clients]
    want, _ = cipherforest.explain(model, x[:1])
    for client, key_id in [*zip(clients, ids), *zip(clients, ids)]:  # each evicts the other
        got, _ = client.decrypt(server.run(client.encrypt(x[0]), store.get(key_id)))
        assert abs(got - want[0]) < 1e-9
        assert store.loaded == [key_id]
    again = KeyStore(server, tmp_path / "keys")
    assert ids[0] in again and "../etc" not in again
    got, _ = clients[0].decrypt(server.run(clients[0].encrypt(x[0]), again.get(ids[0])))
    assert abs(got - want[0]) < 1e-9


@pytest.mark.fhe
def test_budget_survives_a_restart(tmp_path: Path) -> None:
    rng = np.random.default_rng(6)
    x = rng.normal(size=(300, 3))
    model = cipherforest.distill(x, (x[:, 0] > 0).astype(int), depth=2, n_trees=1, n_bins=4, background=30)
    export(compile_model(model), model, tmp_path)
    client = Client.load(tmp_path / CLIENT_FILE)
    uploaded = client.evaluation_keys()
    first = BudgetedServer(tmp_path / SERVER_FILE, budget=2, state=tmp_path / "used.json")
    keys = first.load_keys(uploaded)
    first.run(client.encrypt(x[0]), keys)
    second = BudgetedServer(tmp_path / SERVER_FILE, budget=2, state=tmp_path / "used.json")
    keys = second.load_keys(uploaded)
    second.run(client.encrypt(x[0]), keys)
    with pytest.raises(BudgetExceeded):
        second.run(client.encrypt(x[0]), keys)


@pytest.mark.fhe
def test_public_splits_give_the_same_answers(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(300, 3))
    model = cipherforest.distill(x, (x[:, 0] + x[:, 1] > 0).astype(int), depth=3, n_trees=2, n_bins=8, background=30)
    compiled = compile_model(model, public_splits=True)
    assert compiled.circuit.programmable_bootstrap_count == sum(t.players.size for t in model.trees)
    export(compiled, model, tmp_path)
    server = Server(tmp_path / SERVER_FILE)
    client = Client.load(tmp_path / CLIENT_FILE)
    keys = server.load_keys(client.evaluation_keys())
    want_pred, want_shap = cipherforest.explain(model, x[:3])
    for row, p, s in zip(x[:3], want_pred, want_shap, strict=True):
        pred, shap = client.decrypt(server.run(client.encrypt(row), keys))
        assert abs(pred - p) < 1e-9 and np.allclose(shap, s)
