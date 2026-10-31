"""Client and server loaded only from exported files, talking in bytes. Needs concrete-python."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("concrete.fhe")

from cipherforest.circuit import compile_model  # noqa: E402
from cipherforest.deploy import CLIENT_FILE, SERVER_FILE, Client, ClientBundle, Server, export, laplace_noise  # noqa: E402
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
