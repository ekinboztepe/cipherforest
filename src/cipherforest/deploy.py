"""Client/server split.

The model owner exports two files. server.zip is the compiled circuit (thresholds
and SHAP tables) and stays on the server. client.json holds the client specs, the
quantiser edges and what is needed to decode (scale, base value, active features).

    client  keygen, send evaluation keys once
    client  encode + encrypt a sample        -> ciphertext
    server  run the circuit                  -> encrypted SHAP vector
    client  decrypt, prediction = base + sum(SHAP)

The server never holds the secret key. The client never sees thresholds or tables,
only the quantiser edges (training-data quantiles per feature).
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from concrete import fhe

from cipherforest.circuit import CompiledModel
from cipherforest.oblivious import FloatArray, IntArray
from cipherforest.quantize import FeatureQuantizer, QuantizedModel

SERVER_FILE = "server.zip"
CLIENT_FILE = "client.json"


@dataclass(frozen=True)
class ClientBundle:
    client_specs: bytes
    quantizer: FeatureQuantizer
    scale: float
    base_value: float
    active_features: IntArray

    def to_json(self) -> str:
        return json.dumps(
            {
                "client_specs": base64.b64encode(self.client_specs).decode("ascii"),
                "edges": [e.tolist() for e in self.quantizer.edges],
                "scale": self.scale,
                "base_value": self.base_value,
                "active_features": self.active_features.tolist(),
            }
        )

    @classmethod
    def from_json(cls, text: str) -> ClientBundle:
        doc = json.loads(text)
        return cls(
            client_specs=base64.b64decode(doc["client_specs"]),
            quantizer=FeatureQuantizer(tuple(np.asarray(e, dtype=np.float64) for e in doc["edges"])),
            scale=float(doc["scale"]),
            base_value=float(doc["base_value"]),
            active_features=np.asarray(doc["active_features"], dtype=np.int64),
        )


def export(compiled: CompiledModel, model: QuantizedModel, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    compiled.circuit.server.save(str(directory / SERVER_FILE))
    bundle = ClientBundle(
        client_specs=compiled.circuit.client.specs.serialize(),
        quantizer=model.quantizer,
        scale=model.scale,
        base_value=model.base_value,
        active_features=compiled.active_features,
    )
    (directory / CLIENT_FILE).write_text(bundle.to_json())


class Server:
    def __init__(self, path: Path) -> None:
        self._server = fhe.Server.load(str(path))

    @staticmethod
    def load_keys(evaluation_keys: bytes) -> fhe.EvaluationKeys:
        # hundreds of MB: deserialise once per client, not per request
        return fhe.EvaluationKeys.deserialize(evaluation_keys)

    def run(self, ciphertext: bytes, evaluation_keys: fhe.EvaluationKeys) -> bytes:
        return self._server.run(fhe.Value.deserialize(ciphertext), evaluation_keys=evaluation_keys).serialize()


class Client:
    def __init__(self, bundle: ClientBundle, secret_seed: int | None = None) -> None:
        # secret_seed makes keys reproducible; tests only
        self._bundle = bundle
        self._client = fhe.Client(fhe.ClientSpecs.deserialize(bundle.client_specs))
        self._client.keys.generate(secret_seed=secret_seed)

    @classmethod
    def load(cls, path: Path, secret_seed: int | None = None) -> Client:
        return cls(ClientBundle.from_json(path.read_text()), secret_seed=secret_seed)

    def evaluation_keys(self) -> bytes:
        return self._client.evaluation_keys.serialize()

    def encrypt(self, x: FloatArray) -> bytes:
        codes = self._bundle.quantizer.transform(np.atleast_2d(x))[0]
        return self._client.encrypt(codes).serialize()

    def decrypt(self, result: bytes) -> tuple[float, FloatArray]:
        outputs = np.atleast_1d(np.asarray(self._client.decrypt(fhe.Value.deserialize(result)), dtype=np.int64))
        shap = np.zeros(self._bundle.quantizer.n_features)
        shap[self._bundle.active_features] = outputs / self._bundle.scale
        return self._bundle.base_value + float(shap.sum()), shap
