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

A circuit compiled with noise_bound also takes clear noise from the server, which
the server draws per request with laplace_noise.

A "score" circuit returns the prediction only and a "decision" circuit one bit;
Client.decrypt then gives (prediction, None) or (1.0 or 0.0, None).
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from concrete import fhe

from cipherforest.circuit import CompiledModel
from cipherforest.oblivious import FloatArray, IntArray
from cipherforest.quantize import FeatureQuantizer, MultiClassModel, QuantizedModel

SERVER_FILE = "server.zip"
CLIENT_FILE = "client.json"


@dataclass(frozen=True)
class ClientBundle:
    client_specs: bytes
    quantizer: FeatureQuantizer
    scale: float
    base_value: float
    active_features: IntArray
    server_noise: bool = False
    output: str = "shap"
    # multiclass: one scale and base value per class, and decrypt returns arrays
    scales: tuple[float, ...] = ()
    base_values: tuple[float, ...] = ()

    def to_json(self) -> str:
        return json.dumps(
            {
                "client_specs": base64.b64encode(self.client_specs).decode("ascii"),
                **self.quantizer.to_doc(),
                "scale": self.scale,
                "base_value": self.base_value,
                "active_features": self.active_features.tolist(),
                "server_noise": self.server_noise,
                "output": self.output,
                **({"scales": list(self.scales), "base_values": list(self.base_values)} if self.scales else {}),
            }
        )

    @classmethod
    def from_json(cls, text: str) -> ClientBundle:
        doc = json.loads(text)
        return cls(
            client_specs=base64.b64decode(doc["client_specs"]),
            quantizer=FeatureQuantizer.from_doc(doc),
            scale=float(doc["scale"]),
            base_value=float(doc["base_value"]),
            active_features=np.asarray(doc["active_features"], dtype=np.int64),
            server_noise=bool(doc.get("server_noise", False)),
            output=str(doc.get("output", "shap")),
            scales=tuple(float(v) for v in doc.get("scales", ())),
            base_values=tuple(float(v) for v in doc.get("base_values", ())),
        )


def export(compiled: CompiledModel, model: QuantizedModel | MultiClassModel, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    compiled.circuit.server.save(str(directory / SERVER_FILE))
    multi = isinstance(model, MultiClassModel)
    first = model.classes[0] if multi else model
    bundle = ClientBundle(
        client_specs=compiled.circuit.client.specs.serialize(),
        quantizer=model.quantizer,
        scale=first.scale,
        base_value=first.base_value,
        active_features=compiled.active_features,
        server_noise=compiled.noise_bound > 0,
        output=compiled.output,
        scales=tuple(m.scale for m in model.classes) if multi else (),
        base_values=tuple(m.base_value for m in model.classes) if multi else (),
    )
    (directory / CLIENT_FILE).write_text(bundle.to_json())


class Server:
    def __init__(self, path: Path) -> None:
        self._server = fhe.Server.load(str(path))

    @staticmethod
    def load_keys(evaluation_keys: bytes) -> fhe.EvaluationKeys:
        # hundreds of MB: deserialise once per client, not per request
        return fhe.EvaluationKeys.deserialize(evaluation_keys)

    def run(self, ciphertext: bytes, evaluation_keys: fhe.EvaluationKeys, noise: IntArray | None = None) -> bytes:
        args: list[object] = [fhe.Value.deserialize(ciphertext)]
        if noise is not None:
            args.append(np.asarray(noise, dtype=np.int64))  # clear, no Value needed
        return self._server.run(*args, evaluation_keys=evaluation_keys).serialize()


class KeyStore:
    """Clients' evaluation keys on disk, only the most recently used few in memory.

    A loaded keyset holds about 270 MB, and after its first request about 2 GB (CatBoost
    depth 4 x 30), so a server cannot keep many clients loaded. add() writes the uploaded
    bytes to directory/<id>.keys and returns the id; get() loads them when needed and
    keeps the max_loaded most recent. A client loaded again pays the load (about 2 s)
    and a slower first request (about 5 s more), as on its first visit. Keys on disk
    survive a restart; so do a BudgetedServer's request counts if it has a state file.
    """

    def __init__(self, server: Server, directory: Path, max_loaded: int = 2) -> None:
        self.server = server
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_loaded = max_loaded
        self._loaded: OrderedDict[str, fhe.EvaluationKeys] = OrderedDict()
        self._lock = threading.Lock()

    def _path(self, key_id: str) -> Path:
        if not key_id.isalnum():
            raise KeyError(key_id)
        return self.directory / f"{key_id}.keys"

    def add(self, evaluation_keys: bytes) -> str:
        key_id = hashlib.sha256(evaluation_keys).hexdigest()[:32]
        path = self._path(key_id)
        if not path.exists():
            tmp = path.with_suffix(".part")
            tmp.write_bytes(evaluation_keys)
            tmp.replace(path)
        return key_id

    def __contains__(self, key_id: object) -> bool:
        return isinstance(key_id, str) and key_id.isalnum() and self._path(key_id).exists()

    @property
    def loaded(self) -> list[str]:
        return list(self._loaded)

    def get(self, key_id: str) -> fhe.EvaluationKeys:
        with self._lock:
            if key_id in self._loaded:
                self._loaded.move_to_end(key_id)
                return self._loaded[key_id]
            keys = self.server.load_keys(self._path(key_id).read_bytes())
            self._loaded[key_id] = keys
            while len(self._loaded) > self.max_loaded:
                self._loaded.popitem(last=False)
            return keys


class BudgetExceeded(RuntimeError):
    """A client's evaluation keys have used up their requests."""


class BudgetedServer(Server):
    """A Server that answers each set of evaluation keys at most `budget` times.

    The server cannot see inputs or answers, but it sees every request and the keys
    it comes with, and a new identity means a new key upload of hundreds of MB.
    Copying the model takes thousands of requests (RESULTS.md, Label-only answers),
    so a budget is the defence that remains. Keys are told apart by a SHA-256 of
    their bytes; `used` maps that fingerprint to the requests answered so far.

    With `state`, the counts are kept in that JSON file and read back on start, so a
    restart does not hand every client a fresh budget. A request is counted before it
    runs, under a lock, so concurrent requests cannot overrun the budget.
    """

    def __init__(self, path: Path, budget: int, state: Path | None = None) -> None:
        super().__init__(path)
        self.budget = budget
        self.state = None if state is None else Path(state)
        self.used: dict[str, int] = {}
        if self.state is not None and self.state.exists():
            self.used = {str(k): int(v) for k, v in json.loads(self.state.read_text()).items()}
        self._fingerprint: dict[int, str] = {}
        self._lock = threading.Lock()

    def load_keys(self, evaluation_keys: bytes) -> fhe.EvaluationKeys:  # type: ignore[override]
        keys = Server.load_keys(evaluation_keys)
        fingerprint = hashlib.sha256(evaluation_keys).hexdigest()
        with self._lock:
            self._fingerprint[id(keys)] = fingerprint
            self.used.setdefault(fingerprint, 0)
        return keys

    def run(self, ciphertext: bytes, evaluation_keys: fhe.EvaluationKeys, noise: IntArray | None = None) -> bytes:
        with self._lock:
            fingerprint = self._fingerprint.get(id(evaluation_keys))
            if fingerprint is None:
                raise KeyError("load these keys with this server's load_keys first")
            if self.used[fingerprint] >= self.budget:
                raise BudgetExceeded(f"these keys have used their {self.budget} requests")
            self.used[fingerprint] += 1
            if self.state is not None:
                tmp = self.state.with_suffix(".part")
                tmp.write_text(json.dumps(self.used))
                tmp.replace(self.state)
        return super().run(ciphertext, evaluation_keys, noise)


def laplace_noise(b: float, bound: int, size: int, rng: np.random.Generator) -> IntArray:
    """Rounded Laplace(b), clipped to the bound the circuit was compiled for."""
    return np.clip(np.rint(rng.laplace(0.0, b, size=size)), -bound, bound).astype(np.int64)


class Client:
    """keys: bytes from secret_keys() of an earlier client, to keep the same keys for a new
    model compiled with compile_model(..., same_keys_as=...). Otherwise new keys."""

    def __init__(self, bundle: ClientBundle, secret_seed: int | None = None, keys: bytes | None = None) -> None:
        # secret_seed makes keys reproducible; tests only
        self._bundle = bundle
        self._client = fhe.Client(fhe.ClientSpecs.deserialize(bundle.client_specs))
        if keys is None:
            self._client.keys.generate(secret_seed=secret_seed)
        else:
            self._client.keys.load_from_bytes(keys)

    @classmethod
    def load(cls, path: Path, secret_seed: int | None = None, keys: bytes | None = None) -> Client:
        return cls(ClientBundle.from_json(path.read_text()), secret_seed=secret_seed, keys=keys)

    def secret_keys(self) -> bytes:
        """The whole keyset, secret key included: store it on the client, never send it."""
        return self._client.keys.serialize()

    def evaluation_keys(self) -> bytes:
        return self._client.evaluation_keys.serialize()

    def encrypt(self, x: FloatArray) -> bytes:
        codes = self._bundle.quantizer.transform(np.array(x, dtype=object, ndmin=2))[0]
        if self._bundle.server_noise:
            return self._client.encrypt(codes, None)[0].serialize()
        return self._client.encrypt(codes).serialize()

    def decrypt(self, result: bytes) -> tuple[float | FloatArray, FloatArray | None]:
        """(prediction, SHAP vector); for a multiclass model (scores per class, (classes, features))."""
        outputs = np.asarray(self._client.decrypt(fhe.Value.deserialize(result)), dtype=np.int64)
        bundle = self._bundle
        if bundle.scales:
            scales, bases = np.array(bundle.scales), np.array(bundle.base_values)
            if bundle.output == "score":
                return bases + outputs.reshape(-1) / scales, None
            shap = np.zeros((len(scales), bundle.quantizer.n_features))
            shap[:, bundle.active_features] = outputs.reshape(len(scales), -1) / scales[:, None]
            return bases + shap.sum(axis=1), shap
        outputs = np.atleast_1d(outputs)
        if self._bundle.output == "decision":
            return float(outputs[0]), None
        if self._bundle.output == "score":
            return self._bundle.base_value + float(outputs[0]) / self._bundle.scale, None
        shap = np.zeros(self._bundle.quantizer.n_features)
        shap[self._bundle.active_features] = outputs / self._bundle.scale
        return self._bundle.base_value + float(shap.sum()), shap
