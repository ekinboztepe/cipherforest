"""From a CatBoost model to an encrypted prediction with its explanation.

    pip install "cipherforest[catboost]"           # any OS: steps 1 and 2
    pip install "cipherforest[catboost,fhe]"       # Linux or macOS: all steps
    python examples/quickstart.py

On Windows run it in the Docker image (see README).
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from catboost import CatBoostClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

import cipherforest

# 1. Train as usual. Symmetric trees (CatBoost's default), small and shallow:
#    each tree costs depth + (features it uses) bootstraps.
x, y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
cb = CatBoostClassifier(iterations=10, depth=4, verbose=0, random_seed=0, allow_writing_files=False)
cb.fit(x_train, y_train)

# 2. Turn it into the integer model the circuit serves, and check it in the clear.
model = cipherforest.from_catboost(cb, background=x_train[:100], x_train=x_train)
score, shap = cipherforest.explain(model, x_test)
print("max |score - CatBoost raw score|:", np.abs(score - cb.predict(x_test, prediction_type="RawFormulaVal")).max())
print("first test row, top features:", np.argsort(-np.abs(shap[0]))[:3], "score", round(score[0], 3))

try:
    from concrete import fhe  # noqa: F401
except ImportError:
    raise SystemExit("concrete-python is not installed: stopping before the encrypted part")

# 3. Compile and export. server.zip goes to the server, client.json to clients.
with TemporaryDirectory() as tmp:
    out = Path(tmp)
    cipherforest.export(cipherforest.compile_model(model), model, out)

    # 4. Client and server, talking in bytes.
    server = cipherforest.Server(out / "server.zip")
    client = cipherforest.Client.load(out / "client.json")
    keys = server.load_keys(client.evaluation_keys())  # sent once per client
    pred, explanation = client.decrypt(server.run(client.encrypt(x_test[0]), keys))
    assert abs(pred - score[0]) < 1e-9 and np.allclose(explanation, shap[0])
    print("encrypted run agrees with the clear one:", round(pred, 3))

    # An approve/decline server instead: one bit back, score > 0 means class 1.
    decision = cipherforest.compile_model(model, output="decision", cut=model.cut_for(0.0))
    cipherforest.export(decision, model, out / "decision")
    server = cipherforest.Server(out / "decision" / "server.zip")
    client = cipherforest.Client.load(out / "decision" / "client.json")
    keys = server.load_keys(client.evaluation_keys())
    approved, _ = client.decrypt(server.run(client.encrypt(x_test[0]), keys))
    print("decision:", int(approved), "expected:", int(score[0] > 0))
