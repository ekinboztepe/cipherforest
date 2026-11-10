"""A small HTTP server for encrypted requests, and a client for it. Standard library only.

    python examples/http_server.py serve artifacts/server.zip --budget 500
    python examples/http_server.py ask artifacts/client.json x.csv --url http://127.0.0.1:8000

The server holds no secret: it sees evaluation keys and ciphertexts, never an
input or an answer. A client uploads its evaluation keys once (POST /keys, hundreds
of MB) and gets an id, then sends one ciphertext per request (POST /run/<id>).
BudgetedServer answers each set of keys a fixed number of times, the defence
SECURITY.md recommends against copying the model by queries.

This is a sketch for a trusted network. A real deployment adds TLS, client
authentication (the budget is per set of keys, so it counts only if getting new
keys costs the client something), limits on upload size, and keys kept on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

import cipherforest


def serve(args: argparse.Namespace) -> None:
    server = cipherforest.BudgetedServer(args.server_zip, budget=args.budget)
    keys: dict[str, object] = {}

    class Handler(BaseHTTPRequestHandler):
        def body(self) -> bytes:
            return self.rfile.read(int(self.headers.get("Content-Length", 0)))

        def reply(self, status: int, data: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:  # noqa: N802
            if self.path == "/keys":
                data = self.body()
                key_id = hashlib.sha256(data).hexdigest()[:16]
                keys.setdefault(key_id, server.load_keys(data))
                self.reply(200, key_id.encode())
            elif self.path.startswith("/run/") and self.path[5:] in keys:
                try:
                    self.reply(200, server.run(self.body(), keys[self.path[5:]]))
                except cipherforest.BudgetExceeded:
                    self.reply(429, b"request budget used up for these keys")
            else:
                self.reply(404, b"POST /keys, then POST /run/<id>")

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"serving {args.server_zip} on http://{args.host}:{args.port}, {args.budget} requests per set of keys")
    httpd.serve_forever()


def post(url: str, data: bytes) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, data=data, method="POST")) as response:
        return response.read()


def ask(args: argparse.Namespace) -> None:
    client = cipherforest.Client.load(args.client_json)
    key_id = post(f"{args.url}/keys", client.evaluation_keys()).decode()
    rows = np.loadtxt(args.rows, delimiter=",", ndmin=2)
    for row in rows:
        score, shap = client.decrypt(post(f"{args.url}/run/{key_id}", client.encrypt(row)))
        top = "" if shap is None else "  top features " + ", ".join(map(str, np.argsort(-np.abs(shap))[:3]))
        print(f"score {score:.4f}{top}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(required=True)
    p = commands.add_parser("serve")
    p.add_argument("server_zip", type=Path)
    p.add_argument("--budget", type=int, default=500)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(run=serve)
    p = commands.add_parser("ask")
    p.add_argument("client_json", type=Path)
    p.add_argument("rows", type=Path, help="numeric .csv, one row per request, no header")
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.set_defaults(run=ask)
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
