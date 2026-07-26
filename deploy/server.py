#!/usr/bin/env python3
"""Serve the dashboard, and accept pushed snapshots from the machine running
the experiments.

The training and the AutoLab node run on a laptop; Maritime hosts the public
face. Rather than give the container credentials or a GPU, the laptop POSTs its
data.json here every few seconds. The page a viewer loads is therefore live
without the container knowing anything about training.
"""

import json
import os
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB = Path(__file__).resolve().parent / "web"
TOKEN = os.environ.get("INGEST_TOKEN", "")
PORT = int(os.environ.get("PORT", "8080"))
last_push = {"at": None}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(WEB), **kw)

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            return self._json(200, {"ok": True, "last_push": last_push["at"]})
        return super().do_GET()

    def do_POST(self):
        if self.path != "/ingest":
            return self._json(404, {"ok": False, "error": "not found"})
        if not TOKEN or self.headers.get("X-Ingest-Token") != TOKEN:
            return self._json(403, {"ok": False, "error": "bad token"})
        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            json.loads(body)  # reject anything that isn't a valid snapshot
        except (ValueError, TypeError) as e:
            return self._json(400, {"ok": False, "error": str(e)})
        (WEB / "data.json").write_bytes(body)
        last_push["at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return self._json(200, {"ok": True, "at": last_push["at"]})

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        if not self.path.startswith("/health"):
            super().log_message(fmt, *args)


if __name__ == "__main__":
    print(f"growlab dashboard on :{PORT} (ingest {'on' if TOKEN else 'DISABLED'})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
