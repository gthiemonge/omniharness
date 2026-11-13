"""A stand-in for llama-server + d1-2B: /health and /v1/systemone.

Answers are driven by markers in the prompt: for a score question, "[level:N]"
peaks the distribution on level N (default 0); for a choice question,
"[pick:LABEL]" picks that option (default: the first). "[fail]" returns HTTP 500.

Used in-process by the tests, and as a fake `llama-server` executable for the
autostart path:  python fake_d1.py --port 8080 -m model.gguf ...
As an executable, env vars mimic llama-server's output:
  FAKE_D1_NOISE_KB=N  write N KB of log (with invalid UTF-8) to stderr before serving
  FAKE_D1_CRASH=1     write a log with invalid UTF-8, then exit 1
  FAKE_D1_CRASH=arch  same, failing like a llama.cpp too old for d1-2B
"""

import argparse
import json
import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar


def answer_for(question: dict, state: str) -> dict:
    if question["type"] == "choice":
        labels = list(question["criteria"])
        m = re.search(r"\[pick:(\w+)\]", state)
        pick = m.group(1) if m and m.group(1) in labels else labels[0]
        probs = {label: (0.1 / (len(labels) - 1)) for label in labels}
        probs[pick] = 0.9
        return {"type": "choice", "choice": pick, "confidence": 0.9, "probabilities": probs}
    m = re.search(r"\[level:(\d)\]", state)
    level = int(m.group(1)) if m else 0
    probs = {str(i): 0.0 for i in range(5)}
    probs[str(level)] = 0.8
    probs[str(min(level + 1, 4) if level < 4 else 3)] = 0.2
    mean = sum(int(k) * v for k, v in probs.items())
    return {
        "type": "score", "score": float(level), "confidence": 0.8,
        "legend": {str(i): str(i) for i in range(5)}, "probabilities": probs, "mean": mean,
    }


class Handler(BaseHTTPRequestHandler):
    requests: ClassVar[list] = []

    def log_message(self, *args):
        pass

    def _send(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append(body)
        if "[fail]" in str(body.get("state")):
            self._send(500, {"error": {"message": "boom"}})
            return
        answers = {qid: answer_for(q, str(body["state"])) for qid, q in body["questions"].items()}
        self._send(200, {"answers": answers, "usage": {}})


def serve(port: int = 0) -> ThreadingHTTPServer:
    """Start in a background thread; the URL is http://127.0.0.1:{server.server_port}."""
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    args, _ = parser.parse_known_args()
    noise = b"load_tensors: \xc4\x80 ok \xc4 bad byte\n"
    if os.getenv("FAKE_D1_CRASH"):
        if os.getenv("FAKE_D1_CRASH") == "arch":
            sys.stderr.buffer.write(b"llama_model_load: error loading model: error loading "
                                    b"model architecture: unknown model architecture: 'lfm3'\n")
        sys.stderr.buffer.write(noise * 50 + b"error: failed to load model\n")
        sys.exit(1)
    kb = int(os.getenv("FAKE_D1_NOISE_KB", "0"))
    if kb:
        sys.stderr.buffer.write(noise * (kb * 1024 // len(noise)))
        sys.stderr.flush()
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
