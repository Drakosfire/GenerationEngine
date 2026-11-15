"""Expose the CUDA OCR runner on loopback for an authenticated private tunnel."""

import argparse
import hmac
import io
import json
import os
import selectors
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

from PIL import Image

MAX_INPUT = 4_500_000
MAX_OUTPUT = 1_000_000


def run_ocr(checkpoint: Path, image: Path) -> bytes:
    command = [
        sys.executable, str(Path(__file__).with_name("run_local_ocr.py")),
        "--checkpoint", str(checkpoint), "--image", str(image),
    ]
    deadline = monotonic() + 120
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as worker:
        assert worker.stdout is not None
        try:
            output = bytearray()
            with selectors.DefaultSelector() as selector:
                selector.register(worker.stdout, selectors.EVENT_READ)
                while selector.select(max(0, deadline - monotonic())):
                    chunk = os.read(worker.stdout.fileno(), min(65536, MAX_OUTPUT + 1 - len(output)))
                    if not chunk:
                        break
                    output.extend(chunk)
                    if len(output) > MAX_OUTPUT:
                        raise ValueError("OCR result too large")
                else:
                    raise subprocess.TimeoutExpired(command, 120)
            if worker.wait(timeout=max(0, deadline - monotonic())):
                raise subprocess.CalledProcessError(worker.returncode, command)
            return bytes(output)
        finally:
            if worker.poll() is None:
                worker.kill()


def serve(checkpoint: Path, port: int, token: str) -> None:
    class Handler(BaseHTTPRequestHandler):
        timeout = 10

        def reply(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            authorization = self.headers.get("Authorization", "").encode()
            if not hmac.compare_digest(authorization, f"Bearer {token}".encode()):
                self.reply(401, {"error": "Unauthorized"})
                return
            if self.path != "/ocr":
                self.reply(404, {"error": "Unknown route"})
                return
            length = self.headers.get("Content-Length", "")
            if (
                self.headers.get("Transfer-Encoding")
                or len(self.headers.get_all("Content-Length", [])) != 1
                or not length.isascii()
                or not length.isdecimal()
                or len(length) > len(str(MAX_INPUT))
                or not 0 < int(length) <= MAX_INPUT
                or self.headers.get("Content-Type") != "image/png"
            ):
                self.reply(400, {"error": "Expected a bounded PNG request"})
                return
            raw = self.rfile.read(int(length))
            try:
                if len(raw) != int(length):
                    raise ValueError("Incomplete body")
                with Image.open(io.BytesIO(raw), formats=["PNG"]) as image:
                    if max(image.size) > 1024 or image.n_frames != 1:
                        raise ValueError("Unsupported dimensions or frame count")
                    image.load()
            except (OSError, ValueError, Image.DecompressionBombError):
                self.reply(422, {"error": "Invalid PNG"})
                return
            try:
                with TemporaryDirectory(prefix="generationengine-ocr-http-") as scratch:
                    path = Path(scratch) / "capture.png"
                    path.write_bytes(raw)
                    output = run_ocr(checkpoint, path)
                text = json.loads(output)["text"]
                if not isinstance(text, str):
                    raise ValueError("Invalid OCR result")
            except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
                self.reply(503, {"error": "OCR failed"})
                return
            self.reply(200, {"text": text})

    with HTTPServer(("127.0.0.1", port), Handler) as server:
        server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8798)
    args = parser.parse_args()
    token = os.environ.get("GENERATIONENGINE_OCR_TOKEN", "")
    if not args.checkpoint.is_dir() or len(token) < 32 or not token.isascii() or not token.isprintable():
        parser.error("Provide a local checkpoint directory and a private OCR token of at least 32 characters.")
    serve(args.checkpoint.resolve(), args.port, token)


if __name__ == "__main__":
    main()
