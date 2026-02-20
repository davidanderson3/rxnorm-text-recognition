#!/usr/bin/env python3
"""Local web UI for the RxNorm text recognition script."""

from __future__ import annotations

import argparse
import json
import mimetypes
import subprocess
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse


class AppConfig:
    def __init__(
        self,
        infer_script: Path,
        index_dir: Path,
        web_dir: Path,
        top_k: int,
        exact_boost: float,
        max_graph_depth: int,
        max_ngram: int,
        max_exact_candidates: int,
    ) -> None:
        self.infer_script = infer_script
        self.index_dir = index_dir
        self.web_dir = web_dir
        self.top_k = top_k
        self.exact_boost = exact_boost
        self.max_graph_depth = max_graph_depth
        self.max_ngram = max_ngram
        self.max_exact_candidates = max_exact_candidates
        self.artifact_aliases = {
            "rxnorm_index",
            "rxnorm_mvp",
            "rxnorm_mvp_smoke",
            index_dir.name,
        }


def run_infer(config: AppConfig, text: str) -> Dict[str, object]:
    cmd = [
        sys.executable,
        str(config.infer_script),
        "infer",
        "--index-dir",
        str(config.index_dir),
        "--text",
        text,
        "--top-k",
        str(config.top_k),
        "--exact-boost",
        str(config.exact_boost),
        "--max-graph-depth",
        str(config.max_graph_depth),
        "--max-ngram",
        str(config.max_ngram),
        "--max-exact-candidates",
        str(config.max_exact_candidates),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        stderr = proc.stderr.strip() or "Inference failed."
        raise RuntimeError(stderr)

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse inference output: {exc}") from exc


def build_handler(config: AppConfig):
    class Handler(BaseHTTPRequestHandler):
        server_version = "RxNormWeb/1.0"

        def _send_json(self, status: int, payload: Dict[str, object]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_file(self, path: Path) -> None:
            if not path.exists() or not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "File not found")
                return
            content = path.read_bytes()
            mime, _ = mimetypes.guess_type(str(path))
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", f"{mime or 'application/octet-stream'}")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def _load_json_body(self) -> Optional[Dict[str, object]]:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0:
                return None
            raw = self.rfile.read(content_length)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                return None
            if not isinstance(payload, dict):
                return None
            return payload

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.send_response(HTTPStatus.FOUND)
                self.send_header("Location", "/web/index.html")
                self.end_headers()
                return
            if parsed.path == "/health":
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        "index_dir": str(config.index_dir),
                        "infer_script": str(config.infer_script),
                    },
                )
                return
            if parsed.path == "/rxnorm_text_recognition.py":
                self._serve_file(config.infer_script)
                return
            if parsed.path.startswith("/artifacts/"):
                artifact_rel = parsed.path[len("/artifacts/") :]
                parts = artifact_rel.split("/", 1)
                if len(parts) == 2:
                    alias, rel_path = parts
                    if alias in config.artifact_aliases:
                        safe_path = (config.index_dir / rel_path).resolve()
                        if not str(safe_path).startswith(str(config.index_dir.resolve())):
                            self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
                            return
                        self._serve_file(safe_path)
                        return
            if parsed.path.startswith("/web/"):
                rel_path = parsed.path[len("/web/") :]
                safe_path = (config.web_dir / rel_path).resolve()
                if not str(safe_path).startswith(str(config.web_dir.resolve())):
                    self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
                    return
                self._serve_file(safe_path)
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/api/infer":
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return

            payload = self._load_json_body()
            if payload is None:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "Expected JSON payload with a text field."},
                )
                return

            text = payload.get("text")
            if not isinstance(text, str) or not text.strip():
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "Field `text` must be a non-empty string."},
                )
                return

            try:
                result = run_infer(config, text=text)
            except RuntimeError as exc:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": str(exc)},
                )
                return

            self._send_json(HTTPStatus.OK, result)

        def log_message(self, fmt: str, *args: object) -> None:
            sys.stderr.write(f"[rxnorm-web] {fmt % args}\n")

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local web UI for RxNorm text recognition.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind.")
    parser.add_argument(
        "--index-dir",
        default="artifacts/rxnorm_index",
        help="Index dir generated by rxnorm_text_recognition.py build-index.",
    )
    parser.add_argument(
        "--infer-script",
        default="rxnorm_text_recognition.py",
        help="Path to inference CLI script.",
    )
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--exact-boost", type=float, default=0.35)
    parser.add_argument("--max-graph-depth", type=int, default=3)
    parser.add_argument("--max-ngram", type=int, default=8)
    parser.add_argument("--max-exact-candidates", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    infer_script = Path(args.infer_script).expanduser().resolve()
    index_dir = Path(args.index_dir).expanduser().resolve()
    web_dir = Path("web").resolve()

    if not infer_script.exists():
        raise FileNotFoundError(f"Cannot find infer script: {infer_script}")
    if not (index_dir / "rxnorm_index.sqlite").exists():
        raise FileNotFoundError(
            "Index not found. Expected: " f"{index_dir / 'rxnorm_index.sqlite'}"
        )
    if not (web_dir / "index.html").exists():
        raise FileNotFoundError(f"Web UI file missing: {web_dir / 'index.html'}")

    config = AppConfig(
        infer_script=infer_script,
        index_dir=index_dir,
        web_dir=web_dir,
        top_k=args.top_k,
        exact_boost=args.exact_boost,
        max_graph_depth=args.max_graph_depth,
        max_ngram=args.max_ngram,
        max_exact_candidates=args.max_exact_candidates,
    )

    handler = build_handler(config)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(
        f"RxNorm web UI running at http://{args.host}:{args.port} "
        f"(index: {index_dir})",
        file=sys.stderr,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
