"""A small JSON HTTP API over a pipeline, using only the standard library.

``POST /search`` and ``POST /ask`` take ``{"query": str, "k": int?, "doc_prefix": [str]?}``.
``GET /healthz`` reports the number of indexed chunks.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from minirag.pipeline import RAGPipeline, doc_prefix
from minirag.retrievers import Hit

MAX_BODY = 1 << 20
MAX_K = 50


class BadRequest(ValueError):
    pass


def _hit(h: Hit) -> dict[str, Any]:
    c = h.chunk
    return {"id": c.id, "doc_id": c.doc_id, "score": h.score, "text": c.text, "metadata": c.metadata}


def _parse(body: bytes) -> tuple[str, int, list[str] | None]:
    try:
        data = json.loads(body or b"{}")
    except json.JSONDecodeError as e:
        raise BadRequest(f"invalid JSON: {e.msg}") from None
    if not isinstance(data, dict):
        raise BadRequest("body must be a JSON object")
    query = data.get("query")
    if not isinstance(query, str) or not query.strip():
        raise BadRequest('"query" must be a non-empty string')
    k = data.get("k", 4)
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= MAX_K:
        raise BadRequest(f'"k" must be an integer from 1 to {MAX_K}')
    prefixes = data.get("doc_prefix")
    if prefixes is not None and (
        not isinstance(prefixes, list) or not all(isinstance(p, str) for p in prefixes)
    ):
        raise BadRequest('"doc_prefix" must be a list of strings')
    return query, k, prefixes


def make_handler(rag: RAGPipeline) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "minirag"

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/healthz":
                self._send(200, {"status": "ok", "chunks": len(rag.chunks)})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path not in ("/search", "/ask"):
                self._send(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                self._send(413, {"error": "body too large"})
                return
            try:
                query, k, prefixes = _parse(self.rfile.read(length))
            except BadRequest as e:
                self._send(400, {"error": str(e)})
                return
            where = doc_prefix(*prefixes) if prefixes else None
            if self.path == "/search":
                self._send(200, {"hits": [_hit(h) for h in rag.search(query, k=k, where=where)]})
                return
            answer = rag.ask(query, k=k, where=where)
            self._send(200, {
                "question": answer.question,
                "prompt": answer.prompt,
                "answer": answer.text,
                "hits": [_hit(h) for h in answer.hits],
            })

        def log_message(self, format: str, *args: Any) -> None:  # quieter than the default
            pass

    return Handler


def serve(rag: RAGPipeline, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    """Create a server; call ``serve_forever()`` on it to start handling requests."""
    return ThreadingHTTPServer((host, port), make_handler(rag))
