import json
import threading
import urllib.error
import urllib.request

import pytest

from minirag import RAGPipeline
from minirag.server import serve


@pytest.fixture()
def base_url():
    rag = RAGPipeline()
    rag.add_documents([
        ("guides/install.md", "# Install\n\nInstall minirag with pip install minirag."),
        ("notes/cats.md", "Cats sleep for most of the day."),
    ])
    httpd = serve(rag, port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def call(url, body=None):
    data = None if body is None else json.dumps(body).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=data)) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_healthz(base_url):
    assert call(base_url + "/healthz") == (200, {"status": "ok", "chunks": 2})


def test_search_and_filter(base_url):
    status, out = call(base_url + "/search", {"query": "how to install", "k": 2})
    assert status == 200 and out["hits"][0]["doc_id"] == "guides/install.md"
    assert out["hits"][0]["metadata"]["heading"] == "Install"
    _, out = call(base_url + "/search", {"query": "install", "doc_prefix": ["notes/"]})
    assert all(h["doc_id"].startswith("notes/") for h in out["hits"])


def test_ask_returns_prompt(base_url):
    status, out = call(base_url + "/ask", {"query": "do cats sleep?", "k": 1})
    assert status == 200 and "Cats sleep" in out["prompt"] and out["answer"] is None


@pytest.mark.parametrize("body", [{}, {"query": ""}, {"query": "x", "k": 0}, {"query": "x", "k": True},
                                  {"query": "x", "doc_prefix": "notes/"}, [1]])
def test_bad_requests(base_url, body):
    status, out = call(base_url + "/search", body)
    assert status == 400 and "error" in out


def test_unknown_route(base_url):
    assert call(base_url + "/nope", {"query": "x"})[0] == 404
