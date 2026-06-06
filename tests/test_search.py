# import pytest
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ── Health ──────────────────────────────────────────────────────────────────

def test_health():
    res = requests.get(f"{API_URL}/health", timeout=10)
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}

# ── Stats ───────────────────────────────────────────────────────────────────

def test_stats_returns_count():
    res = requests.get(f"{API_URL}/stats", timeout=10)
    assert res.status_code == 200
    data = res.json()
    assert "total_documents" in data
    assert isinstance(data["total_documents"], int)
    assert data["total_documents"] >= 0

# ── Ingest ──────────────────────────────────────────────────────────────────

def test_ingest_txt():
    content = b"""This is the first paragraph of a test document.
It has enough content to be indexed properly.

This is the second paragraph with more information.
It also contains enough text to pass the chunk filter.

This is the third paragraph for good measure.
Adding more content to ensure multiple chunks are created."""

    files = [("files", ("test.txt", content, "text/plain"))]
    res = requests.post(f"{API_URL}/ingest", files=files, timeout=30)
    assert res.status_code == 200
    data = res.json()
    assert "files" in data
    assert data["files"][0]["source"] == "test.txt"
    assert data["files"][0]["ingested"] > 0

def test_ingest_unsupported_format():
    files = [("files", ("test.xyz", b"some content", "application/octet-stream"))]
    res = requests.post(f"{API_URL}/ingest", files=files, timeout=30)
    assert res.status_code == 200
    data = res.json()
    assert "error" in data["files"][0]

# ── Search ──────────────────────────────────────────────────────────────────

def test_search_returns_results():
    res = requests.post(
        f"{API_URL}/search",
        json={"query": "test document paragraph", "top_k": 3},
        timeout=30
    )
    assert res.status_code == 200
    data = res.json()
    assert "query" in data
    assert "results" in data
    assert "meta" in data
    assert isinstance(data["results"], list)

def test_search_result_structure():
    res = requests.post(
        f"{API_URL}/search",
        json={"query": "test document", "top_k": 2},
        timeout=30
    )
    assert res.status_code == 200
    data = res.json()
    if data["results"]:
        result = data["results"][0]
        assert "content" in result
        assert "score" in result
        assert isinstance(result["score"], float)

def test_search_meta_has_latency():
    res = requests.post(
        f"{API_URL}/search",
        json={"query": "test", "top_k": 1},
        timeout=30
    )
    assert res.status_code == 200
    meta = res.json()["meta"]
    assert "retrieval_ms" in meta
    assert "rerank_ms" in meta
    assert "total_ms" in meta

def test_search_top_k_respected():
    res = requests.post(
        f"{API_URL}/search",
        json={"query": "paragraph content", "top_k": 2},
        timeout=30
    )
    assert res.status_code == 200
    results = res.json()["results"]
    assert len(results) <= 2