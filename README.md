# Semantic Search Engine with pgvector + Cross-Encoder Reranking

A two-stage semantic search engine built from raw components — no LangChain, no vector DB abstraction.

**Stack:** sentence-transformers · PostgreSQL + pgvector · FastAPI · Streamlit · Docker

---

## Architecture

```
[Streamlit UI]
    │
    ├── Upload tab: user uploads .txt / .pdf / .csv
    │       ↓
    │   [FastAPI POST /ingest]
    │       → parse → chunk → embed → insert into pgvector
    │
    └── Search tab: user types query
            ↓
        [FastAPI POST /search]
            ↓
        [Bi-Encoder: all-MiniLM-L6-v2]
            → 384-dim embedding
            ↓
        [pgvector ANN Search — top 20 candidates]
            ↓
        [Cross-Encoder: ms-marco-MiniLM-L-6-v2]
            → reranks top 20 → returns top 5
            ↓
        Results with latency metrics
```

**Why two stages?**
- Bi-encoder is fast but approximate — embeds query and docs independently
- Cross-encoder is slow but accurate — scores (query, doc) pairs together
- Use both: bi-encoder for bulk retrieval, cross-encoder to rerank the shortlist

---

## Setup

```bash
git clone https://github.com/yourusername/semantic-search-engine
cd semantic-search-engine

cp .env.example .env

docker compose up --build
```

| Service  | URL                         |
|----------|-----------------------------|
| Frontend | http://localhost:8501       |
| API docs | http://localhost:8000/docs  |
| API      | http://localhost:8000       |

---

## Project Structure

```
semantic-search-engine/
├── src/
│   ├── embedder.py     # Bi-encoder (sentence-transformers)
│   ├── db.py           # pgvector: connect, create table, insert, search
│   ├── reranker.py     # Cross-encoder reranker
│   ├── parser.py       # txt / pdf / csv → list of chunks
│   └── api.py          # FastAPI app
├── frontend/
│   └── app.py          # Streamlit UI
├── tests/
│   └── test_search.py
├── .github/workflows/
│   └── ci.yml
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## API Endpoints

| Method | Path      | Description                        |
|--------|-----------|------------------------------------|
| POST   | /ingest   | Upload and index .txt/.pdf/.csv    |
| POST   | /search   | Semantic search with reranking     |
| GET    | /stats    | Total documents in index           |
| GET    | /health   | Health check                       |

---

## Deployment

- **API + DB:** Railway (Docker + PostgreSQL with pgvector)
- **Frontend:** Streamlit Community Cloud (auto-deploys from GitHub)
