# Project #7 — Semantic Search Engine with Reranking
> Full context document. Paste this at the start of a new chat and say: "Help me build this project."

---

## Who I Am

- Name: Tanish Sarkar
- Final-year Computer Engineering student (Marwadi University, Rajkot)
- Background: Strong in ML, DL, NLP, Transformers. Implemented "Attention is All You Need" from scratch. Built GPT-2 style model via HuggingFace manually (no AutoModel). Deployed ML models with FastAPI on Render.
- Comfortable with: Python, PyTorch, FastAPI, PostgreSQL, Docker basics
- Not yet comfortable with: pgvector, cross-encoders, advanced Docker Compose

---

## What I'm Building

**Project Name:** Semantic Search Engine with pgvector + Cross-Encoder Reranking

**One-line description:** A semantic search engine that uses sentence embeddings stored in PostgreSQL (pgvector) for fast ANN retrieval, then applies a cross-encoder reranker to improve result quality — built from raw components, no LangChain or any vector DB abstraction.

**Why this project:**
- Teaches the actual mechanics behind every RAG system
- Uses real infrastructure (PostgreSQL + pgvector), not toy in-memory stores
- Shows understanding of retrieval quality — rare for freshers
- Directly maps to AI/GenAI Engineer job descriptions

---

## Architecture

```
[Streamlit UI]
    │
    ├── Upload tab: user uploads .txt / .pdf / .csv
    │       ↓
    │   [FastAPI POST /ingest]
    │       → parse file → split into chunks → embed → insert into pgvector
    │
    └── Search tab: user types query
            ↓
        [FastAPI POST /search]
            ↓
        [Bi-Encoder: sentence-transformers]
            → Converts query to 384-dim embedding vector
            ↓
        [pgvector ANN Search in PostgreSQL]
            → Returns top 20 candidate chunks by cosine similarity
            ↓
        [Cross-Encoder Reranker]
            → Scores each (query, candidate) pair individually
            → Returns top 5 by relevance score
            ↓
        Results displayed in UI
```

**Two-stage retrieval logic (important to understand):**
- **Bi-encoder** — fast, embeds query and documents independently, used for bulk retrieval (top 20)
- **Cross-encoder** — slow but accurate, looks at query+document together, used to rerank the top 20 down to top 5
- You need BOTH because cross-encoder alone on a large corpus is too slow

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Embeddings | `sentence-transformers` — model: `all-MiniLM-L6-v2` |
| Vector DB | PostgreSQL + `pgvector` extension |
| ANN Index | IVFFlat index in pgvector |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Backend API | FastAPI |
| Frontend | Streamlit (with file upload UI) |
| File parsing | `pypdf2` (PDF), built-in (TXT), `pandas` (CSV) |
| Infrastructure | Docker Compose (all containers together) |
| Deployment — API | Railway (free tier, supports Docker + PostgreSQL add-on) |
| Deployment — Frontend | Streamlit Community Cloud (free, connects to GitHub) |
| Language | Python 3.10+ |

---

## Project Directory Structure

```
semantic-search-engine/
├── src/
│   ├── __init__.py
│   ├── embedder.py        # Bi-encoder: generates embeddings
│   ├── db.py              # pgvector: connect, create table, insert, search
│   ├── reranker.py        # Cross-encoder: reranks top-k results
│   ├── parser.py          # File parsing: txt, pdf, csv → list of chunks
│   └── api.py             # FastAPI app (search + ingest endpoints)
├── frontend/
│   └── app.py             # Streamlit UI (upload tab + search tab)
├── tests/
│   └── test_search.py
├── .github/
│   └── workflows/
│       └── ci.yml         # GitHub Actions: lint + health check
├── docker-compose.yml     # Runs db + api + frontend together
├── Dockerfile             # For the FastAPI app
├── requirements.txt
├── .env.example
└── README.md
```

Note: No `data/` folder and no `scripts/ingest_sample.py` — ingestion is now done through the UI.

---

## Docker Compose (Full Stack — DB + API + Frontend)

```yaml
# docker-compose.yml
version: '3.8'
services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: tanish
      POSTGRES_PASSWORD: password
      POSTGRES_DB: searchdb
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U tanish -d searchdb"]
      interval: 5s
      timeout: 5s
      retries: 5

  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://tanish:password@db:5432/searchdb
    depends_on:
      db:
        condition: service_healthy

  frontend:
    image: python:3.11-slim
    working_dir: /app
    volumes:
      - ./frontend:/app
      - ./requirements.txt:/app/requirements.txt
    command: >
      bash -c "pip install streamlit requests --quiet &&
               streamlit run app.py --server.port 8501 --server.address 0.0.0.0"
    ports:
      - "8501:8501"
    depends_on:
      - api

volumes:
  pgdata:
```

Run everything: `docker compose up --build`

---

## Dockerfile (for FastAPI app)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Dependencies

```txt
# requirements.txt
sentence-transformers
psycopg2-binary
pgvector
fastapi
uvicorn
python-dotenv
streamlit
requests
pypdf2
pandas
python-multipart
```

---

## Environment Variables

```bash
# .env.example
DATABASE_URL=postgresql://tanish:password@localhost:5432/searchdb
API_URL=http://localhost:8000
```

---

## Core Code

### `src/embedder.py`
```python
from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)
```

### `src/db.py`
```python
import psycopg2
from pgvector.psycopg2 import register_vector
import numpy as np

def get_connection(dsn: str):
    conn = psycopg2.connect(dsn)
    register_vector(conn)
    return conn

def create_table(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                source TEXT,
                embedding vector(384)
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS documents_embedding_idx
            ON documents USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        """)
    conn.commit()

def insert_documents_batch(conn, contents: list[str], embeddings: np.ndarray, source: str = "upload"):
    with conn.cursor() as cur:
        for content, embedding in zip(contents, embeddings):
            cur.execute(
                "INSERT INTO documents (content, source, embedding) VALUES (%s, %s, %s)",
                (content, source, embedding)
            )
    conn.commit()

def search_similar(conn, query_embedding: np.ndarray, top_k: int = 20):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, content, 1 - (embedding <=> %s) AS score
            FROM documents
            ORDER BY embedding <=> %s
            LIMIT %s
        """, (query_embedding, query_embedding, top_k))
        return cur.fetchall()

def get_doc_count(conn) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM documents")
        return cur.fetchone()[0]
```

### `src/parser.py`
```python
import io
import pandas as pd

def parse_file(filename: str, file_bytes: bytes) -> list[str]:
    """Parse uploaded file into a list of text chunks."""
    ext = filename.lower().split(".")[-1]

    if ext == "txt":
        text = file_bytes.decode("utf-8", errors="ignore")
        # Split into chunks by double newline or every ~500 chars
        chunks = [c.strip() for c in text.split("\n\n") if len(c.strip()) > 40]
        return chunks

    elif ext == "pdf":
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        chunks = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]
                chunks.extend(paragraphs)
        return chunks

    elif ext == "csv":
        df = pd.read_csv(io.BytesIO(file_bytes))
        # Combine all text columns into one string per row
        chunks = df.apply(lambda row: " | ".join(str(v) for v in row.values if pd.notna(v)), axis=1).tolist()
        return [c for c in chunks if len(c) > 40]

    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: .txt, .pdf, .csv")
```

### `src/reranker.py`
```python
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[str], top_k: int = 5) -> list[tuple]:
        pairs = [[query, doc] for doc in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
```

### `src/api.py`
```python
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.embedder import Embedder
from src.reranker import Reranker
from src.db import get_connection, create_table, insert_documents_batch, search_similar, get_doc_count
from src.parser import parse_file
import os, time
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Semantic Search Engine")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

embedder = Embedder()
reranker = Reranker()
conn = get_connection(os.getenv("DATABASE_URL"))
create_table(conn)

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    results = []
    for file in files:
        file_bytes = await file.read()
        try:
            chunks = parse_file(file.filename, file_bytes)
        except ValueError as e:
            results.append({"source": file.filename, "error": str(e)})
            continue
        embeddings = embedder.embed(chunks)
        insert_documents_batch(conn, chunks, embeddings, source=file.filename)
        results.append({"source": file.filename, "ingested": len(chunks)})
    return {"files": results}

@app.post("/search")
def search(req: SearchRequest):
    t0 = time.time()
    query_embedding = embedder.embed([req.query])[0]
    raw_results = search_similar(conn, query_embedding, top_k=20)
    retrieval_ms = round((time.time() - t0) * 1000, 2)

    candidates = [row[1] for row in raw_results]

    t1 = time.time()
    reranked = reranker.rerank(req.query, candidates, top_k=req.top_k)
    rerank_ms = round((time.time() - t1) * 1000, 2)

    return {
        "query": req.query,
        "results": [{"content": doc, "score": float(score)} for doc, score in reranked],
        "meta": {
            "retrieval_ms": retrieval_ms,
            "rerank_ms": rerank_ms,
            "total_ms": retrieval_ms + rerank_ms
        }
    }

@app.get("/stats")
def stats():
    return {"total_documents": get_doc_count(conn)}

@app.get("/health")
def health():
    return {"status": "ok"}
```

### `frontend/app.py`
```python
import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Semantic Search", layout="centered")
st.title("🔍 Semantic Search Engine")
st.caption("pgvector + cross-encoder reranking — no LangChain")

tab1, tab2 = st.tabs(["🔎 Search", "📤 Upload Documents"])

# --- SEARCH TAB ---
with tab1:
    try:
        stats = requests.get(f"{API_URL}/stats", timeout=3).json()
        st.info(f"📚 {stats['total_documents']} documents in the index")
    except:
        st.warning("API not reachable. Is it running?")

    query = st.text_input("Enter your search query")
    top_k = st.slider("Number of results", 1, 10, 5)

    if query:
        with st.spinner("Searching..."):
            res = requests.post(f"{API_URL}/search", json={"query": query, "top_k": top_k})
            if res.status_code == 200:
                data = res.json()
                meta = data.get("meta", {})
                st.caption(
                    f"⏱ Retrieval: {meta.get('retrieval_ms')}ms | "
                    f"Reranking: {meta.get('rerank_ms')}ms | "
                    f"Total: {meta.get('total_ms')}ms"
                )
                for i, r in enumerate(data["results"]):
                    st.markdown(f"**{i+1}.** {r['content']}")
                    st.caption(f"Score: {r['score']:.4f}")
                    st.divider()
            else:
                st.error("Search failed.")

# --- UPLOAD TAB ---
with tab2:
    st.subheader("Upload a document to index")
    st.caption("Supported formats: .txt, .pdf, .csv")

    uploaded_files = st.file_uploader(
        "Choose files", type=["txt", "pdf", "csv"], accept_multiple_files=True
    )

    if uploaded_files:
        for f in uploaded_files:
            st.write(f"**{f.name}** — {round(f.size / 1024, 1)} KB")

        if st.button("📥 Ingest all into search index"):
            with st.spinner("Parsing and embedding..."):
                files_payload = [
                    ("files", (f.name, f.getvalue(), f.type))
                    for f in uploaded_files
                ]
                res = requests.post(f"{API_URL}/ingest", files=files_payload)
                if res.status_code == 200:
                    data = res.json()
                    for result in data["files"]:
                        if "error" in result:
                            st.error(f"❌ {result['source']}: {result['error']}")
                        else:
                            st.success(f"✅ {result['source']}: {result['ingested']} chunks ingested")
                else:
                    st.error("Ingestion failed.")
```

---

## How to Run Locally

```bash
# Clone the repo
git clone https://github.com/yourusername/semantic-search-engine
cd semantic-search-engine

# Copy env file
cp .env.example .env

# Start everything
docker compose up --build

# Open in browser:
# Frontend → http://localhost:8501
# API docs  → http://localhost:8000/docs
```

---

## Deployment Plan

### Overview
Two separate deployments:
- **FastAPI + PostgreSQL** → Railway (handles Docker + managed Postgres with pgvector)
- **Streamlit frontend** → Streamlit Community Cloud (connects to GitHub, free)

The Streamlit app talks to the Railway API via `API_URL` env variable.

---

### Deploy FastAPI + PostgreSQL on Railway

**Why Railway:** Free tier ($5 credit/month), supports Docker deployments, has a managed PostgreSQL add-on with pgvector available. Much simpler than AWS/GCP for a portfolio project.

**Steps:**

1. Push your code to GitHub (the Railway deployment pulls from there)

2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub repo

3. Railway will detect your `Dockerfile` and build the API container automatically

4. Add a PostgreSQL database:
   - In Railway dashboard → New Service → Database → PostgreSQL
   - Railway gives you a `DATABASE_URL` automatically
   - Go to your API service → Variables → add `DATABASE_URL` = `${{Postgres.DATABASE_URL}}` (Railway's reference syntax)

5. Enable pgvector on the Railway Postgres instance:
   - Railway's Postgres doesn't have pgvector by default on free tier
   - **Alternative:** Use **Supabase** for the database instead (free tier, pgvector enabled out of the box)
   - Supabase steps: create project → go to SQL editor → run `CREATE EXTENSION vector;` → copy the connection string → paste as `DATABASE_URL` in Railway

6. After deploy, Railway gives you a public URL like `https://semantic-search-engine-production.up.railway.app`

7. Test: `curl https://your-railway-url/health` → should return `{"status": "ok"}`

**Railway `railway.toml` (optional, for config):**
```toml
[build]
builder = "dockerfile"

[deploy]
startCommand = "uvicorn src.api:app --host 0.0.0.0 --port $PORT"
healthcheckPath = "/health"
healthcheckTimeout = 60
```

---

### Deploy Streamlit Frontend on Streamlit Community Cloud

**Steps:**

1. Make sure `frontend/app.py` is in your GitHub repo

2. Go to [share.streamlit.io](https://share.streamlit.io) → New app

3. Connect your GitHub repo → set Main file path to `frontend/app.py`

4. Under Advanced settings → Secrets, add:
   ```
   API_URL = "https://your-railway-url"
   ```

5. Click Deploy → Streamlit gives you a public URL like `https://tanish-semantic-search.streamlit.app`

**That's it.** Streamlit Community Cloud redeploys automatically on every push to `main`.

---

### Supabase Setup (for pgvector — recommended over Railway Postgres)

1. Go to [supabase.com](https://supabase.com) → New project
2. After creation → SQL Editor → run:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
3. Go to Settings → Database → copy the **Connection string (URI)**
4. Paste as `DATABASE_URL` in Railway environment variables

Supabase free tier gives you 500MB storage — more than enough for a portfolio project.

---

## GitHub Flow (Proper Branch + Commit Workflow)

This is how you should work on this project day to day. It mirrors what real engineering teams do.

### Branch Strategy

```
main          → always deployable, protected. Never commit directly here.
dev           → integration branch. Features merge here first, then to main.
feature/*     → individual features
fix/*         → bug fixes
chore/*       → setup, config, docs
```

### Setup (one time)

```bash
git init
git remote add origin https://github.com/yourusername/semantic-search-engine

# Create and push dev branch
git checkout -b dev
git push -u origin dev

# Protect main on GitHub:
# Repo Settings → Branches → Add rule → Branch name: main
# Check: "Require pull request before merging"
```

### Day-to-Day Workflow

**Step 1 — Create an Issue first**

Before writing any code, open a GitHub Issue describing what you're building:
```
Title: Add file upload endpoint to FastAPI
Body:
- Add POST /ingest endpoint that accepts UploadFile
- Parse txt, pdf, csv formats
- Embed and insert chunks into pgvector
- Return count of ingested chunks

Acceptance criteria:
- [ ] /ingest accepts .txt .pdf .csv
- [ ] Returns {"ingested": N, "source": filename}
- [ ] Error message for unsupported formats
```
Label it: `feature` / `bug` / `chore`

**Step 2 — Create a branch from dev**

```bash
git checkout dev
git pull origin dev
git checkout -b feature/file-upload-ingest
```

Branch naming convention:
- `feature/file-upload-ingest`
- `feature/streamlit-upload-tab`
- `fix/db-connection-timeout`
- `chore/add-dockerfile`
- `chore/setup-ci`

**Step 3 — Commit with meaningful messages**

Follow Conventional Commits format:
```
<type>(<scope>): <short description>

Types: feat, fix, docs, chore, refactor, test, ci
```

Examples:
```bash
git commit -m "feat(api): add POST /ingest endpoint for file uploads"
git commit -m "feat(parser): add pdf and csv parsing support"
git commit -m "feat(frontend): add upload tab with file picker and progress"
git commit -m "fix(db): handle connection timeout on startup"
git commit -m "chore(docker): add app container to docker-compose"
git commit -m "docs(readme): add architecture diagram and setup steps"
```

**Never commit:**
```bash
git commit -m "fix"
git commit -m "changes"
git commit -m "wip"
git commit -m "asdfgh"
```

**Step 4 — Push and open a Pull Request**

```bash
git push origin feature/file-upload-ingest
```

Go to GitHub → Pull Requests → New PR:
- Base: `dev` (not main)
- Compare: `feature/file-upload-ingest`
- Title: `feat: add file upload and ingestion pipeline`
- Description: `Closes #<issue number>` (this auto-closes the issue when merged)
- Add yourself as reviewer

**Step 5 — Merge PR into dev**
- Squash and merge (keeps dev history clean)
- Delete the feature branch after merge

**Step 6 — Merge dev → main for deployment**

```bash
# When a set of features is complete and tested on dev
git checkout main
git merge dev --no-ff -m "release: file upload + ingest pipeline"
git push origin main
```

Railway auto-deploys on push to main. Streamlit Cloud also auto-deploys.

---

### GitHub Issues to Create at Start of Project

Create these issues before you write a single line of code. This is also good to show on your GitHub — it signals professional process.

```
#1  chore: project setup, directory structure, docker-compose for db
#2  feat: embedder module (bi-encoder with sentence-transformers)
#3  feat: db module (pgvector connect, create table, insert, search)
#4  feat: parser module (txt, pdf, csv → chunks)
#5  feat: reranker module (cross-encoder)
#6  feat: FastAPI /search endpoint
#7  feat: FastAPI /ingest endpoint (file upload)
#8  feat: FastAPI /stats and /health endpoints
#9  feat: streamlit search tab
#10 feat: streamlit upload tab
#11 chore: dockerfile for api
#12 chore: full docker-compose (db + api + frontend)
#13 chore: ci workflow (github actions — lint + health check)
#14 chore: deploy api to Railway + Supabase
#15 chore: deploy frontend to Streamlit Community Cloud
#16 docs: readme with architecture diagram, setup, metrics
```

---

### GitHub Actions CI (`.github/workflows/ci.yml`)

```yaml
name: CI

on:
  push:
    branches: [main, dev]
  pull_request:
    branches: [main, dev]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install ruff
      - run: ruff check src/ frontend/

  health-check:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v3
      - name: Start services
        run: docker compose up -d --build
      - name: Wait for API
        run: sleep 20
      - name: Check health endpoint
        run: curl --fail http://localhost:8000/health
      - name: Teardown
        run: docker compose down
```

---

## What Still Needs to Be Built

- [ ] Batch insertion with progress bar in Streamlit
- [ ] Admin dashboard: total docs, avg retrieval latency, avg rerank latency
- [ ] Before/after reranking comparison view in the UI (toggle)
- [ ] `DELETE /document/{id}` endpoint + delete UI
- [ ] Namespaced search (filter by source filename)
- [ ] Proper error handling: DB reconnect on drop, model load failures
- [ ] README with Excalidraw architecture diagram embedded as PNG

---

## Definition of Done

- [ ] `docker compose up --build` starts db + api + frontend in one command
- [ ] User can upload a .txt / .pdf / .csv from the Streamlit UI
- [ ] `/search` returns reranked results with latency metrics
- [ ] FastAPI deployed live on Railway
- [ ] Streamlit deployed live on Streamlit Community Cloud
- [ ] All features built via Issues → feature branches → PRs (visible on GitHub)
- [ ] CI passing on main
- [ ] README has: architecture diagram, live links, setup instructions, sample queries with before/after reranking

---

## Blog Post to Write After This Project

**Title:** "I built semantic search from scratch without LangChain — here's what I learned"

**Key points to cover:**
- Why two-stage retrieval (bi-encoder + cross-encoder) beats single-stage
- What pgvector's IVFFlat index actually does (and what `lists = 100` means)
- The latency tradeoff: ANN search vs reranking (show actual ms numbers)
- A concrete before/after example showing reranking fixing a bad result
- How file upload → chunking → embedding → storage works end to end
- GitHub link + live demo link

---

