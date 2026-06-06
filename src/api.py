from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.embedder import Embedded
from src.reranker import Reranker
from src.db import get_connection, create_table, insert_documents_batch, search_similar, get_doc_count
from src.parser import parse_file
from contextlib import asynccontextmanager
import os, time, logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

embedded = None
reranker = None
conn = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedded, reranker, conn

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL environment variable is required")

    logger.info("Loading embedding model...")
    embedded = Embedded()

    logger.info("Loading reranker model...")
    reranker = Reranker()

    logger.info("Connecting to database...")
    conn = get_connection(db_url)
    create_table(conn)
    logger.info("Startup complete — API ready")

    yield  # app runs here

    # shutdown
    if conn:
        conn.close()
    logger.info("Shutdown complete")

app = FastAPI(title="Semantic Search Engine", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

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

        embeddings = embedded.embed(chunks)
        insert_documents_batch(conn, chunks, embeddings, source=file.filename)
        results.append({"source": file.filename, "ingested": len(chunks)})

    return {"files": results}

@app.post("/search")
def search(req: SearchRequest):
    t0 = time.time()
    query_embedding = embedded.embed([req.query])[0]
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