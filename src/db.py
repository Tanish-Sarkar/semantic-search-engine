import psycopg2
from pgvector.psycopg2 import register_vector
import numpy as np
import logging

logger = logging.getLogger(__name__)

_dsn = None
_conn = None

def get_connection(dsn: str):
    global _dsn, _conn
    _dsn = dsn
    _conn = _make_connection(dsn)
    return _conn

def _make_connection(dsn: str):
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    register_vector(conn)
    logger.info("Database connection established")
    return conn

def _get_conn():
    """Return live connection, reconnecting if dropped."""
    global _conn, _dsn
    try:
        # ping the connection
        _conn.cursor().execute("SELECT 1")
        return _conn
    except Exception:
        logger.warning("Connection lost, reconnecting...")
        _conn = _make_connection(_dsn)
        return _conn

def create_table(conn):
    with conn.cursor() as cur:
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
    conn = _get_conn()
    with conn.cursor() as cur:
        for content, embedding in zip(contents, embeddings):
            cur.execute(
                "INSERT INTO documents (content, source, embedding) VALUES (%s, %s, %s)",
                (content, source, embedding)
            )
    conn.commit()

def search_similar(conn, query_embedding: np.ndarray, top_k: int = 20):
    conn = _get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, content, 1 - (embedding <=> %s) AS score
            FROM documents
            ORDER BY embedding <=> %s
            LIMIT %s
        """, (query_embedding, query_embedding, top_k))
        return cur.fetchall()

def get_doc_count(conn) -> int:
    conn = _get_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM documents")
        return cur.fetchone()[0]