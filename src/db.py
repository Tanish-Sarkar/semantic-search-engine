import psycopg2
from pgvector.psycopg2 import register_vector
import numpy as np

def get_connection(dns: str):
    conn = psycopg2.connect(dns)
    register_vector(conn)
    return conn

def create_table(conn):
    with conn.cursor() as cur:
        cur.execute(" CREATE EXTENSION IF NOT EXSISTS vector")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS document(
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    source TEXT,
                    embedding vector(384)    
                )
        """)
        cur.execute("""
                    CREATE INDEX IF NOT EXSITS documents_embedding_idx
                    ON documents USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                    """)
    conn.commit()


def inser_document_batch(conn, contents: list[str], embedding: np.ndarray, source: str = "upload"):
    with conn.cursor() as cur:
        for content, embedding in zip(contents, embedding):
            cur.execute(
                "INSERT INTO documents (content, source, embedding) VALUES (%s, %s, %s)",
                (content, source, embedding)
            )


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