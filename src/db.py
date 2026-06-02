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