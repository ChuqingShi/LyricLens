import psycopg


def check_vector_index(conn: psycopg.Connection) -> None:
    exists = conn.execute("""
        SELECT EXISTS (
            SELECT 1
            FROM pg_indexes
            WHERE indexname = 'documents_embedding_hnsw_idx'
        );
    """).fetchone()[0]

    if not exists:
        raise RuntimeError("HNSW index is not initialized.")
