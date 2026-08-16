import psycopg
from psycopg.rows import dict_row
from .embeddings.embedder import Embedder
from .ingest_lyrics import vec_to_str
from .config import TOP_K


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


def search_sections(
    query, embedder: Embedder, conn: psycopg.Connection, num_results: int = TOP_K
) -> list[dict]:
    query_vector = embedder.encode(query)
    query_str = vec_to_str(query_vector)

    check_vector_index(conn)
    with conn.cursor(row_factory=dict_row) as cur:
        results = cur.execute(
            """
            SELECT
                d.document_id,
                d.song_id,
                s.title,
                s.performer,
                s.wks_on_chart,
                s.peak_pos,
                d.section_id,
                d.section,
                d.num_lines,
                1 - (d.embedding <=> %s::vector) AS similarity
            FROM documents AS d
            JOIN songs AS s
                ON d.song_id = s.song_id
            ORDER BY d.embedding <=> %s::vector
            LIMIT %s
            """,
            (query_str, query_str, num_results),
        ).fetchall()

    return results


if __name__ == "__main__":
    from src.ingest_lyrics import setup_postgres, check_tables

    embedder = Embedder()
    conn = setup_postgres()
    check_tables(conn)

    query = "Feeling excited about a new relationship"
    top_k_sections = search_sections(query, embedder, conn)
    for item in top_k_sections:
        print(item)
