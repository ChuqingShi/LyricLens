import psycopg
from psycopg.rows import dict_row
from .embeddings.embedder import Embedder
from .ingest_lyrics import vec_to_str
from .config import TOP_K_SECTIONS, TOP_K_SONGS


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
    query, embedder: Embedder, conn: psycopg.Connection, num_results: int = TOP_K_SECTIONS
) -> list[dict]:
    query_vector = embedder.encode(query)
    query_str = vec_to_str(query_vector)

    check_vector_index(conn)
    with conn.cursor(row_factory=dict_row) as cur:
        section_results = cur.execute(
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

    return section_results


def aggregate_song_results(
    section_results: list[dict], num_results: int = TOP_K_SONGS
) -> list[dict]:
    song_results = {}

    for r in section_results:
        song_id = r["song_id"]
        if song_id not in song_results:
            song_results[song_id] = {
                "song_id": song_id,
                "title": r["title"],
                "performer": r["performer"],
                "wks_on_chart": r["wks_on_chart"],
                "peak_pos": r["peak_pos"],
                "best_similarity": r["similarity"],  # input ordered by similarity
                "num_matches": 0,
                "matched_sections": [],
            }

        # song_results[song_id]["best_similarity"] = max(
        #     song_results[song_id]["best_similarity"],
        #     r["similarity"],
        # )
        song_results[song_id]["num_matches"] = song_results[song_id]["num_matches"] + 1
        song_results[song_id]["matched_sections"].append(
            {
                "section_id": r["section_id"],
                "section": r["section"],
                "num_lines": r["num_lines"],
                "similarity": r["similarity"],
            }
        )
    song_results = sorted(song_results.values(), key=lambda x: x["best_similarity"], reverse=True)

    if len(song_results) > num_results:
        return song_results[:num_results]
    else:
        return song_results


if __name__ == "__main__":
    from src.ingest_lyrics import setup_postgres, check_tables

    embedder = Embedder()
    conn = setup_postgres()
    check_tables(conn)

    query = "Feeling excited about a new relationship"
    top_k_sections = search_sections(query, embedder, conn)
    for item in top_k_sections:
        print(item["document_id"])

    top_k_songs = aggregate_song_results(top_k_sections)
    for i, item in enumerate(top_k_songs):
        print(f"------result {i}: song #{item["song_id"]}------")
        print(f"{item["title"]} by {item["performer"]}")
        print(f"popularity: wks_on_chart={item["wks_on_chart"]}, peak_pos={item["peak_pos"]}.")
        print(
            f"similarity: best similarity={item["best_similarity"]} with {item["num_matches"]} matches in top {TOP_K_SECTIONS}."
        )
