import psycopg
from psycopg.rows import dict_row
from .embeddings.embedder import Embedder
from .ingest_postgres import vec_to_str
from .config import TOP_K_SECTIONS, TOP_K_SONGS


def check_vector_index(conn: psycopg.Connection) -> None:
    """Check whether the HNSW vector index exists."""

    exists = conn.execute("""
        SELECT EXISTS (
            SELECT 1
            FROM pg_indexes
            WHERE indexname = 'documents_embedding_hnsw_idx'
        );
    """).fetchone()[0]

    if not exists:
        raise RuntimeError("HNSW index is not initialized.")


def search_sections_vector(
    query, embedder: Embedder, conn: psycopg.Connection, num_section_results: int = TOP_K_SECTIONS
) -> list[dict]:
    """Embed the query and perform vector search on the documents table.
    Return `num_section_results` lyric chunk documents ordered by vector similarity to the query embedding.
    """

    query_vector = embedder.encode(query)
    query_str = vec_to_str(query_vector)

    check_vector_index(conn)
    with conn.cursor(row_factory=dict_row) as cur:
        section_vs_results = cur.execute(
            """
            SELECT
                d.song_id,
                s.title,
                s.performer,
                s.wks_on_chart,
                s.peak_pos,
                d.section_id,
                d.section,
                d.num_lines,
                1 - (d.embedding <=> %s::vector) AS vector_similarity
            FROM documents AS d
            JOIN songs AS s
                ON d.song_id = s.song_id
            ORDER BY d.embedding <=> %s::vector
            LIMIT %s
            """,
            (query_str, query_str, num_section_results),
        ).fetchall()

    return section_vs_results


def aggregate_vector_results(
    section_vs_results: list[dict], num_song_results: int = TOP_K_SONGS
) -> list[dict]:
    """Aggregate vector search section results by song, preserving the highest cosine similarity, number of matches and matched sections.
    Return up to `num_song_results` song records ordered by their best section match.
    """

    song_vs_results = {}

    for r in section_vs_results:
        song_id = r["song_id"]
        if song_id not in song_vs_results:
            song_vs_results[song_id] = {
                "song_id": song_id,
                "title": r["title"],
                "performer": r["performer"],
                "wks_on_chart": r["wks_on_chart"],
                "peak_pos": r["peak_pos"],
                "best_vector_similarity": r[
                    "vector_similarity"
                ],  # input ordered by vector similarity
                "num_matches": 0,
                "matched_sections": [],
            }

        song_vs_results[song_id]["num_matches"] = song_vs_results[song_id]["num_matches"] + 1
        song_vs_results[song_id]["matched_sections"].append(
            {
                "section_id": r["section_id"],
                "section": r["section"],
                "num_lines": r["num_lines"],
                "vector_similarity": r["vector_similarity"],
            }
        )

    song_vs_results = sorted(
        song_vs_results.values(), key=lambda x: x["best_vector_similarity"], reverse=True
    )

    if len(song_vs_results) > num_song_results:
        return song_vs_results[:num_song_results]
    else:
        return song_vs_results


def main():
    from src.ingest_postgres import connect_db

    query = "Feeling excited about a new relationship"
    embedder = Embedder()
    with connect_db() as conn:
        top_k_sections = search_sections_vector(query, embedder, conn)

    top_k_songs = aggregate_vector_results(top_k_sections)
    for i, item in enumerate(top_k_songs):
        print(f"------result {i}: song #{item["song_id"]}------")
        print(f"{item["title"]} by {item["performer"]}")
        print(f"popularity: wks_on_chart={item["wks_on_chart"]}, peak_pos={item["peak_pos"]}.")
        print(
            f"similarity: best_vector_similarity={item["best_vector_similarity"]} with {item["num_matches"]} matches in top {TOP_K_SECTIONS}."
        )


if __name__ == "__main__":
    main()
