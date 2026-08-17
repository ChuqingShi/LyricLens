import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import json
import psycopg
from .config import (
    HOT100_LYRICS_OUTPUT,
    HOT100_CHUNKS_OUTPUT,
    EMBEDDINGS_OUTPUT,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
    POSTGRES_DB,
    HOST,
    HOST_PORT,
    EMBEDDING_DIM,
)


def connect_db() -> psycopg.Connection:
    """Connect to the PostgreSQL database POSTGRES_DB and return the connection."""

    print(f"Connecting to PostgreSQL database {POSTGRES_DB}...")
    conn = psycopg.connect(
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=HOST,
        port=HOST_PORT,
        connect_timeout=5,
    )
    print("Connected!")
    return conn


def setup_db(conn: psycopg.Connection) -> None:
    """Set up the pgvector extension."""

    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    print("pgvector extension ready!")


def create_tables(conn: psycopg.Connection) -> None:
    """Create the songs and documents tables if they do not exist."""

    print("Preparing database tables...")
    conn.execute("DROP TABLE IF EXISTS documents;")
    conn.execute("DROP TABLE IF EXISTS songs;")
    conn.execute("""
        CREATE TABLE songs (
            song_id INTEGER PRIMARY KEY,
            title TEXT,
            performer TEXT,
            lyrics TEXT,
            wks_on_chart INTEGER,
            peak_pos INTEGER
        );
        """)
    conn.execute(f"""
        CREATE TABLE documents (
            document_id SERIAL PRIMARY KEY,
            song_id INTEGER REFERENCES songs(song_id) ON DELETE CASCADE,
            section_id INTEGER,
            section TEXT,
            num_lines INTEGER,
            embedding VECTOR({EMBEDDING_DIM}),

            UNIQUE (song_id, section_id)
        );
        """)
    print("Database tables ready.")


def check_tables(conn: psycopg.Connection) -> tuple[int, int]:
    """Check current num_songs and num_documents in tables."""

    num_songs, num_documents = conn.execute("""
        SELECT
            (SELECT COUNT(*) FROM songs) AS num_songs,
            (SELECT COUNT(*) FROM documents) AS num_documents;
        """).fetchone()
    print(f"{num_songs} songs and {num_documents} lyrics chunk documents in database.")
    return num_songs, num_documents


def load_lyrics() -> pd.DataFrame:
    """Load stored lyrics from HOT100_LYRICS_OUTPUT."""

    return pd.read_parquet(HOT100_LYRICS_OUTPUT)


def ingest_songs(conn: psycopg.Connection, lyrics_df: pd.DataFrame) -> None:
    """Ingest song metadata into the songs table.
    song_id follows the lyrics_df index and is inconsistent with build_chunk_documents.
    """

    num_songs = conn.execute("""
        SELECT COUNT(*) 
        FROM songs;
    """).fetchone()[0]
    print(f"Table songs currently contains {num_songs} songs.")

    print(f"Ingesting {len(lyrics_df)} songs into the songs table...")
    for df_song_id, row in tqdm(lyrics_df.iterrows(), total=len(lyrics_df)):
        conn.execute(
            """
            INSERT INTO songs (song_id, title, performer, lyrics, wks_on_chart, peak_pos)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                df_song_id,
                row["title"],
                row["performer"],
                row["plain_lyrics"],
                row["wks_on_chart"],
                row["peak_pos"],
            ),
        )

    num_songs = conn.execute("""
        SELECT COUNT(*) 
        FROM songs;
    """).fetchone()[0]
    print(f"Ingestion completed: {num_songs} songs in the songs table.")


def load_documents() -> list[dict]:
    """Load documents from HOT100_CHUNKS_OUTPUT"""

    with open(HOT100_CHUNKS_OUTPUT, "r", encoding="utf-8") as f:
        documents = json.load(f)
    return documents


def load_vectors() -> np.ndarray:
    """Load lyrics chunk embeddings from EMBEDDINGS_OUTPUT"""

    vectors = np.load(EMBEDDINGS_OUTPUT)
    return vectors


def vec_to_str(vector: np.ndarray) -> str:
    """Convert a NumPy vector to a pgvector string."""

    return "[" + ",".join(str(x) for x in vector) + "]"


def ingest_documents(
    conn,
    documents: list[dict],
    vectors: np.ndarray,
) -> None:
    """Ingest lyrics chunks, their metadata, and embeddings into the documents table with dimension mismatch error handling."""

    num_documents = conn.execute("""
        SELECT COUNT(*) 
        FROM documents;
    """).fetchone()[0]
    print(f"Table documents currently contains {num_documents} lyrics chunk documents.")

    print(f"Ingesting {len(documents)} lyrics chunk documents into the table documents...")
    if len(documents) != len(vectors):
        raise ValueError(f"Expected {len(documents)} vectors, got {len(vectors)}.")

    for doc, vec in tqdm(zip(documents, vectors), total=len(documents)):
        conn.execute(
            """
            INSERT INTO documents (song_id, section_id, section, num_lines, embedding)
            VALUES (%s, %s, %s, %s, %s::vector)
            """,
            (
                doc["df_song_id"],
                doc["section_id"],
                doc["section"],
                doc["num_lines"],
                vec_to_str(vec),
            ),
        )

    num_documents = conn.execute("""
        SELECT COUNT(*) 
        FROM documents;
    """).fetchone()[0]
    print(f"Ingestion completed: {num_documents} lyrics chunk documents in the documents table.")


def create_vector_index(conn: psycopg.Connection) -> None:
    """Create an HNSW index on embeddings in the `documents` table for vector search."""

    print("Creating HNSW index...")  # for ANN search
    conn.execute("""
        CREATE INDEX IF NOT EXISTS documents_embedding_hnsw_idx
        ON documents
        USING hnsw (embedding vector_cosine_ops);
    """)
    print("HNSW index ready.")


def main():
    lyrics_df = load_lyrics()
    documents = load_documents()
    vectors = load_vectors()

    with connect_db() as conn:  # automatically take care of commit(), rollback(), and close()
        setup_db(conn)
        create_tables(conn)

        ingest_songs(conn, lyrics_df)
        ingest_documents(conn, documents, vectors)

        check_tables(conn)
        create_vector_index(conn)


if __name__ == "__main__":
    main()
