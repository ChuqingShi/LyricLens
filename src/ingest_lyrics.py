import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import psycopg
from .config import (
    HOT100_LYRICS_OUTPUT,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
    POSTGRES_DB,
    HOST,
    HOST_PORT,
)


def setup_postgres() -> psycopg.Connection:
    print("Connecting to PostgreSQL...")
    conn = psycopg.connect(
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=HOST,
        port=HOST_PORT,
        connect_timeout=5,
    )
    print("Connected!")

    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    print("pgvector extension ready!")

    return conn


def prepare_tables(conn: psycopg.Connection) -> None:
    print("Preparing database tables...")
    conn.execute("DROP TABLE IF EXISTS documents;")
    conn.execute("DROP TABLE IF EXISTS songs;")
    conn.execute("""
        CREATE TABLE songs (
            song_id INTEGER PRIMARY KEY,
            title TEXT,
            performer TEXT,
            wks_on_chart INTEGER,
            peak_pos INTEGER
        );
    """)
    conn.execute("""
        CREATE TABLE documents (
            document_id SERIAL PRIMARY KEY,
            song_id INTEGER REFERENCES songs(song_id) ON DELETE CASCADE,
            section_id INTEGER,
            section TEXT,
            num_lines INTEGER,
            embedding VECTOR(384),

            UNIQUE (song_id, section_id)
        );
    """)
    conn.commit()
    print("Database tables ready.")


def check_tables(conn: psycopg.Connection) -> tuple[int, int]:
    num_songs, num_documents = conn.execute("""
        SELECT
            (SELECT COUNT(*) FROM songs) AS num_songs,
            (SELECT COUNT(*) FROM documents) AS num_documents;
        """).fetchone()
    print(f"{num_songs} songs and {num_documents} lyrics chunk documents in database.")
    return num_songs, num_documents


def load_lyrics() -> pd.DataFrame:
    return pd.read_parquet(HOT100_LYRICS_OUTPUT)


def ingest_songs(conn: psycopg.Connection, lyrics_df: pd.DataFrame) -> None:
    num_songs = conn.execute("""
        SELECT COUNT(*) 
        FROM songs;
    """).fetchone()[0]
    print(f"Table songs currently contains {num_songs} songs.")

    print(f"Ingesting {len(lyrics_df)} songs into table songs...")
    try:
        for df_song_id, row in tqdm(lyrics_df.iterrows(), total=len(lyrics_df)):
            conn.execute(
                """
                INSERT INTO songs (song_id, title, performer, wks_on_chart, peak_pos)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    df_song_id,
                    row["title"],
                    row["performer"],
                    row["wks_on_chart"],
                    row["peak_pos"],
                ),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    num_songs = conn.execute("""
        SELECT COUNT(*) 
        FROM songs;
    """).fetchone()[0]
    print(f"Ingestion complete: {num_songs} songs in table songs.")


def vec_to_str(vector: np.ndarray) -> str:
    return "[" + ",".join(str(x) for x in vector) + "]"


def ingest_documents(
    conn,
    documents: list[dict],
    vectors: list[np.ndarray],
) -> None:
    num_documents = conn.execute("""
        SELECT COUNT(*) 
        FROM documents;
    """).fetchone()[0]
    print(f"Table documents currently contains {num_documents} lyrics chunk documents.")

    try:
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
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    num_documents = conn.execute("""
        SELECT COUNT(*) 
        FROM documents;
    """).fetchone()[0]
    print(f"Ingestion complete: {num_documents} lyrics chunk documents in table documents.")


if __name__ == "__main__":
    from src.chunk_lyrics import build_chunk_documents
    from src.embeddings.embed_texts import embed_texts
    from src.embeddings.embedder import Embedder

    conn = setup_postgres()
    prepare_tables(conn)

    lyrics_df = load_lyrics()
    ingest_songs(conn, lyrics_df)

    documents = build_chunk_documents(lyrics_df)
    texts = [doc["section"] for doc in documents]
    embedder = Embedder()
    vectors = embed_texts(texts, embedder)
    ingest_documents(conn, documents, vectors)

    check_tables(conn)
