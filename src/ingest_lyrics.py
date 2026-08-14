import pandas as pd
import numpy as np
import psycopg
from .config import HOT100_LYRICS_OUTPUT


def load_lyrics() -> pd.DataFrame:
    return pd.read_parquet(HOT100_LYRICS_OUTPUT)


def ingest_songs(conn: psycopg.Connection, lyrics_df: pd.DataFrame) -> dict[int, int]:
    pass


def ingest_documents(
    conn,
    documents: list[dict],
    vectors: list[np.ndarray],
    song_id_map: dict[int, int],
) -> None:
    pass


if __name__ == "__main__":
    conn = psycopg.connect("postgresql://lyricslens:pswd@localhost:5432/lyricslensDB")
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    pass
