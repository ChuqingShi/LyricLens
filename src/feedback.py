import psycopg


def ensure_feedback_table(conn: psycopg.Connection) -> None:
    """Create the feedback table if it does not exist."""

    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            feedback_id SERIAL PRIMARY KEY,
            query TEXT,
            song_title TEXT,
            performer TEXT,
            rating INTEGER,
            created_at TIMESTAMP DEFAULT now()
        );
        """)


def log_feedback(
    conn: psycopg.Connection, query: str, song_title: str, performer: str, rating: int
) -> None:
    """Insert a single thumbs up/down rating for a recommended song."""

    conn.execute(
        """
        INSERT INTO feedback (query, song_title, performer, rating)
        VALUES (%s, %s, %s, %s)
        """,
        (query, song_title, performer, rating),
    )
