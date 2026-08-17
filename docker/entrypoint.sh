#!/bin/sh
set -e

echo "Waiting for database..."
until uv run python - <<'PY'
import sys
from src.ingest_postgres import connect_db

try:
    connect_db().close()
except Exception:
    sys.exit(1)
PY
do
  sleep 2
done

echo "Checking vector index..."
if uv run python - <<'PY'
import sys
from src.ingest_postgres import connect_db
from src.search_pgvector import check_vector_index

try:
    with connect_db() as conn:
        check_vector_index(conn)
except RuntimeError:
    sys.exit(1)
PY
then
  echo "Vector index found, skipping ingest."
else
  echo "Vector index not found, running ingest..."
  uv run python -m src.ingest_postgres
fi

echo "Starting Streamlit..."
exec uv run streamlit run app.py --server.port=8501 --server.address=0.0.0.0
