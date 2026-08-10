from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

HOT100_OUTPUT = RAW_DATA_DIR / "hot-100-current.csv"
CHECKPOINT_OUTPUT = PROCESSED_DATA_DIR / "checkpoint.parquet"

# filter time period
START_WEEK = "2020-01-01"
END_WEEK = "2026-08-01"
HOT100_LYRICS_OUTPUT = PROCESSED_DATA_DIR / f"hot-100-lyrics_{START_WEEK}_{END_WEEK}.parquet"
