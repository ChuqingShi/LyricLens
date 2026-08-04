import requests
import pandas as pd
import random
import time
from tqdm.auto import tqdm

from pathlib import Path
from .clean_hot100 import OUTPUT_DIR, filter_hot100_song

CHECKPOINT_OUTPUT = Path(OUTPUT_DIR) / "checkpoint.parquet"
HOT100_WITH_LYRICS_OUTPUT = Path(OUTPUT_DIR) / "hot-100-with-lyrics.parquet"


LRCLIB_SEARCH_URL = "https://lrclib.net/api/search"
HEADERS = {"User-Agent": "LyricLens/0.1.0 (your-email@example.com)"}


def search_song_lyrics(
    title: str, performer: str, session: requests.Session | None = None
) -> str | None:
    """Search for lyrics of a song using LRCLib API.
    Never raise an exception,  return None if lyrics are not found, return "<error>" if there is an error.
    """
    track_name = title.strip().title()
    artist_name = performer.strip().title()

    client = session or requests  # session for batch_search, requests for single search

    result = None  # lyrics not found yet

    try:
        response = client.get(
            LRCLIB_SEARCH_URL,
            params={"track_name": track_name, "artist_name": artist_name},
            headers=None if session else HEADERS,
            timeout=(3, 5),  # wait for 3s until connect timeout, 5s until read timeout
        )
        response.raise_for_status()
        results = response.json()
        if results:
            result = results[0].get("plainLyrics")

    except requests.HTTPError as e:
        print(f"Failed (LRCLib): {title} by {performer}: {e}")

        status_code = e.response.status_code
        if status_code == 429:
            print("Rate limit exceeded.")
        elif status_code >= 500:
            print("Server error.")
        else:
            print(f"HTTP error: {status_code}")
        result = "<error>"

    except requests.RequestException as e:  # Timeout, ConnectionError, TooManyRedirects, etc.
        print(f"Failed (LRCLib): {title} by {performer}: {e}")
        result = "<error>"

    return result


def load_checkpoint(songs_df: pd.DataFrame) -> pd.DataFrame:
    """Load an existing checkpoint, or initialize a new result dataframe."""
    if CHECKPOINT_OUTPUT.exists():
        result_df = pd.read_parquet(CHECKPOINT_OUTPUT)
        print(f"Resuming {len(result_df)} records from {CHECKPOINT_OUTPUT}.")
        return result_df

    lyrics_df = songs_df.copy()
    lyrics_df["plain_lyrics"] = pd.NA

    return lyrics_df


def save_checkpoint(lyrics_df: pd.DataFrame) -> None:
    """Save the current progress."""
    lyrics_df.to_parquet(CHECKPOINT_OUTPUT, index=False)


def search_batch_lyrics(
    songs_df: pd.DataFrame,
    checkpoint_every: int = 100,
) -> pd.DataFrame:
    """Download lyrics for all songs with checkpointing."""

    lyrics_df = load_checkpoint(songs_df)

    pending_indices = lyrics_df.index[lyrics_df["plain_lyrics"].apply(lambda x: x is pd.NA)]

    with requests.Session() as session:
        session.headers.update(HEADERS)

        for count, index in tqdm(enumerate(pending_indices, start=1), total=len(pending_indices)):
            row = lyrics_df.loc[index]
            lyrics_df.at[index, "plain_lyrics"] = search_song_lyrics(
                row["title"], row["performer"], session=session
            )
            time.sleep(
                random.uniform(0.2, 0.5)
            )  # wait 200-500ms between requests to avoid rate limiting

            if count % checkpoint_every == 0:
                save_checkpoint(lyrics_df)
                print(f"Saved checkpoints of {count} songs at {CHECKPOINT_OUTPUT}.")

    save_checkpoint(lyrics_df)
    return lyrics_df


def retry_batch_errors(
    lyrics_df: pd.DataFrame,
    max_retry_time: int = 5,
) -> pd.DataFrame:
    """Retry songs whose plain_lyrics value is '<error>'."""

    if max_retry_time < 1:
        raise ValueError("max_retry_time must be at least 1.")

    lyrics_0error_df = lyrics_df.copy()

    with requests.Session() as session:
        session.headers.update(HEADERS)

        for retry_number in range(1, max_retry_time + 1):
            error_indices = lyrics_0error_df.index[lyrics_0error_df["plain_lyrics"] == "<error>"]

            if len(error_indices) == 0:
                print("All request errors have been resolved.")
                break

            print(
                f"Retry {retry_number}/{max_retry_time}: " f"{len(error_indices)} songs remaining."
            )

            for index in error_indices:
                row = lyrics_0error_df.loc[index]
                lyrics = search_song_lyrics(
                    title=row["title"], performer=row["performer"], session=session
                )
                if lyrics != "<error>":
                    lyrics_0error_df.at[index, "plain_lyrics"] = lyrics

                time.sleep(random.uniform(0.2, 0.5))

            save_checkpoint(lyrics_0error_df)

    remaining_errors = (lyrics_0error_df["plain_lyrics"] == "<error>").sum()
    print(f"Retry finished. {remaining_errors} errors remain.")
    return lyrics_0error_df


def retry_batch_none(
    lyrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Retry songs whose plain_lyrics value is None."""

    lyrics_0none_df = lyrics_df.copy()

    with requests.Session() as session:
        session.headers.update(HEADERS)

        none_indices = lyrics_0none_df.index[
            lyrics_0none_df["plain_lyrics"].isna()
        ]  # including None and pd.NA, but lyrics_df should not have pd.NA values

        if len(none_indices) == 0:
            print("All None values have been resolved.")

        print(f"{len(none_indices)} songs have None lyrics before retry.")

        for index in tqdm(none_indices):
            row = lyrics_0none_df.loc[index]
            performer_retry = (
                row["performer"].split(" Featuring ")[0].strip()
            )  # remove featuring artists
            lyrics = search_song_lyrics(
                title=row["title"], performer=performer_retry, session=session
            )
            if lyrics is not None:  # could be "<error>"
                row["performer"] = performer_retry
                lyrics_0none_df.at[index, "plain_lyrics"] = lyrics

        save_checkpoint(lyrics_0none_df)

    remaining_nones = lyrics_0none_df["plain_lyrics"].isna().sum()
    print(f"Retry finished. {remaining_nones} Nones remain.")
    return lyrics_0none_df


# def finalize_batch_lyrics(
#     lyrics_df: pd.DataFrame,
#     save: bool = True,
#     output_name: str = HOT100_WITH_LYRICS_OUTPUT
# ) -> pd.DataFrame:
#     # remove CHECKPOINT_OUTPUT if it exists

if __name__ == "__main__":
    from src.clean_hot100 import filter_hot100_song

    # Example usage
    hot100_song_df = pd.read_parquet(Path(OUTPUT_DIR) / "hot-100-song_current.parquet")
    filtered_hot100_song_df = filter_hot100_song(
        hot100_song_df, start_wk="2020-01-01", end_wk="2021-01-01", save=False
    )

    filtered_hot100_lyrics_df = search_batch_lyrics(filtered_hot100_song_df)
    filtered_hot100_lyrics_0error_df = retry_batch_errors(filtered_hot100_lyrics_df)
    filtered_hot100_lyrics_0none_df = retry_batch_none(filtered_hot100_lyrics_0error_df)
    filtered_hot100_lyrics_0none_0error_df = retry_batch_none(filtered_hot100_lyrics_0none_df)
    filtered_hot100_lyrics_0none_0error_df.to_parquet(HOT100_WITH_LYRICS_OUTPUT, index=False)
    print(f"Saved final results to {HOT100_WITH_LYRICS_OUTPUT}.")
