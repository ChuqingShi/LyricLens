import requests
import pandas as pd
import random
import time
from tqdm.auto import tqdm

from pathlib import Path
from .clean_hot100 import OUTPUT_DIR, filter_hot100_song

CHECKPOINT_OUTPUT = Path(OUTPUT_DIR) / "checkpoint.parquet"
HOT100_LYRICS_OUTPUT_NAME = "hot-100-lyrics-new.parquet"


LRCLIB_SEARCH_URL = "https://lrclib.net/api/search"
HEADERS = {"User-Agent": "LyricLens/0.1.0 (your-email@example.com)"}


def search_song_lyrics(
    title: str, performer: str, session: requests.Session | None = None
) -> str | None:
    """Search for lyrics of a song using LRCLib API.
    Never raise an exception,  return None if lyrics are not found, return "<error>" if there is an error.
    """
    track_name = title.strip().title()
    artist_name = (
        performer.strip()
        .title()
        .replace(" Featuring ", " ")
        .replace(" With ", " ")
        .replace(" X ", " ")
        .replace(" & ", " ")
    )  # format featuring artists, and collaborators
    q = f"{track_name} {artist_name}"

    client = session or requests  # session for batch_search, requests for single search

    result = None  # lyrics not found yet

    try:
        response = client.get(
            LRCLIB_SEARCH_URL,
            params={"q": q},
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


def load_checkpoint(songs_df: pd.DataFrame, resume: bool = False) -> pd.DataFrame:
    """Resume loading data from an existing checkpoint, or loading data from scratch."""
    checkpoint_df = pd.DataFrame()
    num_chpt = 0

    if resume and CHECKPOINT_OUTPUT.exists():
        checkpoint_df = pd.read_parquet(CHECKPOINT_OUTPUT)
        num_chpt = len(checkpoint_df)
        print(f"Resuming {num_chpt} records from {CHECKPOINT_OUTPUT}.")

    lyrics_df = songs_df.copy()
    lyrics_df["plain_lyrics"] = pd.NA
    lyrics_df = pd.concat([checkpoint_df, lyrics_df.iloc[num_chpt:]])
    print(f"Loading {len(lyrics_df) - num_chpt} records from scratch.")

    return lyrics_df


def save_checkpoint(lyrics_df: pd.DataFrame) -> None:
    """Save the current progress."""
    lyrics_df.to_parquet(CHECKPOINT_OUTPUT, index=False)  # rewrite
    print(f"{len(lyrics_df)} songs saved at {CHECKPOINT_OUTPUT}.")


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

    save_checkpoint(lyrics_df)
    return lyrics_df


def retry_batch_error(lyrics_df: pd.DataFrame) -> pd.DataFrame:
    """Retry songs whose plain_lyrics value is '<error>'."""

    lyrics_0error_df = lyrics_df.copy()

    error_indices = lyrics_0error_df.index[lyrics_0error_df["plain_lyrics"] == "<error>"]

    if error_indices.empty:
        print("All request errors have been resolved.")
        return lyrics_0error_df

    print("Retrying search with request errors...")
    print(f"{len(error_indices)} songs have request errors before retry.")

    with requests.Session() as session:
        session.headers.update(HEADERS)

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
    print(f"{remaining_errors} errors remain after retry.")
    return lyrics_0error_df


def primary_performer(performer: str) -> str:
    preformer = performer.strip().title()
    for sep in [" Featuring ", " With ", " X ", " & ", ", "]:
        preformer = preformer.split(sep, 1)[0]
    return f"{preformer}*"


def retry_batch_none(
    lyrics_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retry songs whose plain_lyrics value is None."""

    lyrics_0none_df = lyrics_df.copy()

    none_indices = lyrics_0none_df.index[
        lyrics_0none_df["plain_lyrics"].isna()
    ]  # including None and pd.NA, but lyrics_df should not have pd.NA values

    if none_indices.empty:
        print("All missing lyrics have been resolved.")
        return lyrics_0none_df, pd.DataFrame()

    print("Retrying search for missing lyrics...")
    print(f"{len(none_indices)} songs have missing lyrics before retry.")

    with requests.Session() as session:
        session.headers.update(HEADERS)

        for index in tqdm(none_indices):
            row = lyrics_0none_df.loc[index]
            performer_retry = primary_performer(row["performer"])
            lyrics = search_song_lyrics(
                title=row["title"], performer=performer_retry, session=session
            )
            if lyrics is not None:  # could be "<error>"
                row["performer"] = performer_retry
                lyrics_0none_df.at[index, "plain_lyrics"] = lyrics

    remaining_nones = lyrics_0none_df["plain_lyrics"].isna()
    lyrics_none_df = lyrics_0none_df[remaining_nones]
    lyrics_0none_df = lyrics_0none_df[~remaining_nones]
    save_checkpoint(lyrics_0none_df)

    print(f"{len(lyrics_none_df)} songs still have missing lyrics after retry, and are removed.")
    return lyrics_0none_df, lyrics_none_df


def retry_batch_lyrics(
    lyrics_df: pd.DataFrame,
    max_retry_time: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retry songs whose plain_lyrics value is '<error>' or None in a loop."""

    lyrics_ready_df, lyrics_removed_df = retry_batch_none(lyrics_df)

    if max_retry_time < 1:
        raise ValueError("max_retry_time must be at least 1.")

    for retry_number in range(1, max_retry_time + 1):
        error_indices = lyrics_ready_df.index[lyrics_ready_df["plain_lyrics"] == "<error>"]

        if error_indices.empty:
            print("There are no request errors.")
            break

        print(f"Retrying {retry_number}/{max_retry_time}: ")

        lyrics_ready_df = retry_batch_error(lyrics_ready_df)  # could result in new None
        lyrics_ready_df, lyrics_removing_df = retry_batch_none(lyrics_ready_df)

        lyrics_removed_df = pd.concat([lyrics_removed_df, lyrics_removing_df])

        save_checkpoint(lyrics_ready_df)

    lyrics_errors_df = lyrics_ready_df[lyrics_ready_df["plain_lyrics"] == "<error>"]
    lyrics_removed_df = pd.concat([lyrics_removed_df, lyrics_errors_df])
    print(
        f"Retry finished. A total of {len(lyrics_removed_df)} songs are removed because no lyrics could be retrieved from LRCLib."
    )
    return lyrics_ready_df, lyrics_removed_df


def generate_batch_lyrics(
    songs_df: pd.DataFrame, save: bool = True, output_name: str = HOT100_LYRICS_OUTPUT_NAME
) -> pd.DataFrame:
    lyrics_df = search_batch_lyrics(songs_df)
    lyrics_ready_df, _ = retry_batch_lyrics(lyrics_df)
    print(f"Successfully retrieved lyrics for {len(lyrics_ready_df)} songs.")

    if save:
        lyrics_ready_df.to_parquet(Path(OUTPUT_DIR) / output_name, index=False)
        print(f"Final results saved at {Path(OUTPUT_DIR) / output_name}.")

    print("Finalizing lyrics generation: removing intermediate checkpoint.")
    CHECKPOINT_OUTPUT.unlink()  # remove checkpoint at last no matter save or not
    return lyrics_ready_df


if __name__ == "__main__":
    from src.clean_hot100 import filter_hot100_song

    hot100_song_df = pd.read_parquet(Path(OUTPUT_DIR) / "hot-100-song_current.parquet")

    start_wk = pd.Timestamp("2020-01-01")
    end_wk = pd.Timestamp("2021-01-01")
    filtered_hot100_song_df = filter_hot100_song(hot100_song_df, start_wk=start_wk, end_wk=end_wk)

    print("Initializing lyrics generation: removing existing checkpoint.")
    CHECKPOINT_OUTPUT.unlink(missing_ok=True)  # clear checkpoint for a fresh start

    filtered_hot100_lyrics_df = generate_batch_lyrics(
        filtered_hot100_song_df, output_name=f"hot-100-lyrics_{start_wk}_{end_wk}.parquet"
    )
    # HOT100_LYRICS_OUTPUT = Path(OUTPUT_DIR) / HOT100_LYRICS_OUTPUT_NAME
    # filtered_hot100_lyrics_df = search_batch_lyrics(filtered_hot100_song_df)
    # # filtered_hot100_lyrics_0error_df = retry_batch_error(filtered_hot100_lyrics_df)
    # # filtered_hot100_lyrics_0error_df.to_parquet(HOT100_LYRICS_OUTPUT, index=False)

    # # filtered_hot100_lyrics_0error_df = retry_batch_error(filtered_hot100_lyrics_df)
    # # filtered_hot100_lyrics_0none_df = retry_batch_none(filtered_hot100_lyrics_0error_df)
    # # filtered_hot100_lyrics_0none_0error_df = retry_batch_error(filtered_hot100_lyrics_0none_df)
    # # filtered_hot100_lyrics_0none_0error_df = filtered_hot100_lyrics_0none_0error_df[filtered_hot100_lyrics_0none_0error_df["plain_lyrics"].notna()]
    # # filtered_hot100_lyrics_0none_0error_df.to_parquet(HOT100_LYRICS_OUTPUT, index=False)

    # filtered_hot100_lyrics_df = retry_batch_lyrics(filtered_hot100_lyrics_df)
    # filtered_hot100_lyrics_df.to_parquet(HOT100_LYRICS_OUTPUT, index=False)
    # print(f"Saved final results to {HOT100_LYRICS_OUTPUT}.")
    # CHECKPOINT_OUTPUT.unlink()
