import requests
import pandas as pd
from tqdm.auto import tqdm
import re
import random
import time
from .config import CHECKPOINT_OUTPUT, HOT100_LYRICS_OUTPUT, CHECKPOINT_EVERY, MAX_RETRIES

LRCLIB_SEARCH_URL = "https://lrclib.net/api/search"
HEADERS = {"User-Agent": "LyricLens/0.1.0 (your-email@example.com)"}

PATTERN = r"\s+(?:feat(?:uring)?\.?|with|x|&)\s+"  # >=1 spaces + "feat" with optional "uring" with optional "."/ "with"/ "x"/ "&" + >=1 spaces


def search_song_lyrics(
    title: str, performer: str, session: requests.Session | None = None
) -> str | None:
    """Search for a song's lyrics using LRCLib API.
    Never raise an exception,  return None if lyrics are not found, return "<error>" if there is an error.
    """

    # track_name = title.strip().title()
    # artist_name = (
    #     performer.strip()
    #     .title() # for easy handling afterwards but actually changes original
    #     .replace(" Featuring ", " ")
    #     .replace(" With ", " ")
    #     .replace(" X ", " ")
    #     .replace(" & ", " ")
    # )  # format featuring artists, and collaborators
    track_name = title.strip()
    artist_name = re.sub(
        PATTERN, " ", performer, flags=re.IGNORECASE
    ).strip()  # ignore upper/lowercase letters when matching

    q = f"{track_name} {artist_name}"

    client = session or requests  # session for batch_search, requests for single search

    result = None  # lyrics not found yet

    try:
        response = client.get(
            LRCLIB_SEARCH_URL,
            params={
                "q": q
            },  # q param is tested to give better results than track_name and artist_name
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


def load_checkpoint(songs_df: pd.DataFrame, resume_pos: int = 0) -> pd.DataFrame:
    """Load data from scratch or resume from an existing checkpoint.
    Input:
        `resume_pos`: Number of existing records to keep before resuming from checkpoint. Set to 0 to start from scratch.
        Make sure songs_df and the existing checkpoint align.
    """

    if resume_pos < 0:
        raise ValueError("resume_pos must not be negative.")
    elif resume_pos == 0:
        print("Initializing lyrics generation: removing existing checkpoint.")
        CHECKPOINT_OUTPUT.unlink(missing_ok=True)  # clear checkpoint for a fresh start
        checkpoint_df = pd.DataFrame()
    else:
        if CHECKPOINT_OUTPUT.exists():
            checkpoint_df = pd.read_parquet(CHECKPOINT_OUTPUT)
            print(f"Resuming {resume_pos} records from {CHECKPOINT_OUTPUT}.")
        else:
            raise ValueError("There is no exsiting checkpoint. resume_pos must be 0.")

    lyrics_df = songs_df.copy()
    lyrics_df["plain_lyrics"] = pd.NA
    lyrics_df = pd.concat(
        [checkpoint_df.iloc[:resume_pos], lyrics_df.iloc[resume_pos:]], ignore_index=True
    )  # REMARK: parameter 'resume_pos' needs to be properly selected to keep checkpoint_df.iloc[:resume_pos] without pd.NA
    print(f"Loading {len(lyrics_df) - resume_pos} records from scratch.")

    return lyrics_df


def save_checkpoint(lyrics_df: pd.DataFrame, logging: bool = False) -> None:
    """Save a checkpoint and optionally log the current progress."""

    lyrics_df.to_parquet(CHECKPOINT_OUTPUT, index=False)  # rewrite

    if logging:
        num_records = sum(
            lyrics_df["plain_lyrics"].apply(lambda x: x is not pd.NA)
        )  # OK: all pd.NA is from fresh df
        print(f"{num_records} songs with lyrics saved at {CHECKPOINT_OUTPUT}.")


def search_batch_lyrics(
    songs_df: pd.DataFrame,
    resume_pos: int = 0,
    checkpoint_every: int = CHECKPOINT_EVERY,
) -> pd.DataFrame:
    """Download lyrics for all songs with request throttling and batched checkpointing."""

    lyrics_df = load_checkpoint(songs_df, resume_pos)

    pending_indices = lyrics_df.index[
        lyrics_df["plain_lyrics"].apply(lambda x: x is pd.NA)
    ]  # OK: all pd.NA is added from songs_df not parquet

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
                save_checkpoint(lyrics_df, logging=True)

    save_checkpoint(lyrics_df, logging=True)
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
                lyrics_0error_df.at[index, "plain_lyrics"] = lyrics  # could be None

            time.sleep(random.uniform(0.2, 0.5))

    save_checkpoint(lyrics_0error_df)

    num_remaining_errors = (lyrics_0error_df["plain_lyrics"] == "<error>").sum()
    print(f"{num_remaining_errors} errors remain after retry.")
    return lyrics_0error_df


def primary_performer(performer: str) -> str:
    """Generate the primary performer using PATTERN"""

    performer = re.split(PATTERN, performer, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    return f"{performer}*"


def retry_batch_none(
    lyrics_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retry songs whose plain_lyrics value is None (missing) using a modified search.
    Output:
        Data with non-None lyrics and data with lyrics still None after retrying.
    """

    lyrics_0none_df = lyrics_df.copy()

    none_indices = lyrics_0none_df.index[
        lyrics_0none_df["plain_lyrics"].isna()
    ]  # OK: the function is only called after search_batch_lyrics, so lyrics_df should not have pd.NA values.

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
                # row["performer"] = performer_retry
                lyrics_0none_df.at[index, "plain_lyrics"] = lyrics

        time.sleep(random.uniform(0.2, 0.5))

    remaining_nones = lyrics_0none_df["plain_lyrics"].isna()
    lyrics_none_df = lyrics_0none_df[remaining_nones]
    lyrics_0none_df = lyrics_0none_df[~remaining_nones]
    save_checkpoint(lyrics_0none_df)

    print(f"{len(lyrics_none_df)} songs still have missing lyrics after retry, and are removed.")
    return lyrics_0none_df, lyrics_none_df


def retry_batch_lyrics(
    lyrics_df: pd.DataFrame,
    max_retries: int = MAX_RETRIES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retry songs whose plain_lyrics value is '<error>' or None.
    After each retry, '<error>' entries either remain '<error>', return valid lyrics, or return None.
    After each retry, None entries either remain None and get removed, or saved otherwise.
    After all retries, remaining '<error>' entries are removed.
    Output:
        Data with valid lyrics and data with unresolved None or "<error>" after max_retries.
    """

    lyrics_ready_df, lyrics_removed_df = retry_batch_none(lyrics_df)

    if max_retries < 1:
        raise ValueError("max_retries must be at least 1.")

    for retry_number in range(1, max_retries + 1):
        error_indices = lyrics_ready_df.index[lyrics_ready_df["plain_lyrics"] == "<error>"]

        if error_indices.empty:
            print("There are no request errors.")
            break

        print(f"Retrying {retry_number}/{max_retries}: ")

        lyrics_ready_df = retry_batch_error(lyrics_ready_df)  # could result in new None
        lyrics_ready_df, lyrics_removing_df = retry_batch_none(lyrics_ready_df)

        lyrics_removed_df = pd.concat([lyrics_removed_df, lyrics_removing_df], ignore_index=True)

        save_checkpoint(lyrics_ready_df)

    remaining_errors = lyrics_ready_df["plain_lyrics"] == "<error>"
    lyrics_errors_df = lyrics_ready_df[remaining_errors]
    lyrics_ready_df = lyrics_ready_df[~remaining_errors]
    save_checkpoint(lyrics_ready_df)

    lyrics_removed_df = pd.concat([lyrics_removed_df, lyrics_errors_df], ignore_index=True)
    print(
        f"{len(lyrics_errors_df)} songs still have request errors after {max_retries} retries, and are removed."
    )
    print(
        f"Retry finished. A total of {len(lyrics_removed_df)} songs are removed because no lyrics could be retrieved from LRCLib."
    )
    return lyrics_ready_df, lyrics_removed_df


def generate_batch_lyrics(
    songs_df: pd.DataFrame,
    resume_pos: int = 0,
    checkpoint_every: int = CHECKPOINT_EVERY,
    max_retries: int = MAX_RETRIES,
    save: bool = True,
) -> pd.DataFrame:
    """Generate and save hot 100 lyrics data."""

    lyrics_df = search_batch_lyrics(
        songs_df, resume_pos=resume_pos, checkpoint_every=checkpoint_every
    )
    lyrics_ready_df, _ = retry_batch_lyrics(lyrics_df, max_retries=max_retries)
    lyrics_ready_df = lyrics_ready_df.sort_values(["title", "performer"])
    print(f"Successfully retrieved lyrics for {len(lyrics_ready_df)} songs.")

    if save:
        lyrics_ready_df.to_parquet(HOT100_LYRICS_OUTPUT, index=False)
        print(f"Final results saved at {HOT100_LYRICS_OUTPUT}.")

    print("Finalizing lyrics generation: removing intermediate checkpoint.")
    CHECKPOINT_OUTPUT.unlink()  # remove checkpoint at last no matter save or not
    return lyrics_ready_df


def main():
    # from src.download_hot100 import download_hot100
    # from src.clean_hot100 import clean_hot100_song, filter_hot100_song
    # from .config import START_WEEK, END_WEEK

    # download_hot100()
    # hot100_song_df = clean_hot100_song()

    from src.clean_hot100 import filter_hot100_song
    from src.config import HOT100_SONG_OUTPUT, START_WEEK, END_WEEK

    hot100_song_df = pd.read_parquet(HOT100_SONG_OUTPUT)

    start_wk = START_WEEK
    end_wk = END_WEEK
    filtered_hot100_song_df = filter_hot100_song(hot100_song_df, start_wk=start_wk, end_wk=end_wk)
    filtered_hot100_lyrics_df = generate_batch_lyrics(filtered_hot100_song_df)


if __name__ == "__main__":
    main()
