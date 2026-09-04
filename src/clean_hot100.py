import pandas as pd
from .download_hot100 import HOT100_OUTPUT
from .song_id import compute_song_id
from .config import (
    HOT100_ALLSONG_OUTPUT,
    HOT100_SONGS_OUTPUT,
    PROCESSED_DATA_DIR,
    START_WEEK,
    END_WEEK,
)


def clean_hot100_chart(
    save: bool = True, output_name: str = "hot-100-chart_current.parquet"
) -> pd.DataFrame:
    """Clean and save Billboard Hot 100 chart data.
    (handling dates, sorting for readability, standardizing missing values)"""

    hot100_chart_df = pd.read_csv(HOT100_OUTPUT)

    hot100_chart_df["chart_week"] = pd.to_datetime(hot100_chart_df["chart_week"])
    hot100_chart_df = hot100_chart_df.sort_values(["chart_week", "current_week"])

    hot100_chart_df["last_week"] = hot100_chart_df["last_week"].replace(0, pd.NA).astype("Int64")

    if save:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        hot100_chart_df.to_parquet(PROCESSED_DATA_DIR / output_name, index=False)
        print(f"Saved hot100_chart_df to {PROCESSED_DATA_DIR / output_name}.")

    print(f"{len(hot100_chart_df)} entries loaded from {HOT100_OUTPUT}.")

    return hot100_chart_df


def clean_hot100_song(save: bool = False) -> pd.DataFrame:
    """Clean and save Billboard Hot 100 song data.
    (removing unnecessary info, removing exact duplicates, generating popularity features, sorting for readability)
    """

    hot100_chart_df = pd.read_csv(HOT100_OUTPUT)
    hot100_chart_df["chart_week"] = pd.to_datetime(hot100_chart_df["chart_week"])

    hot100_song_df = hot100_chart_df[
        ["title", "performer", "chart_week", "wks_on_chart", "peak_pos"]
    ].copy()
    hot100_song_df = hot100_song_df.sort_values(
        ["title", "performer", "chart_week", "wks_on_chart"]
    )

    # match by exact "title" & "performer", define row-level song identity

    # remove duplicates
    hot100_song_df = hot100_song_df.groupby(
        ["title", "performer", "chart_week"], as_index=False
    ).agg(  # sort=True by default
        wks_on_chart=("wks_on_chart", "max"),
        peak_pos=("peak_pos", "min"),
    )

    # rebuild wks_on_chart: 1, 2, 3, ...
    hot100_song_df["wks_on_chart"] = hot100_song_df.groupby(["title", "performer"]).cumcount() + 1

    # Rebuild peak_pos: best ever position achieved up through that week
    hot100_song_df["peak_pos"] = hot100_song_df.groupby(["title", "performer"])["peak_pos"].cummin()

    # Regenerate popularity features for each song:
    #   chart_weeks: a list of all weeks on chart
    #   wks_on_chart: the total number of weeks on chart
    #   peak_pos: the best ever position on chart
    hot100_song_df = (
        hot100_song_df.sort_values(["title", "performer", "chart_week"])
        .groupby(["title", "performer"], as_index=False)
        .agg(
            chart_weeks=(
                "chart_week",
                lambda s: s.dt.strftime("%Y-%m-%d").tolist(),
            ),  # ordered from earlier sort_values
            wks_on_chart=("wks_on_chart", "max"),
            peak_pos=("peak_pos", "min"),
        )
    )

    if save:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        hot100_song_df.to_parquet(HOT100_ALLSONG_OUTPUT, index=False)
        print(f"Saved hot100_song_df to {HOT100_ALLSONG_OUTPUT}.")

    print(f"{len(hot100_song_df)} songs recorded in {HOT100_OUTPUT}.")

    return hot100_song_df


def deduplicate_songs_by_id(hot100_song_df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Add stable song IDs, combine formatting-based duplicates and re-engineer popularity features."""

    unique_hot100_song_df = hot100_song_df.copy()

    unique_hot100_song_df["song_id"] = unique_hot100_song_df.apply(
        lambda row: compute_song_id(row["title"], row["performer"]),
        axis=1,
    )

    unique_hot100_song_df = unique_hot100_song_df.groupby("song_id", as_index=False).agg(
        title=("title", "first"),
        performer=("performer", "first"),
        chart_weeks=("chart_weeks", lambda s: sorted({week for weeks in s for week in weeks})),
        peak_pos=("peak_pos", "min"),
    )
    unique_hot100_song_df["wks_on_chart"] = unique_hot100_song_df["chart_weeks"].str.len()

    if save:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        unique_hot100_song_df.to_parquet(HOT100_ALLSONG_OUTPUT, index=False)
        print(f"Saved unique_hot100_song_df to {HOT100_ALLSONG_OUTPUT}.")

    print(f"{len(unique_hot100_song_df)} unique songs recorded in {HOT100_OUTPUT}.")

    return unique_hot100_song_df


def filter_hot100_song(
    hot100_song_df: pd.DataFrame,
    start_wk: str | pd.Timestamp = START_WEEK,
    end_wk: str | pd.Timestamp = END_WEEK,
    save: bool = True,
) -> pd.DataFrame:
    """Filter Billboard Hot 100 song data by chart_weeks and save the result."""

    start_wk_str = start_wk
    end_wk_str = end_wk

    if isinstance(start_wk, pd.Timestamp):
        start_wk_str = start_wk.strftime("%Y-%m-%d")
    if isinstance(end_wk, pd.Timestamp):
        end_wk_str = end_wk.strftime("%Y-%m-%d")

    filtered_hot100_song_df = hot100_song_df[
        (hot100_song_df["chart_weeks"].str[0] >= start_wk_str)  # compare with YYYY-MM-DD format str
        & (hot100_song_df["chart_weeks"].str[-1] < end_wk_str)
    ]

    if save:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        filtered_hot100_song_df.to_parquet(HOT100_SONGS_OUTPUT, index=False)
        print(f"Saved filtered_hot100_song_df to {HOT100_SONGS_OUTPUT}.")

    print(
        f"{len(filtered_hot100_song_df)} songs on Billboard Hot-100 from {start_wk_str} to {end_wk_str}."
    )
    return filtered_hot100_song_df


def main():
    hot100_chart_df = clean_hot100_chart()

    hot100_song_df = clean_hot100_song()
    unique_hot100_song_df = deduplicate_songs_by_id(hot100_song_df)

    filtered_hot100_song_df = filter_hot100_song(unique_hot100_song_df)


if __name__ == "__main__":
    main()
