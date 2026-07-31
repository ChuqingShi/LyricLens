from pathlib import Path
import pandas as pd
from download_hot100 import HOT100_OUTPUT

OUTPUT_DIR = Path("data/processed")
START_WEEK = pd.Timestamp("1958-08-04")
END_WEEK = pd.Timestamp("2026-08-01")


def clean_hot100_chart() -> pd.DataFrame:
    hot100_chart_df = pd.read_csv(HOT100_OUTPUT)

    hot100_chart_df["chart_week"] = pd.to_datetime(hot100_chart_df["chart_week"])
    hot100_chart_df = hot100_chart_df.sort_values(["chart_week", "current_week"])

    hot100_chart_df["last_week"] = hot100_chart_df["last_week"].replace(0, pd.NA).astype("Int64")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hot100_chart_df.to_parquet(OUTPUT_DIR / "hot-100-chart_current.parquet", index=False)

    return hot100_chart_df


def clean_hot100_song() -> pd.DataFrame:
    hot100_chart_df = pd.read_csv(HOT100_OUTPUT)
    hot100_chart_df["chart_week"] = pd.to_datetime(hot100_chart_df["chart_week"])

    hot100_song_df = hot100_chart_df.drop(columns=["current_week", "last_week"])
    hot100_song_df = hot100_song_df[
        ["title", "performer", "chart_week", "wks_on_chart", "peak_pos"]
    ]
    hot100_song_df = hot100_song_df.sort_values(
        ["title", "performer", "chart_week", "wks_on_chart"]
    )

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
            chart_weeks=("chart_week", list),  # ordered from earlier sort_values
            wks_on_chart=("wks_on_chart", "max"),
            peak_pos=("peak_pos", "min"),
        )
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hot100_song_df.to_parquet(OUTPUT_DIR / "hot-100-song_current.parquet", index=False)

    return hot100_song_df


def filter_hot100_song(
    hot100_song_df: pd.DataFrame,
    start_wk: pd.Timestamp = START_WEEK,
    end_wk: pd.Timestamp = END_WEEK,
) -> pd.DataFrame:
    filtered_hot100_song_df = hot100_song_df[
        (hot100_song_df["chart_weeks"].str[0] >= start_wk)
        & (hot100_song_df["chart_weeks"].str[-1] < end_wk)
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filtered_hot100_song_df.to_parquet(
        OUTPUT_DIR / "hot-100-song_{start_wk:%Y%m%d}_{end_wk:%Y%m%d}.parquet",
        index=False,
    )

    return filtered_hot100_song_df


def main():
    hot100_chart_df = clean_hot100_chart()
    print(f"{len(hot100_chart_df)} entries loaded from {HOT100_OUTPUT}.")

    hot100_song_df = clean_hot100_song()
    print(f"{len(hot100_song_df)} songs recorded in {HOT100_OUTPUT}.")

    start_wk = pd.Timestamp("2000-01-01")
    end_wk = pd.Timestamp("2026-08-01")
    filtered_hot100_song_df = filter_hot100_song(hot100_song_df, start_wk, end_wk)
    print(
        f"{len(filtered_hot100_song_df)} songs on Billboard Hot-100 from {start_wk:%Y%m%d} to {end_wk:%Y%m%d}."
    )


if __name__ == "__main__":
    main()
