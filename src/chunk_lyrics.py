import pandas as pd
import re

LYRICS_SECTION_SEPARATOR_PATTERN = r"\n\s*\n"  # newline + >=0 whitespace + newline
MIN_LINES_PER_CHUNK = 4
DEFAULT_LINES_PER_CHUNK = int(1.5 * MIN_LINES_PER_CHUNK)


def chunk_song_lyrics(lyrics: str) -> tuple[list[str], list[int]]:
    """The plain_lyrics usually has natural sections marked by LYRICS_SECTION_SEPARATOR_PATTERN.
    Chunk a given song's lyrics into natural sections, and count number of lines per section."""
    sections = [
        section.strip()
        for section in re.split(LYRICS_SECTION_SEPARATOR_PATTERN, lyrics)
        if section.strip()
    ]

    num_lines = [1 + section.count("\n") for section in sections]

    # # error handling: if lyrics is not in ready-to-chunk format
    # if len(sections) == 1:  # never 0 guaranteed by split
    #     raise ValueError("Could not split lyrics into multiple sections.")
    return (sections, num_lines)


def force_chunk_song_lyrics(
    lyrics: str, lines_per_chunk: int = DEFAULT_LINES_PER_CHUNK
) -> tuple[list[str], list[int]]:
    """Fallback for lyrics without predefined section separators.
    Force-chunk lyrics into sections of `lines_per_chunk` lines each.
    """
    lines = [line.strip() for line in lyrics.splitlines() if line.strip()]

    if len(lines) <= lines_per_chunk:
        print("Lyrics are too short to chunk.")
        return ["\n".join(lines)], [len(lines)]

    else:
        sections = []
        num_lines = []
        for i in range(0, len(lines), lines_per_chunk):
            chunk = lines[i : i + lines_per_chunk]
            sections.append("\n".join(chunk))
            num_lines.append(len(chunk))

        # # error handling: if lyrics is not in ready-to-chunk format
        # if len(sections) == 1 and len(lines) > lines_per_chunk:  # never 0 guaranteed by splitlines
        #     raise ValueError(
        #         "All lyrics are force-chunked into one section. Please set lines_per_chunk smaller."
        #     )
        return sections, num_lines


def chunk_song_lyrics_with_fallback(
    lyrics: str, lines_per_chunk: int = DEFAULT_LINES_PER_CHUNK
) -> tuple[list[str], list[int]]:
    sections, num_lines = chunk_song_lyrics(lyrics)

    if len(sections) == 1:  # never 0 guaranteed by chunk_song_lyrics
        return force_chunk_song_lyrics(lyrics, lines_per_chunk)

    return sections, num_lines


def merge_short_sections(
    sections: list[str], num_lines: list[int], min_lines: int = MIN_LINES_PER_CHUNK
) -> tuple[list[str], list[int]]:
    """If the first few sections are too short, concatenate them together as one;
    If any latter section is short, append it to the previous one."""
    while num_lines[0] < min_lines:  # len(list) > 1 is guaranteed by inputs for the first run
        short_section = sections.pop(0)
        num_short_lines = num_lines.pop(0)
        sections[0] = short_section + "\n\n" + sections[0]
        num_lines[0] = num_short_lines + num_lines[0]

        if len(sections) == 1:  # never 0
            raise ValueError(
                "All lyrics were merged into one section. Please set min_lines smaller."
            )

    section_id = 1  # len(list) > 1 is guaranteed by error handling for the first run
    while section_id < len(sections):
        while (
            num_lines[section_id] < min_lines
        ):  # if section_id == 0, num_lines[section_id] >= min_lines is guaranteed by earlier discussion
            short_section = sections.pop(section_id)
            num_short_lines = num_lines.pop(section_id)
            section_id = section_id - 1
            sections[section_id] = sections[section_id] + "\n\n" + short_section
            num_lines[section_id] = num_lines[section_id] + num_short_lines

        section_id = section_id + 1

    return (sections, num_lines)


def build_chunk_documents(lyrics_df: pd.DataFrame) -> list[dict]:
    documents = []

    count = 0
    for song_id, row in lyrics_df.iterrows():

        try:
            sections, num_lines = chunk_song_lyrics_with_fallback(row["plain_lyrics"])
            if len(sections) > 1:  # never 0 guaranteed by chunk_song_lyrics_with_fallback
                sections, num_lines = merge_short_sections(sections, num_lines)
        except IndexError:
            print(count)
            print(row["title"])
            print(row["performer"])
            count = count + 1

        num_sections = len(sections)  # coule be 1

        for section_id in range(num_sections):
            documents.append(
                {
                    "song_id": song_id,
                    "title": row["title"],
                    "performer": row["performer"],
                    "section_id": f"{section_id}/{num_sections}",
                    "section": sections[section_id],
                    "num_lines": num_lines[section_id],
                    # "wks_on_chart": row["wks_on_chart"],
                    # "peak_pos": row["peak_pos"],
                }
            )

    return documents


if __name__ == "__main__":
    from src.config import HOT100_LYRICS_OUTPUT

    filtered_hot100_lyrics_df = pd.read_parquet(HOT100_LYRICS_OUTPUT)
    documents = build_chunk_documents(filtered_hot100_lyrics_df)
    print(f"Chunking {len(filtered_hot100_lyrics_df)} songe into {len(documents)} lyrics sections.")
