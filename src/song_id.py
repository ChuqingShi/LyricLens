import unicodedata
import uuid

# Fixed namespace for stable LyricLens song IDs; changing it regenerates all IDs.
_NAMESPACE = uuid.UUID("2f1a9c3e-6b3f-4b8e-9b0a-7a2d4e9c1f5a")


def _normalize(value: str) -> str:
    """Fold whitespace/case/unicode differences so trivial formatting variations don't produce a different id for the same song."""

    return unicodedata.normalize("NFKC", value).strip().casefold()


def compute_song_id(title: str, performer: str) -> str:
    """Deterministic song id, derived only from (title, performer).
    Unlike DataFrame row index, this stays the same across re-ingestion as a persisted identity key.
    """

    key = f"{_normalize(title)}|{_normalize(performer)}"
    return str(uuid.uuid5(_NAMESPACE, key))


def main():
    a = compute_song_id("Blinding Lights", "The Weeknd")
    b = compute_song_id("Blinding Lights", "The Weeknd")
    c = compute_song_id("  Blinding Lights  ", "THE WEEKND")
    d = compute_song_id("Blinding Lights", "Doja Cat")
    e = compute_song_id("Blinding", "Lights The Weeknd")
    f = compute_song_id("BlindingLights", "The Weeknd")

    assert a == b, "same input should give the same id"
    assert a == c, "normalization should fold whitespace/case differences"
    assert a != d, "different performer should give a different id"
    assert a != e, "mixing title and performer should give a different id"
    assert a != f, "spacing within title/ performer differentiates song id"

    print(f"'Blinding Lights' by 'The Weeknd' -> {a}")
    print(f"'  Blinding Lights  ' by 'THE WEEKND' -> {c} (matches: {a == c})")
    print(f"'Blinding Lights' by 'Doja Cat' -> {d} (differs: {a != d})")
    print(f"'Blinding' by 'Lights The Weekend' -> {e} (differs: {a!= e})")
    print(f"'BlindingLights' by 'The Weeknd' -> {f} (differs: {a!= f})")
    print("All checks passed.")


if __name__ == "__main__":
    main()
