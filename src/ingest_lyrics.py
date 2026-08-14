import pandas as pd
from tqdm.auto import tqdm
import numpy as np

from .embeddings.embedder import Embedder
from .config import HOT100_LYRICS_OUTPUT
from .chunk_lyrics import build_chunk_documents

BATCH_SIZE = 50


def load_lyrics() -> pd.DataFrame:
    lyrics_df = pd.read_parquet(HOT100_LYRICS_OUTPUT)
    return lyrics_df


def load_chunk_data() -> list[dict]:
    lyrics_df = load_lyrics()
    documents = build_chunk_documents(lyrics_df)
    return documents


def embed_chunk(documents: list[dict], embedder: Embedder) -> np.array:
    texts = [doc["section"] for doc in documents]

    X = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch = texts[i : i + BATCH_SIZE]
        batch_vectors = embedder.encode_batch(batch)
        X.extend(batch_vectors)
    X = np.array(X)

    return X


if __name__ == "__main__":
    from src.embeddings.download import download

    download("Xenova/all-MiniLM-L6-v2")

    documents = load_chunk_data()
    embedder = Embedder("models/Xenova/all-MiniLM-L6-v2")

    X = embed_chunk(documents, embedder)

    query = "Grief after losing best friend"
    v_query = embedder.encode(query)

    scores = X.dot(v_query)
    top5 = np.argsort(scores)[-5:][::-1]

    for i in top5:
        print(scores[i])
        print(documents[i])
