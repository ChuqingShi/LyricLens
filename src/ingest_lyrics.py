import pandas as pd
import numpy as np

from .embeddings.embedder import Embedder
from .embeddings.embed_texts import embed_texts
from .config import HOT100_LYRICS_OUTPUT


def load_lyrics() -> pd.DataFrame:
    return pd.read_parquet(HOT100_LYRICS_OUTPUT)


if __name__ == "__main__":
    from src.embeddings.download import download
    from .chunk_lyrics import build_chunk_documents

    download("Xenova/all-MiniLM-L6-v2")
    embedder = Embedder("models/Xenova/all-MiniLM-L6-v2")

    lyrics_df = load_lyrics()
    documents = build_chunk_documents(lyrics_df)

    texts = [doc["section"] for doc in documents]
    X = embed_texts(documents, embedder)

    query = "Grief after losing best friend"
    v_query = embedder.encode(query)

    scores = X.dot(v_query)
    top5 = np.argsort(scores)[-5:][::-1]

    for i in top5:
        print(scores[i])
        print(documents[i])
