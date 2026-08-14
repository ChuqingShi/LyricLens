import numpy as np
from tqdm.auto import tqdm
from .embedder import Embedder
from src.config import BATCH_SIZE


def embed_texts(texts: list[str], embedder: Embedder) -> list[np.ndarray]:

    vectors = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch = texts[i : i + BATCH_SIZE]
        batch_vectors = embedder.encode_batch(batch)
        vectors.extend(batch_vectors)

    return vectors


if __name__ == "__main__":
    from src.embeddings.download import download
    from src.config import MODEL_NAME, MODEL_PATH

    download(MODEL_NAME)
    embedder = Embedder(MODEL_PATH)

    texts = [
        "Grief after losing best friend",
        "Feeling excited about a new relationship",
        "Missing someone who moved away",
    ]

    vectors = embed_texts(texts, embedder)

    print(f"Number of vectors: {len(vectors)}")
    print(f"Embedding shape: {vectors[0].shape}")
