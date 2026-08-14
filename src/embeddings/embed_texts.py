import numpy as np
from tqdm.auto import tqdm
from .embedder import Embedder
from src.config import BATCH_SIZE


def embed_texts(texts: list[str], embedder: Embedder) -> np.ndarray:

    vectors = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch = texts[i : i + BATCH_SIZE]
        batch_vectors = embedder.encode_batch(batch)
        vectors.extend(batch_vectors)
    X = np.array(vectors)

    return X


if __name__ == "__main__":
    pass
