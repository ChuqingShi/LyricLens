import numpy as np
from tqdm.auto import tqdm
from .embedder import Embedder
from src.config import BATCH_SIZE, EMBEDDINGS_OUTPUT


def embed_texts(texts: list[str], embedder: Embedder) -> list[np.ndarray]:
    """Embed a list of texts in batches"""

    vectors = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch = texts[i : i + BATCH_SIZE]
        batch_vectors = embedder.encode_batch(batch)
        vectors.extend(batch_vectors)

    return vectors


def save_embeddings(vectors: list[np.ndarray]):
    """Stack embeddings into a NumPy array and save it in .npy"""
    vectors_array = np.asarray(vectors, dtype=np.float32)
    print(f"embeddings shape: {vectors_array.shape}")  # (num_documents, 384)

    np.save(EMBEDDINGS_OUTPUT, vectors_array)
    print(f"Saved embeddings to {EMBEDDINGS_OUTPUT}.")


def main():
    import json
    from src.embeddings.embedder import Embedder
    from src.config import MODEL_PATH, HOT100_CHUNKS_OUTPUT

    embedder = Embedder(MODEL_PATH)

    with open(HOT100_CHUNKS_OUTPUT, "r", encoding="utf-8") as f:
        documents = json.load(f)

    texts = [doc["section"] for doc in documents]
    vectors = embed_texts(texts, embedder)
    save_embeddings(vectors)


if __name__ == "__main__":
    main()
