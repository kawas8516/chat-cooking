"""Build FAISS index from recipes.csv. Idempotent — safe to re-run."""
import os
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

from config import CSV_PATH, INDEX_PATH, EMBED_MODEL_ID, DATA_DIR

BATCH_SIZE = 256


def build_combined_text(row: pd.Series) -> str:
    name = str(row.get("recipe_name", "")).strip()
    ingredients = str(row.get("ingredients", "")).strip()
    cuisine = str(row.get("cuisine_path", "")).strip()
    return f"{name}. Cuisine: {cuisine}. Ingredients: {ingredients}"


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Loading {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)
    before = len(df)
    df = df.dropna(subset=["recipe_name"])
    df = df.drop_duplicates(subset=["recipe_name"])
    df = df.reset_index(drop=True)
    print(f"  {before} rows -> {len(df)} after dropping missing names and duplicates")

    df["combined_text"] = df.apply(build_combined_text, axis=1)
    texts = df["combined_text"].tolist()

    print(f"Loading embedder '{EMBED_MODEL_ID}' ...")
    model = SentenceTransformer(EMBED_MODEL_ID)

    print(f"Embedding {len(texts)} recipes in batches of {BATCH_SIZE} ...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = embeddings.astype(np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    print(f"Index written to {INDEX_PATH} ({index.ntotal} vectors, dim={dim})")


if __name__ == "__main__":
    main()
