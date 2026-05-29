"""RAG retrieval module with module-level cached singletons."""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

import config

_embedder: SentenceTransformer | None = None
_index: faiss.Index | None = None
_df: pd.DataFrame | None = None

RECIPE_KEYWORDS = {
    "recipe", "cook", "bake", "ingredient", "ingredients",
    "make", "food", "eat", "dish", "meal", "prepare", "how to make",
    "cuisine", "dinner", "lunch", "breakfast", "snack", "dessert",
    "grill", "roast", "fry", "boil", "steam",
}


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(config.EMBED_MODEL_ID)
    return _embedder


def get_index() -> faiss.Index:
    global _index
    if _index is None:
        _index = faiss.read_index(config.INDEX_PATH)
    return _index


def get_recipes_df() -> pd.DataFrame:
    global _df
    if _df is None:
        _df = (pd.read_csv(config.CSV_PATH)
               .dropna(subset=["recipe_name"])
               .drop_duplicates(subset=["recipe_name"])
               .reset_index(drop=True))
    return _df


def is_recipe_query(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(r"\b" + re.escape(kw) + r"\b", lowered) for kw in RECIPE_KEYWORDS)


def retrieve(query: str, top_k: int | None = None) -> list[dict]:
    if top_k is None:
        top_k = config.TOP_K

    index = get_index()
    df = get_recipes_df()

    if index.ntotal == 0:
        return []

    embedder = get_embedder()
    vec = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    scores, indices = index.search(vec, min(top_k, index.ntotal))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(df):
            continue
        row = df.iloc[idx]
        results.append({
            "name": str(row.get("recipe_name", "")),
            "ingredients": str(row.get("ingredients", "")),
            "directions": str(row.get("directions", "")),
            "url": str(row.get("url", "")),
            "cuisine": str(row.get("cuisine_path", "")),
            "score": float(score),
        })
    return results
