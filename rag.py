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
    "recipe", "cook", "cooking", "bake", "baking", "ingredient", "ingredients",
    "how to make", "dish", "meal", "cuisine", "dinner", "lunch", "breakfast",
    "snack", "dessert", "grill", "roast", "fry", "boil", "steam",
    "hungry", "yummy",
}

# Bare "make"/"eat"/"food"/"prepare" are too generic on their own (false positives
# like "make sense", "eat my words") — only treat them as recipe signals when they
# show up in a food-shaped sentence pattern.
RECIPE_PHRASE_PATTERNS = [
    r"\bwhat\b.{0,20}\b(should|can|could)\b.{0,15}\b(i|we)\b.{0,15}\b(eat|cook|have|prepare)\b",
    # "make" alone is too generic ("what should we make of this") — only a
    # recipe signal when paired with "with" ("what can I make with eggs...").
    r"\bwhat\b.{0,20}\b(can|could)\b.{0,15}\b(i|we)\b.{0,15}\bmake\b.{0,10}\bwith\b",
    r"\b(something|anything)\b.{0,15}\bto\b.{0,15}\b(eat|cook|make|prepare)\b",
    r"\bwhip(?:ping)?\s+up\b",
    r"\bgoes\s+(?:well\s+)?with\b",
    r"\bsuggest\b.{0,20}\b(something|a recipe|a dish|a meal)\b",
    r"\bwhat'?s\s+for\s+(dinner|lunch|breakfast)\b",
    r"\b(meal|dinner|lunch|breakfast)\s+ideas?\b",
    r"\bmake\b.{0,15}\b(dinner|lunch|breakfast|meal|dish|recipe)\b",
    r"\bprepare\b.{0,15}\b(meal|dinner|lunch|breakfast|dish|food)\b",
]

# Idioms that use recipe-adjacent verbs non-literally — checked first, force False.
NEGATIVE_PHRASE_PATTERNS = [
    r"\bmakes?\s+sense\b",
    r"\bmake\s+(a\s+)?decision\b",
    r"\bmake\s+friends\b",
    r"\bmake\s+(my|your|his|her|their|our)\s+(day|code|essay|paragraph|point|case)\b",
    r"\beat\s+(my|your|his|her|their|our)\s+words\b",
    r"\bfood\s+for\s+thought\b",
    r"\bprepare\s+(a\s+)?presentation\b",
]


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
    if any(re.search(p, lowered) for p in NEGATIVE_PHRASE_PATTERNS):
        return False
    if any(re.search(r"\b" + re.escape(kw) + r"\b", lowered) for kw in RECIPE_KEYWORDS):
        return True
    return any(re.search(p, lowered) for p in RECIPE_PHRASE_PATTERNS)


def build_retrieval_query(message: str, history: list[dict]) -> str:
    """Prepend the last user turn so follow-ups retrieve in context."""
    last_user = next(
        (t["content"] for t in reversed(history) if t.get("role") == "user"),
        None,
    )
    if last_user:
        return f"{last_user} {message}"
    return message


def retrieve(
    query: str,
    top_k: int | None = None,
    min_score: float | None = None,
) -> list[dict]:
    if top_k is None:
        top_k = config.TOP_K
    if min_score is None:
        min_score = config.SCORE_THRESHOLD

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
        if float(score) < min_score:
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
