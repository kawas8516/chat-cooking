import numpy as np
import pandas as pd
import faiss
import pytest

import rag


FAKE_RECIPES = [
    {"recipe_name": "Pasta Carbonara", "ingredients": "pasta, eggs, bacon", "directions": "boil pasta", "url": "http://a.com", "cuisine_path": "Italian"},
    {"recipe_name": "Caesar Salad", "ingredients": "romaine, croutons, dressing", "directions": "toss salad", "url": "http://b.com", "cuisine_path": "American"},
    {"recipe_name": "Chicken Curry", "ingredients": "chicken, curry paste, coconut milk", "directions": "simmer", "url": "http://c.com", "cuisine_path": "Indian"},
    {"recipe_name": "Chocolate Cake", "ingredients": "flour, cocoa, sugar, butter", "directions": "bake", "url": "http://d.com", "cuisine_path": "American"},
    {"recipe_name": "Sushi Rolls", "ingredients": "rice, nori, fish, avocado", "directions": "roll", "url": "http://e.com", "cuisine_path": "Japanese"},
]
DIM = 4


@pytest.fixture
def synthetic_index_and_df(tmp_path, monkeypatch):
    df = pd.DataFrame(FAKE_RECIPES)
    embeddings = np.random.rand(len(df), DIM).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings /= norms

    index = faiss.IndexFlatIP(DIM)
    index.add(embeddings)

    index_path = str(tmp_path / "test.index")
    csv_path = str(tmp_path / "test.csv")
    faiss.write_index(index, index_path)
    df.to_csv(csv_path, index=False)

    monkeypatch.setattr("config.INDEX_PATH", index_path)
    monkeypatch.setattr("config.CSV_PATH", csv_path)
    monkeypatch.setattr("config.TOP_K", 3)
    monkeypatch.setattr("config.EMBED_MODEL_ID", "all-MiniLM-L6-v2")

    # Reset singletons so the fixture paths are used
    monkeypatch.setattr(rag, "_index", None)
    monkeypatch.setattr(rag, "_df", None)

    # Stub the embedder to return deterministic vectors
    class FakeEmbedder:
        def encode(self, texts, **kwargs):
            vecs = np.random.rand(len(texts), DIM).astype(np.float32)
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs

    monkeypatch.setattr(rag, "_embedder", FakeEmbedder())

    return index, df


class TestIsRecipeQuery:
    def test_positive_bake(self):
        assert rag.is_recipe_query("how do I bake bread") is True

    def test_positive_recipe(self):
        assert rag.is_recipe_query("give me a recipe for soup") is True

    def test_positive_ingredient(self):
        assert rag.is_recipe_query("what can I make with these ingredients") is True

    def test_positive_food(self):
        assert rag.is_recipe_query("what food should I eat tonight") is True

    def test_positive_cook(self):
        assert rag.is_recipe_query("teach me to cook pasta") is True

    def test_negative_weather(self):
        assert rag.is_recipe_query("what's the weather like today") is False

    def test_negative_general(self):
        assert rag.is_recipe_query("tell me a joke") is False

    def test_negative_math(self):
        assert rag.is_recipe_query("what is two plus two") is False

    def test_positive_hungry_paraphrase(self):
        assert rag.is_recipe_query("I'm hungry, suggest something") is True

    def test_positive_goes_well_with(self):
        assert rag.is_recipe_query("What goes well with chicken?") is True

    def test_positive_whip_up_phrase(self):
        assert rag.is_recipe_query("Need something to whip up fast") is True

    def test_positive_what_should_i_eat(self):
        assert rag.is_recipe_query("What should I eat tonight?") is True

    def test_regression_make_with_ingredients_still_positive(self):
        assert rag.is_recipe_query("What can I make with eggs and cheese?") is True

    def test_positive_how_do_i_make(self):
        assert rag.is_recipe_query("How do I make mulligatawny soup?") is True

    def test_negative_make_sense_idiom(self):
        assert rag.is_recipe_query("That decision doesn't make sense to me") is False

    def test_negative_make_friends_idiom(self):
        assert rag.is_recipe_query("How do I make friends in a new city?") is False

    def test_negative_eat_my_words_idiom(self):
        assert rag.is_recipe_query("I'll eat my words if I'm wrong") is False

    def test_negative_food_for_thought_idiom(self):
        assert rag.is_recipe_query("Food for thought: what do you think?") is False

    def test_negative_make_paragraph_idiom(self):
        assert rag.is_recipe_query("Can you make this paragraph shorter?") is False


class TestBuildRetrievalQuery:
    def test_empty_history_returns_message(self):
        assert rag.build_retrieval_query("vegan version", []) == "vegan version"

    def test_prepends_last_user_turn(self):
        history = [
            {"role": "user", "content": "how do I make carbonara"},
            {"role": "assistant", "content": "Here is a recipe..."},
        ]
        query = rag.build_retrieval_query("make it vegetarian", history)
        assert query == "how do I make carbonara make it vegetarian"

    def test_ignores_assistant_only_history(self):
        history = [{"role": "assistant", "content": "hello"}]
        assert rag.build_retrieval_query("soup", history) == "soup"


class TestRetrieve:
    def test_returns_top_k(self, synthetic_index_and_df):
        results = rag.retrieve("pasta recipe", top_k=3)
        assert len(results) == 3

    def test_result_structure(self, synthetic_index_and_df):
        results = rag.retrieve("chicken curry")
        assert len(results) > 0
        r = results[0]
        assert "name" in r
        assert "ingredients" in r
        assert "directions" in r
        assert "url" in r
        assert "cuisine" in r
        assert "score" in r

    def test_returns_fewer_when_index_small(self, synthetic_index_and_df):
        results = rag.retrieve("salad", top_k=10)
        # Only 5 rows in synthetic data
        assert len(results) == 5

    def test_threshold_filters_all(self, synthetic_index_and_df):
        # Cosine scores can't exceed 1.0, so a threshold above 1 drops everything.
        results = rag.retrieve("pasta recipe", top_k=3, min_score=1.1)
        assert results == []

    def test_threshold_zero_keeps_all(self, synthetic_index_and_df):
        results = rag.retrieve("pasta recipe", top_k=3, min_score=-1.0)
        assert len(results) == 3

    def test_empty_index(self, tmp_path, monkeypatch):
        dim = 4
        index = faiss.IndexFlatIP(dim)
        index_path = str(tmp_path / "empty.index")
        faiss.write_index(index, index_path)

        df = pd.DataFrame(FAKE_RECIPES)
        csv_path = str(tmp_path / "test.csv")
        df.to_csv(csv_path, index=False)

        monkeypatch.setattr("config.INDEX_PATH", index_path)
        monkeypatch.setattr("config.CSV_PATH", csv_path)
        monkeypatch.setattr(rag, "_index", None)
        monkeypatch.setattr(rag, "_df", None)

        results = rag.retrieve("anything")
        assert results == []
