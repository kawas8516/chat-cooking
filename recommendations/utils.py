import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from recipes.models import Recipe

def recommend_recipes(user_ingredients, top_n=5):
    recipes = Recipe.objects.all()
    df = pd.DataFrame(list(recipes.values("id", "ingredients")))

    # Convert ingredient lists to text
    df["ingredients"] = df["ingredients"].apply(lambda x: " ".join(x))

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["ingredients"])

    # Convert user ingredients into a vector
    user_input_vector = vectorizer.transform([" ".join(user_ingredients)])

    # Compute similarity
    similarity_scores = cosine_similarity(user_input_vector, tfidf_matrix).flatten()
    df["similarity"] = similarity_scores

    # Get top matching recipes
    recommended_recipes = df.sort_values(by="similarity", ascending=False).head(top_n)

    return Recipe.objects.filter(id__in=recommended_recipes["id"].tolist())
