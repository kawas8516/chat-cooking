import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# ✅ Load the dataset
df = pd.read_csv("recipes.csv")

# ✅ Keep only useful columns
useful_columns = ["recipe_name", "prep_time", "cook_time", "total_time", "servings", "yield", 
                  "ingredients", "directions", "url", "cuisine_path", "nutrition", "timing"]

df = df[useful_columns]  # Select only the required columns

# ✅ Combine text for embedding
df["combined_text"] = df.apply(lambda row: ". ".join([f"{col}: {row[col]}" for col in useful_columns if pd.notna(row[col])]), axis=1)

# ✅ Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # A lightweight but powerful embedding model

# ✅ Generate embeddings
embeddings = embedding_model.encode(df["combined_text"].tolist(), convert_to_numpy=True)

# ✅ Save embeddings with FAISS for efficient retrieval
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# ✅ Save FAISS index
faiss.write_index(index, "recipes_faiss.index")

# ✅ Save dataframe for retrieval
df.to_csv("cleaned_recipes.csv", index=False)

print("Embeddings generated and saved successfully!")
