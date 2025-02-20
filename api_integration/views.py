import os
import torch
import faiss
import pandas as pd
from django.http import JsonResponse
from rest_framework.decorators import api_view
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# ✅ Load Model & Tokenizer
MODEL_PATH = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" 
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

# ✅ Load FAISS index & dataset
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Use absolute paths
BASE_DIR = os.path.dirname(__file__)
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "database.index")
CSV_PATH = os.path.join(BASE_DIR, "cleaned_recipes.csv")

# ✅ Ensure FAISS index & dataset exist
if not os.path.exists(FAISS_INDEX_PATH):
    raise FileNotFoundError(f"FAISS index file not found at: {FAISS_INDEX_PATH}")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Dataset file not found at: {CSV_PATH}")

# ✅ Load FAISS index & dataset
faiss_index = faiss.read_index(FAISS_INDEX_PATH)
df = pd.read_csv(CSV_PATH)

# ✅ Define keywords to detect recipe queries
RECIPE_KEYWORDS = ["recipe", "ingredients", "cook", "bake", "prepare", "dish", "meal", "kitchen"]

def is_recipe_query(query):
    """
    Detect if a query is related to recipes.
    """
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in RECIPE_KEYWORDS)

def retrieve_similar_recipes(query, top_k=3):
    """
    Retrieve top_k most relevant recipes based on the query.
    """
    query_embedding = embedding_model.encode([query])

    if faiss_index.ntotal == 0:
        return ""  # Handle empty FAISS index gracefully

    _, indices = faiss_index.search(query_embedding, top_k)

    retrieved_texts = []
    for idx in indices[0]:
        if 0 <= idx < len(df):  
            retrieved_texts.append(df.iloc[idx]["combined_text"])  

    return " ".join(retrieved_texts) if retrieved_texts else ""

@api_view(["POST"])
def chatbot_response(request):
    """
    API to generate chatbot response using RAG for recipes, and normal LLM for general queries.
    """
    user_input = request.data.get("text", "").strip()

    if not user_input:
        return JsonResponse({"error": "No input text provided"}, status=400)

    if is_recipe_query(user_input):
        # ✅ Use RAG for recipe-related queries
        retrieved_context = retrieve_similar_recipes(user_input)
        if not retrieved_context:
            full_prompt = f"User: {user_input}\n\nBot: Sorry, I couldn't find a matching recipe."
        else:
            full_prompt = f"User: {user_input}\n\nRelevant Recipes: {retrieved_context}\n\nBot:"
    else:
        # ✅ Use normal LLM for general conversations
        full_prompt = f"User: {user_input}\n\nBot:"

    # ✅ Tokenize input
    inputs = tokenizer(full_prompt, return_tensors="pt")

    # ✅ Generate response (limit max tokens to prevent excessive responses)
    output = model.generate(**inputs, max_length=512)  
    response_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return JsonResponse({"response": response_text})
