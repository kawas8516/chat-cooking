---
title: Chat Cooking
emoji: 🍳
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: "6.15.2"
python_version: "3.11"
app_file: app.py
pinned: false
---

# Chat Cooking

A cooking assistant chatbot powered by RAG (retrieval-augmented generation). Ask about recipes, ingredients, or cooking techniques — it retrieves relevant recipes from a database of 1,000+ dishes and generates a helpful response.

## Architecture

```
User message
     |
     v
is_recipe_query()  ----No----> LLM (general answer)
     |
    Yes
     |
     v
retrieve()  <-- FAISS index (sentence-transformers embeddings)
     |
     v
build_messages()  -- system prompt + retrieved context + history
     |
     v
stream_response()  <-- HuggingFace Inference API (streaming)
     |
     v
Gradio UI  -- chat panel (left) + retrieved recipes (right)
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | required | HuggingFace token (read access is enough) |
| `MODEL_ID` | `Qwen/Qwen2.5-7B-Instruct` | Inference API model |
| `EMBED_MODEL_ID` | `all-MiniLM-L6-v2` | Sentence-transformers model for retrieval |
| `TOP_K` | `3` | Number of recipes to retrieve |
| `MAX_NEW_TOKENS` | `512` | Max tokens per response |
| `TEMPERATURE` | `0.7` | Sampling temperature |

**Fallback models** (set `MODEL_ID` to swap):
- `meta-llama/Llama-3.2-3B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`

## Run locally

```bash
git clone https://github.com/kawas8516/chat-cooking
cd chat-cooking
pip install -r requirements.txt
cp .env.example .env   # fill in HF_TOKEN
python build_index.py  # builds data/recipes.index (~15s)
python app.py
```

Open http://localhost:7860.

## Deploy to Hugging Face Spaces

1. Create a new Space at huggingface.co/new-space, SDK = Gradio
2. Push this repo to the Space repository
3. Add `HF_TOKEN` as a Space secret (Settings > Secrets)
4. The Space will run `app.py` automatically — no build step needed (`data/recipes.index` is committed)

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest
ruff check .
```
