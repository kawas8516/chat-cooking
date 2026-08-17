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

# Chat Cooking [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/kawas8516/chat-cooking)

A cooking assistant chatbot powered by **RAG (Retrieval-Augmented Generation)**. Ask about recipes, ingredients, or cooking techniques — it semantically retrieves relevant recipes from a dataset of 1,090 dishes and streams a grounded response via the HuggingFace Inference API.

**Live demo:** [huggingface.co/spaces/kawas8516/chat-cooking](https://huggingface.co/spaces/kawas8516/chat-cooking)

---

## How it works

```
User message
      |
      v
is_recipe_query()          keyword + word-boundary classifier
      |
   Yes |  No ──────────────────────────────────────────> LLM (general answer)
      |
      v
retrieve()                 FAISS IndexFlatIP over sentence-transformers embeddings
      |                    (all-MiniLM-L6-v2, cosine similarity, top-k results)
      v
build_messages()           system prompt + retrieved context + conversation history
      |
      v
stream_response()          HuggingFace Inference API  (Qwen2.5-7B-Instruct, streaming)
      |
      v
Gradio UI                  chat panel (left) | retrieved recipes panel (right)
```

### Key design decisions

| Decision | Rationale |
|---|---|
| FAISS `IndexFlatIP` over normalized vectors | Exact cosine search, no approximation error on a 1K corpus |
| `all-MiniLM-L6-v2` embedder | 384-dim, fast on CPU, strong semantic recall for food queries |
| Keyword classifier before retrieval | Avoids embedding cost and noisy retrieval on non-food questions |
| Retrieved context injected as a second `system` message | Keeps it clearly separated from user history for the LLM |
| Index built at startup if missing | Binary files rejected by HF git; CSV stays in repo, index is ephemeral |

---

## Evaluation

Measured on the 1,090-recipe dataset with `python eval.py`:

| Metric | Score |
|---|---|
| Classifier accuracy (14 labelled queries) | **100%** |
| Retrieval Hit@3 (12 query/recipe pairs) | **92%** |
| Retrieval MRR | **0.861** |

The one retrieval miss ("chocolate cherry dessert") is a data-coverage gap — no exact match exists in the 1,090-recipe subset. MRR of 0.861 means the correct recipe is ranked #1 on average.

Run evaluation yourself:
```bash
python eval.py                  # classifier + retrieval
python eval.py --faithfulness   # + LLM faithfulness judge (uses tokens)
python eval.py --k 5            # evaluate Hit@5 instead
python eval.py --json out.json  # save results to JSON
```

---

## Project structure

```
chat-cooking/
├── app.py              # Gradio Blocks UI
├── rag.py              # retrieval: embedder, FAISS index, is_recipe_query()
├── llm.py              # build_messages(), stream_response()
├── config.py           # env-based configuration (python-dotenv)
├── build_index.py      # build data/recipes.index from data/recipes.csv
├── eval.py             # evaluation suite (classifier + retrieval + faithfulness)
├── data/
│   └── recipes.csv     # 1,090 recipes (name, ingredients, directions, url)
├── tests/
│   ├── test_rag.py     # 12 unit tests for retrieval and classifier
│   └── test_llm.py     # 11 unit tests for message building and streaming
└── .github/workflows/
    └── ci.yml          # ruff + pytest on Python 3.11
```

---

## Quickstart (local)

```bash
git clone https://github.com/kawas8516/chat-cooking
cd chat-cooking
pip install -r requirements.txt
cp .env.example .env        # add your HF_TOKEN
python build_index.py       # embeds 1090 recipes, writes data/recipes.index (~15s)
python app.py               # open http://localhost:7860
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | **required** | HuggingFace token (Inference API access) |
| `MODEL_ID` | `Qwen/Qwen2.5-7B-Instruct` | Generation model |
| `EMBED_MODEL_ID` | `all-MiniLM-L6-v2` | Embedding model for retrieval |
| `TOP_K` | `3` | Number of recipes retrieved per query |
| `MAX_NEW_TOKENS` | `512` | Max tokens per LLM response |
| `TEMPERATURE` | `0.7` | Sampling temperature |

**Alternative models** (set `MODEL_ID` to swap with no code changes):
- `meta-llama/Llama-3.2-3B-Instruct` — lighter, faster
- `mistralai/Mistral-7B-Instruct-v0.3` — strong instruction following

---

## Deploy to Hugging Face Spaces

1. Fork this repo
2. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space) — SDK: **Gradio**
3. Push to the Space remote: `git push hf-space main`
4. Add `HF_TOKEN` under **Settings > Secrets**
5. The Space auto-builds the index on first boot — no manual step needed

---

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest          # 23 tests
ruff check .              # lint
python eval.py            # evaluation
```

CI runs on every push to `main` via GitHub Actions (`.github/workflows/ci.yml`).

---

## Stack

- **UI** — [Gradio](https://gradio.app) Blocks
- **Retrieval** — [FAISS](https://github.com/facebookresearch/faiss) + [sentence-transformers](https://www.sbert.net/)
- **Generation** — [HuggingFace Inference API](https://huggingface.co/inference-api) (streaming)
- **Data** — 1,090 recipes scraped from Allrecipes
