"""LLM streaming module using HuggingFace Inference API."""
from __future__ import annotations

from typing import Generator

from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

import config

_client: InferenceClient | None = None

SYSTEM_PROMPT = (
    "You are a cooking assistant. Use the retrieved recipes below when relevant. "
    "Be concise. Don't invent recipes."
)


def _get_client() -> InferenceClient:
    global _client
    if _client is None:
        _client = InferenceClient(
            model=config.MODEL_ID,
            token=config.HF_TOKEN or None,
        )
    return _client


def _format_retrieved(retrieved: list[dict]) -> str:
    if not retrieved:
        return ""
    lines = []
    for i, r in enumerate(retrieved, 1):
        lines.append(f"{i}. {r['name']} ({r['cuisine']})")
        lines.append(f"   Ingredients: {r['ingredients'][:200]}")
        if r.get("url"):
            lines.append(f"   URL: {r['url']}")
    return "\n".join(lines)


def build_messages(
    user_input: str,
    retrieved: list[dict],
    history: list[dict],
) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if retrieved:
        context = _format_retrieved(retrieved)
        messages.append({
            "role": "system",
            "content": f"Retrieved recipes:\n{context}",
        })

    for turn in history:
        if turn.get("role") in ("user", "assistant"):
            messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content": user_input})
    return messages


def stream_response(messages: list[dict]) -> Generator[str, None, None]:
    try:
        client = _get_client()
        stream = client.chat_completion(
            messages=messages,
            max_tokens=config.MAX_NEW_TOKENS,
            temperature=config.TEMPERATURE,
            stream=True,
        )
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield token
    except HfHubHTTPError as e:
        status = getattr(e, "response", None)
        code = status.status_code if status is not None else "?"
        if code == 429:
            yield "Sorry, the model is rate-limited right now. Please try again in a moment."
        elif code == 503:
            yield "The model is still loading on the server. Please wait a few seconds and retry."
        else:
            yield f"API error ({code}): {e}"
    except Exception as e:
        yield f"Unexpected error: {e}"
