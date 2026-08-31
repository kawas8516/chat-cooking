"""LLM streaming module using HuggingFace Inference API."""
from __future__ import annotations

import time
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
        directions = r.get("directions", "").strip()
        if directions:
            lines.append(f"   Directions: {directions[:400]}")
        if r.get("url"):
            lines.append(f"   URL: {r['url']}")
    return "\n".join(lines)


def _truncate_history(history: list[dict]) -> list[dict]:
    """Keep only the most recent MAX_HISTORY_TURNS turns (1 turn = user+assistant
    pair = 2 entries) so long conversations don't overflow the model's context."""
    max_messages = config.MAX_HISTORY_TURNS * 2
    return history[-max_messages:] if max_messages > 0 else []


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

    for turn in _truncate_history(history):
        if turn.get("role") in ("user", "assistant"):
            messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content": user_input})
    return messages


def stream_response(messages: list[dict]) -> Generator[str, None, None]:
    client = _get_client()
    attempt = 0
    yielded_any = False
    while True:
        try:
            stream = client.chat_completion(
                messages=messages,
                max_tokens=config.MAX_NEW_TOKENS,
                temperature=config.TEMPERATURE,
                stream=True,
            )
            for chunk in stream:
                # Some providers send a trailing chunk (e.g. usage stats) with
                # an empty choices list to mark end-of-stream.
                if not chunk.choices:
                    continue
                token = chunk.choices[0].delta.content
                if token:
                    yielded_any = True
                    yield token
            return
        except HfHubHTTPError as e:
            status = getattr(e, "response", None)
            code = status.status_code if status is not None else "?"
            # Only retry a transient error before any output has been streamed —
            # once tokens are out, retrying would resend the whole prompt and
            # produce duplicate/garbled output.
            if code in (429, 503) and not yielded_any and attempt < config.LLM_MAX_RETRIES:
                attempt += 1
                delay = min(
                    config.LLM_RETRY_BACKOFF_BASE * (2 ** (attempt - 1)),
                    config.LLM_RETRY_BACKOFF_MAX,
                )
                time.sleep(delay)
                continue
            if code == 429:
                yield "Sorry, the model is rate-limited right now. Please try again in a moment."
            elif code == 503:
                yield "The model is still loading on the server. Please wait a few seconds and retry."
            else:
                yield f"API error ({code}): {e}"
            return
        except Exception as e:
            yield f"Unexpected error: {e}"
            return
