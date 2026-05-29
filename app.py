"""Gradio Blocks app for the cooking chatbot."""
import os
import gradio as gr

import config
from llm import build_messages, stream_response
from rag import is_recipe_query, retrieve

# Build index on first boot if it doesn't exist (HF Spaces doesn't store binaries in git)
if not os.path.exists(config.INDEX_PATH):
    print("Index not found — building now...")
    import build_index
    build_index.main()

EXAMPLE_PROMPTS = [
    "How do I make pasta carbonara?",
    "Give me a chicken recipe under 30 minutes",
    "What can I cook with tomatoes, cheese and basil?",
    "Suggest a vegetarian dinner for two",
    "What's a good dessert recipe with chocolate?",
]


def _recipes_markdown(recipes: list[dict]) -> str:
    if not recipes:
        return "_No recipes retrieved — this was a general question._"
    lines = ["### Retrieved recipes\n"]
    for i, r in enumerate(recipes, 1):
        name = r["name"]
        cuisine = r["cuisine"].split("/")[-1].strip() if r["cuisine"] else ""
        url = r["url"]
        cuisine_tag = f" · {cuisine}" if cuisine else ""
        link = f"[{name}]({url})" if url and url != "nan" else name
        lines.append(f"{i}. {link}{cuisine_tag}")
    return "\n".join(lines)


def chat(message: str, history: list[dict]):
    retrieved: list[dict] = []
    if is_recipe_query(message):
        retrieved = retrieve(message, top_k=config.TOP_K)

    messages = build_messages(message, retrieved, history)

    response = ""
    for token in stream_response(messages):
        response += token
        yield response, _recipes_markdown(retrieved)


def set_textbox(prompt: str) -> str:
    return prompt


with gr.Blocks(title="Chat Cooking") as demo:
    gr.Markdown("# Chat Cooking\nAsk anything about recipes, ingredients, or cooking techniques.")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=500, label="Conversation")
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask about a recipe or ingredient...",
                    show_label=False,
                    scale=4,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

        with gr.Column(scale=1):
            recipes_panel = gr.Markdown(
                value="_No recipes retrieved yet._",
                label="Retrieved recipes",
            )

    with gr.Row():
        for prompt in EXAMPLE_PROMPTS:
            btn = gr.Button(prompt, size="sm")
            btn.click(fn=lambda p=prompt: p, outputs=msg)

    def submit(message: str, history: list[dict]):
        if not message.strip():
            return history, "", gr.update()
        history = history + [{"role": "user", "content": message}]
        partial_response = ""
        recipes_md = "_No recipes retrieved._"
        for partial_response, recipes_md in chat(message, history[:-1]):
            yield history + [{"role": "assistant", "content": partial_response}], "", recipes_md
        yield history + [{"role": "assistant", "content": partial_response}], "", recipes_md

    send_btn.click(
        fn=submit,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg, recipes_panel],
    )
    msg.submit(
        fn=submit,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg, recipes_panel],
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
