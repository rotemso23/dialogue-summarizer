"""
app.py — Gradio demo for the fine-tuned dialogue summarizer.

Entry point for HuggingFace Spaces. Loads the LoRA adapter from Hub on first
inference and exposes a simple UI: paste a conversation, get a summary.
"""

import gradio as gr

from src.infer import summarize

# ---------------------------------------------------------------------------
# Example conversation (pre-populates the input so the demo isn't blank)
# ---------------------------------------------------------------------------

EXAMPLE_DIALOGUE = """\
Amanda: I baked cookies. Do you want some?
Jerry: Sure! What kind?
Amanda: Chocolate chip. I'll bring them to the office tomorrow.
Jerry: Amazing, I can't wait. Thanks Amanda!
Amanda: No problem :)"""

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

demo = gr.Interface(
    fn=summarize,
    inputs=gr.Textbox(
        label="Conversation",
        placeholder="Paste a multi-turn conversation here...",
        lines=10,
        value=EXAMPLE_DIALOGUE,
    ),
    outputs=gr.Textbox(
        label="Summary",
        lines=4,
    ),
    title="Dialogue Summarizer",
    description=(
        "Fine-tuned Phi-3-mini with LoRA on DialogSum. "
        "Paste any messenger-style conversation and get a concise summary."
    ),
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
