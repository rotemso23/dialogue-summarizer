"""
src/infer.py — Single-conversation summarization using the fine-tuned LoRA adapter.

Exposes one public function: summarize(dialogue) -> str.
The model is loaded from HuggingFace Hub on first call and cached for all subsequent calls.

This module is imported by app.py (Gradio demo). Keep it clean and import-safe.
"""

from __future__ import annotations

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.data import INSTRUCTION
from src.model import HUB_REPO, MODEL_ID

# ---------------------------------------------------------------------------
# Module-level cache — loaded once on first call to summarize()
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None


def _load() -> tuple:
    """
    Load the tokenizer and fine-tuned model if not already cached.

    Loads Phi-3-mini and applies the LoRA adapter from Hub.
    Uses 4-bit quantization on GPU, float16 on CPU.
    Results are stored in module-level globals so subsequent calls are instant.

    Returns:
        Tuple of (model, tokenizer).
    """
    global _model, _tokenizer

    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=False,
            dtype=torch.float16,
        )
    else:
        # CPU fallback (HF Spaces free tier) — no quantization, float32
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.float32,
            trust_remote_code=False,
        )
    model = PeftModel.from_pretrained(base, HUB_REPO)
    model.eval()

    _model = model
    _tokenizer = tokenizer
    return _model, _tokenizer


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def summarize(dialogue: str) -> str:
    """
    Summarize a multi-turn conversation using the fine-tuned LoRA adapter.

    Loads the model from HuggingFace Hub on first call (cached for subsequent calls).
    Uses greedy decoding for reproducible outputs.

    Args:
        dialogue: A multi-turn conversation string (e.g. "Alice: Hi\nBob: Hello").

    Returns:
        A plain string containing the generated summary.
    """
    model, tokenizer = _load()

    messages = [
        {"role": "user", "content": f"{INSTRUCTION}\n\nConversation:\n{dialogue}"}
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = output_ids[0][input_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_dialogue = (
        "Amanda: I baked cookies. Do you want some?\n"
        "Jerry: Sure! What kind?\n"
        "Amanda: Chocolate chip. I'll bring them to the office tomorrow.\n"
        "Jerry: Amazing, I can't wait. Thanks Amanda!"
    )

    print("Dialogue:")
    print(test_dialogue)
    print("\nGenerating summary...")
    result = summarize(test_dialogue)
    print(f"\nSummary:\n{result}")
