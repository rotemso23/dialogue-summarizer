"""
src/model.py — Base model + LoRA setup for Phi-3-mini fine-tuning.

Loads microsoft/Phi-3-mini-4k-instruct in 4-bit quantization (BitsAndBytes),
applies a LoRA adapter via PEFT, and returns the ready-to-train model + tokenizer.

Call load_model_and_tokenizer() from train.py — do not import data.py from here.
"""

from __future__ import annotations

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import PreTrainedTokenizerBase

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
HUB_REPO = "rotemso23/dialogsum-phi3-lora"

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["qkv_proj", "o_proj"]


# ---------------------------------------------------------------------------
# Model + tokenizer loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    model_id: str = MODEL_ID,
    load_in_4bit: bool = True,
) -> tuple[object, PreTrainedTokenizerBase]:
    """
    Load Phi-3-mini with 4-bit quantization and apply a LoRA adapter.

    Steps:
        1. Load tokenizer with right-padding (required for causal LM training).
        2. Build BitsAndBytesConfig for 4-bit NF4 quantization with fp16 compute.
        3. Load the base model with device_map='auto' so it lands on GPU when available.
        4. Call prepare_model_for_kbit_training() to enable gradient checkpointing
           and cast layer norms to fp32 — required before applying LoRA to a
           quantized model.
        5. Apply LoraConfig targeting qkv_proj and o_proj attention projections.
        6. Return (peft_model, tokenizer).

    The returned model has ~1-2% trainable parameters (the LoRA adapter weights).
    All base model weights are frozen and kept in 4-bit.

    Args:
        model_id: HuggingFace model identifier. Defaults to Phi-3-mini-4k-instruct.
        load_in_4bit: Whether to use 4-bit quantization. Set False for CPU testing
                      (model will be large and slow, but functional for import checks).

    Returns:
        Tuple of (peft_model, tokenizer).
        peft_model: PeftModel wrapping the quantized base — ready for Trainer.
        tokenizer: AutoTokenizer with padding_side='right' and pad_token set.
    """
    # Step 1: tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "right"
    # Phi-3 tokenizer already has pad_token (<|endoftext|> / id=32000).
    # Guard in case a variant doesn't:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Step 2: 4-bit quantization config
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",           # NF4 is optimal for LLM weights
            bnb_4bit_compute_dtype=torch.float16, # fp16 compute for speed
            bnb_4bit_use_double_quant=True,       # nested quantization saves ~0.4 bits/param
        )
    else:
        bnb_config = None

    # Step 3: base model
    # trust_remote_code=False: transformers 5.x natively supports Phi-3 — using the
    # built-in implementation avoids the RoPE scaling KeyError in the custom modeling_phi3.py.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=False,
    )

    # Step 4: prepare for k-bit training
    # Enables gradient checkpointing, casts layer norms to fp32, disables cache.
    # Must be called BEFORE get_peft_model().
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    # Step 5: LoRA adapter
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    return model, tokenizer


def print_trainable_parameters(model: object) -> None:
    """
    Print the number of trainable vs. total parameters and the trainable %.

    Expected output for Phi-3-mini with r=16, target=[qkv_proj, o_proj]:
        trainable params: ~8,388,608 (8M)
        total params: ~3,821,079,552 (3.8B)
        trainable %: ~0.22%

    (Exact numbers depend on the model revision.)

    Args:
        model: A PeftModel or any nn.Module.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total
    print(f"trainable params : {trainable:,}")
    print(f"total params     : {total:,}")
    print(f"trainable %%     : {pct:.4f}%%")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    print(f"Loading model: {MODEL_ID}")
    print("(This downloads ~2.3 GB on first run; cached on subsequent runs)\n")

    model, tokenizer = load_model_and_tokenizer()

    print("\n--- Trainable parameter count ---")
    print_trainable_parameters(model)

    print("\n--- LoRA adapter summary ---")
    model.print_trainable_parameters()  # PEFT's built-in version

    print("\n--- Tokenizer ---")
    print(f"vocab size   : {tokenizer.vocab_size:,}")
    print(f"pad_token    : {tokenizer.pad_token!r}  (id={tokenizer.pad_token_id})")
    print(f"eos_token    : {tokenizer.eos_token!r}  (id={tokenizer.eos_token_id})")
    print(f"padding_side : {tokenizer.padding_side}")

    print("\nmodel.py OK — model and tokenizer ready for train.py")
