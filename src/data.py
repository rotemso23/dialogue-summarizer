"""
src/data.py — Data preparation for DialogSum fine-tuning of Phi-3-mini with LoRA.

Loads DialogSum from HuggingFace Hub, formats examples with the Phi-3 chat template,
tokenizes with a 1024-token limit, and applies manual label masking so the Trainer
computes loss only on the assistant's summary tokens.

The tokenizer is NOT loaded here — it is passed in as a parameter by train.py.
"""

from __future__ import annotations

import functools
from typing import Callable

import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INSTRUCTION = "Summarize the following conversation in a few sentences."
DATASET_NAME = "knkarthick/dialogsum"
DEFAULT_MAX_LENGTH = 1024


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_example(
    example: dict,
    tokenizer: PreTrainedTokenizerBase,
) -> str:
    """
    Format a single DialogSum example into a Phi-3 chat template string.

    Uses tokenizer.apply_chat_template with role=user for the instruction+dialogue
    and role=assistant for the summary. Returns the raw formatted text (not tokenized).
    This is the string that gets printed in the CLI sanity check.

    Args:
        example: A DialogSum example dict with keys 'dialogue' and 'summary'.
        tokenizer: A Phi-3 tokenizer with apply_chat_template support.

    Returns:
        Formatted string with Phi-3 special tokens embedded.
    """
    messages = [
        {
            "role": "user",
            "content": f"{INSTRUCTION}\n\nConversation:\n{example['dialogue']}",
        },
        {
            "role": "assistant",
            "content": example["summary"],
        },
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


# ---------------------------------------------------------------------------
# Tokenization + label masking
# ---------------------------------------------------------------------------

def tokenize_and_mask(
    example: dict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> dict:
    """
    Tokenize one DialogSum example and apply prompt-masking on labels.

    Builds input_ids from the full formatted sequence (prompt + summary).
    Builds labels as a copy of input_ids with prompt positions replaced by -100,
    so cross-entropy loss is computed ONLY on the assistant's summary tokens.

    The prompt boundary is determined by tokenizing just the prompt portion
    (with add_generation_prompt=True) and using its token count as the mask cutoff.

    If the full sequence exceeds max_length, transformers truncates from the right
    (cutting the end of the dialogue, not the summary).

    If prompt_len >= len(input_ids) after truncation (extremely long dialogue edge case),
    all labels are -100 and this example contributes zero loss — this is acceptable.

    Args:
        example: A DialogSum example dict with keys 'dialogue' and 'summary'.
        tokenizer: A Phi-3 tokenizer (must have apply_chat_template, padding_side='right').
        max_length: Maximum token sequence length. Defaults to 1024.

    Returns:
        Dict with keys:
            - 'input_ids': List[int], length <= max_length
            - 'attention_mask': List[int], all 1s (pre-padding)
            - 'labels': List[int], same length as input_ids, prompt positions are -100
    """
    # Step 1: build full text (prompt + summary)
    messages_full = [
        {
            "role": "user",
            "content": f"{INSTRUCTION}\n\nConversation:\n{example['dialogue']}",
        },
        {
            "role": "assistant",
            "content": example["summary"],
        },
    ]
    full_text: str = tokenizer.apply_chat_template(messages_full, tokenize=False)

    # Step 2: build prompt-only text (up to and including <|assistant|>\n)
    messages_prompt = [
        {
            "role": "user",
            "content": f"{INSTRUCTION}\n\nConversation:\n{example['dialogue']}",
        },
    ]
    prompt_text: str = tokenizer.apply_chat_template(
        messages_prompt,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Step 3: tokenize full sequence with truncation
    enc = tokenizer(
        full_text,
        add_special_tokens=False,
        max_length=max_length,
        truncation=True,
    )
    input_ids: list[int] = enc["input_ids"]
    attention_mask: list[int] = enc["attention_mask"]

    # Step 4: tokenize prompt only to find the boundary
    prompt_enc = tokenizer(
        prompt_text,
        add_special_tokens=False,
        max_length=max_length,
        truncation=True,
    )
    prompt_len: int = len(prompt_enc["input_ids"])

    # Step 5: clamp and build labels
    # Guard: if an extremely long dialogue was truncated so hard that the summary
    # was completely cut off, prompt_len could equal len(input_ids).
    # In that case all labels are -100 (zero loss contribution). Acceptable.
    prompt_len = min(prompt_len, len(input_ids))
    labels: list[int] = [-100] * prompt_len + input_ids[prompt_len:]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def prepare_datasets(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = DEFAULT_MAX_LENGTH,
    dataset_name: str = DATASET_NAME,
    num_proc: int = 4,
) -> tuple[Dataset, Dataset, Dataset]:
    """
    Load DialogSum, format, tokenize, and return train/val/test splits.

    Pipeline:
        1. load_dataset(dataset_name) — uses HF Hub, returns DatasetDict with
           train (~14732), validation (~818), test (~819) splits.
        2. Dataset.map(tokenize_and_mask, ...) on each split.
        3. Remove original 'id', 'dialogue', 'summary' columns.

    The returned datasets contain only 'input_ids', 'attention_mask', 'labels'.
    They have variable sequence lengths (no padding) — pass the collator returned
    by make_data_collator() to the Trainer's data_collator argument.

    Args:
        tokenizer: Loaded and configured Phi-3 tokenizer. Caller must ensure
                   tokenizer.padding_side == 'right' for training.
        max_length: Max token length for truncation. Defaults to 1024.
        dataset_name: HuggingFace dataset identifier. Defaults to 'knkarthick/dialogsum'.
        num_proc: Number of processes for Dataset.map. Defaults to 4.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    raw: DatasetDict = load_dataset(dataset_name)

    _map_fn = functools.partial(
        tokenize_and_mask,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    original_columns = raw["train"].column_names  # ['id', 'dialogue', 'summary']

    tokenized: DatasetDict = raw.map(
        _map_fn,
        batched=False,
        num_proc=num_proc,
        remove_columns=original_columns,
        desc="Tokenizing and masking labels",
    )

    return tokenized["train"], tokenized["validation"], tokenized["test"]


# ---------------------------------------------------------------------------
# Collator (padding)
# ---------------------------------------------------------------------------

def make_data_collator(
    tokenizer: PreTrainedTokenizerBase,
) -> Callable[[list[dict]], dict[str, torch.Tensor]]:
    """
    Return a collate function that pads a batch of tokenized examples.

    Pads 'input_ids' to the longest sequence in the batch using tokenizer.pad_token_id.
    Pads 'attention_mask' with 0.
    Pads 'labels' with -100 (ignored by cross-entropy loss).
    Padding is applied on the RIGHT (required for causal LM training).

    This collator is intentionally minimal — it does NOT do label shifting,
    DataCollatorForSeq2Seq masking, or any other transformation. All label
    masking was already done in tokenize_and_mask().

    Args:
        tokenizer: Must have pad_token_id set (Phi-3: <|endoftext|>, id=32000).

    Returns:
        A collate_fn compatible with HuggingFace Trainer's data_collator argument.
    """
    pad_id: int = tokenizer.pad_token_id

    def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in batch)

        input_ids_padded = []
        attention_mask_padded = []
        labels_padded = []

        for item in batch:
            seq_len = len(item["input_ids"])
            pad_len = max_len - seq_len

            input_ids_padded.append(item["input_ids"] + [pad_id] * pad_len)
            attention_mask_padded.append(item["attention_mask"] + [0] * pad_len)
            labels_padded.append(item["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids_padded, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_padded, dtype=torch.long),
            "labels": torch.tensor(labels_padded, dtype=torch.long),
        }

    return collate_fn


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    from dotenv import load_dotenv
    from transformers import AutoTokenizer

    load_dotenv()  # loads HF_TOKEN from .env if present

    MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
    print(f"Loading tokenizer: {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.padding_side = "right"  # required for training

    print(f"Loading dataset: {DATASET_NAME} ...")
    raw = load_dataset(DATASET_NAME)
    print("\nSplit sizes:")
    print(f"  train:      {len(raw['train']):>6,}")
    print(f"  validation: {len(raw['validation']):>6,}")
    print(f"  test:       {len(raw['test']):>6,}")

    # Show one formatted example
    example = raw["train"][0]
    formatted = format_example(example, tokenizer)
    print("\n--- Formatted example (raw chat template text) ---")
    print(formatted)

    # Sanity check: label masking
    tokenized_example = tokenize_and_mask(example, tokenizer)
    non_masked_ids = [t for t in tokenized_example["labels"] if t != -100]
    decoded = tokenizer.decode(non_masked_ids, skip_special_tokens=True)

    print("\n--- Sanity check: decoded labels vs. original summary ---")
    print(f"Original summary : {example['summary']!r}")
    print(f"Decoded labels   : {decoded!r}")
    match = decoded.strip() == example["summary"].strip()
    print(f"Match            : {match}")
    if not match:
        print("WARNING: Mismatch detected. Check tokenize_and_mask boundary logic.")

    # Token length stats on a random sample of 500 train examples
    sample = random.sample(list(raw["train"]), min(500, len(raw["train"])))
    lengths = []
    for ex in sample:
        enc = tokenize_and_mask(ex, tokenizer)
        lengths.append(len(enc["input_ids"]))

    print(f"\n--- Token length stats (sample n={len(sample)}) ---")
    print(f"  min:    {min(lengths)}")
    print(f"  max:    {max(lengths)}")
    print(f"  mean:   {sum(lengths) / len(lengths):.1f}")
    over_limit = sum(1 for length in lengths if length >= DEFAULT_MAX_LENGTH)
    print(f"  >= {DEFAULT_MAX_LENGTH} (truncated): {over_limit} ({100 * over_limit / len(lengths):.1f}%)")
