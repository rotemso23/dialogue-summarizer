"""
src/evaluate.py — ROUGE evaluation: fine-tuned vs. zero-shot baseline on DialogSum test split.

Loads the fine-tuned LoRA adapter from HuggingFace Hub and the base model (no adapter),
runs greedy inference on the 819-example test split, computes ROUGE-1/2/L, and saves
results to evaluation_results.json.

Run on Colab T4:
    python src/evaluate.py
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import torch
from datasets import load_dataset
from peft import PeftModel
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.data import DATASET_NAME, INSTRUCTION
from src.model import HUB_REPO, MODEL_ID

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZE = 4
MAX_NEW_TOKENS = 128
NUM_QUALITATIVE = 5


# ---------------------------------------------------------------------------
# Prompt formatting (inference only — user turn, no assistant content)
# ---------------------------------------------------------------------------

def format_inference_prompt(dialogue: str, tokenizer: Any) -> str:
    """
    Format a dialogue into an inference prompt (user turn only).

    Uses add_generation_prompt=True so the model continues with the assistant turn.
    This is the inference-time counterpart of tokenize_and_mask's prompt_text.

    Args:
        dialogue: Raw conversation string from the dataset.
        tokenizer: Phi-3 tokenizer with apply_chat_template support.

    Returns:
        Prompt string ending with the assistant generation trigger token.
    """
    messages = [
        {"role": "user", "content": f"{INSTRUCTION}\n\nConversation:\n{dialogue}"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _load_tokenizer(model_id: str = MODEL_ID) -> Any:
    """Load tokenizer with left-padding (required for batched generation)."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_base_model(model_id: str = MODEL_ID) -> Any:
    """Load Phi-3-mini in 4-bit quantization without any LoRA adapter."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=False,
        dtype=torch.float16,
    )
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    model: Any,
    tokenizer: Any,
    dialogues: list[str],
    batch_size: int = BATCH_SIZE,
) -> list[str]:
    """
    Run batched greedy inference on a list of dialogues.

    Formats each dialogue into an inference prompt, tokenizes in batches with
    left-padding, generates with max_new_tokens=128 and do_sample=False, then
    strips the prompt prefix from each output to return only the generated summary.

    Args:
        model: Loaded causal LM (base model or PeftModel).
        tokenizer: Matching tokenizer with padding_side='left'.
        dialogues: List of raw dialogue strings.
        batch_size: Number of examples per forward pass.

    Returns:
        List of generated summary strings, one per dialogue.
    """
    prompts = [format_inference_prompt(d, tokenizer) for d in dialogues]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_summaries: list[str] = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Inferring"):
        batch_prompts = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        for out in output_ids:
            generated_ids = out[input_len:]
            summary = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            all_summaries.append(summary)

    return all_summaries


# ---------------------------------------------------------------------------
# ROUGE scoring
# ---------------------------------------------------------------------------

def compute_rouge(predictions: list[str], references: list[str]) -> dict[str, float]:
    """
    Compute average ROUGE-1, ROUGE-2, and ROUGE-L F-scores.

    Args:
        predictions: Generated summaries (one per test example).
        references: Ground-truth summaries from the dataset.

    Returns:
        Dict with keys 'rouge1', 'rouge2', 'rougeL' — mean F-scores in [0, 1].
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals: dict[str, float] = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        totals["rouge1"] += scores["rouge1"].fmeasure
        totals["rouge2"] += scores["rouge2"].fmeasure
        totals["rougeL"] += scores["rougeL"].fmeasure

    n = len(predictions)
    return {k: v / n for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Qualitative display
# ---------------------------------------------------------------------------

def print_qualitative_examples(
    dialogues: list[str],
    references: list[str],
    finetuned_preds: list[str],
    baseline_preds: list[str],
    n: int = NUM_QUALITATIVE,
) -> None:
    """Print n side-by-side examples: dialogue, reference, fine-tuned, baseline."""
    print("\n" + "=" * 80)
    print(f"QUALITATIVE EXAMPLES (n={n})")
    print("=" * 80)
    for i in range(n):
        print(f"\n--- Example {i + 1} ---")
        print(f"[Dialogue]\n{dialogues[i]}\n")
        print(f"[Reference]\n{references[i]}\n")
        print(f"[Fine-tuned]\n{finetuned_preds[i]}\n")
        print(f"[Baseline]\n{baseline_preds[i]}\n")
        print("-" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()

    print("Loading DialogSum test split...")
    test_data = load_dataset(DATASET_NAME, split="test")
    dialogues: list[str] = test_data["dialogue"]
    references: list[str] = test_data["summary"]
    print(f"Test examples: {len(dialogues)}")

    tokenizer = _load_tokenizer()

    # --- Fine-tuned model ---
    print(f"\nLoading fine-tuned model from Hub: {HUB_REPO}")
    base_model = _load_base_model()
    finetuned_model = PeftModel.from_pretrained(base_model, HUB_REPO)
    finetuned_model.eval()

    print("Running fine-tuned inference...")
    finetuned_preds = run_inference(finetuned_model, tokenizer, dialogues)

    finetuned_rouge = compute_rouge(finetuned_preds, references)
    print("\nFine-tuned ROUGE scores:")
    for k, v in finetuned_rouge.items():
        print(f"  {k}: {v:.4f}")

    # Free GPU memory before loading the baseline
    del finetuned_model
    del base_model
    torch.cuda.empty_cache()

    # --- Baseline model (no adapter) ---
    print(f"\nLoading baseline model (no adapter): {MODEL_ID}")
    baseline_model = _load_base_model()

    print("Running baseline inference...")
    baseline_preds = run_inference(baseline_model, tokenizer, dialogues)

    baseline_rouge = compute_rouge(baseline_preds, references)
    print("\nBaseline ROUGE scores:")
    for k, v in baseline_rouge.items():
        print(f"  {k}: {v:.4f}")

    del baseline_model
    torch.cuda.empty_cache()

    # --- Results table ---
    print("\n" + "=" * 52)
    print(f"{'Metric':<12} {'Baseline':>10} {'Fine-tuned':>12} {'Delta':>10}")
    print("-" * 52)
    for k in ["rouge1", "rouge2", "rougeL"]:
        base_val = baseline_rouge[k]
        ft_val = finetuned_rouge[k]
        delta = ft_val - base_val
        print(f"{k:<12} {base_val:>10.4f} {ft_val:>12.4f} {delta:>+10.4f}")
    print("=" * 52)

    # --- Save results ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"evaluation_results_{timestamp}.json"
    results = {
        "timestamp": timestamp,
        "fine_tuned": finetuned_rouge,
        "baseline": baseline_rouge,
    }
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_file}")

    # --- Qualitative examples ---
    print_qualitative_examples(dialogues, references, finetuned_preds, baseline_preds)


if __name__ == "__main__":
    main()
